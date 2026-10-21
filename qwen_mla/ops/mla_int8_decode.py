"""Experimental single-pass INT8 MLA decode ("int8-tok" cache). Not used by prod: nothing imports this module.

Cache page layout (one page = PAGE tokens, PAGE % BN == 0, bytes, structure-of-arrays inside the page):
    [0,            PAGE*R)          latent  int8 [PAGE, R]      round(c / sigma_t), sigma_t = absmax(c_t) / 127
    [PAGE*R,       PAGE*(R+256))    rope    int8 [PAGE, 256]    4 KV groups x 58 dims, each padded to 64, scale rho_t
    [.. + 0,       .. + 2*PAGE)     sigma   fp16 [PAGE]
    [.. + 2*PAGE,  .. + 4*PAGE)     rho     fp16 [PAGE]
    [.. + 4*PAGE,  .. + 28*PAGE)    rms     uint8 [24, PAGE]    log8 code of the per-(head, token) rms divisor
The page is a contiguous byte block, so it pages, evicts and prefix-caches like today's [PAGE, width] rows.

Queries: per request 4 MTP rows (lens L-3..L, causal) x 24 heads. q is quantized per (head, row) to int8 every
step (prep_q); head h uses only rope slot g = h // 6. Scores factor exactly:
    s = (qs * sigma_t * int(q8 . c8) + qrs * rho_t * int(qr8 . r8)) / rms[h, t]        (sm_scale*log2e in qs, qrs)
One CTA = (head block of HB heads x 4 rows, kv split, request); head blocks of one request run back to back so
the key tile they share is read from DRAM once. Each CTA keeps the full [M, R] fp32 accumulator (R in DC chunks).
PV: P (with sigma folded in, per-tile normalised) in fp16 x V int8->fp16, fp16-accumulate MMA per tile, added
into fp32 (PV_INT8=0), or P quantized to int8 per (row, tile) x V int8 -> int32 (PV_INT8=1).
"""
import math

import torch
import triton
import triton.language as tl

HEADS, ROWS, GROUPS, RG, RGP = 24, 4, 4, 58, 64
ROPE_PAD = GROUPS * RGP                       # 256
LOG2E = 1.4426950408889634


def token_bytes(rank):
    return rank + ROPE_PAD + 4 + HEADS


def page_bytes(rank, page):
    return page * token_bytes(rank)


# ------------------------------------------------------------------------------------------------ host packing
def pack_cache(lat, rp, rms):
    """lat [NP, P, R], rp [NP, P, 232], rms [NP, P, 24] (fp32) -> uint8 cache [NP, P * token_bytes(R)]."""
    NP, P, R = lat.shape
    lat, rp = lat.float(), rp.float()
    sig = (lat.abs().amax(-1) / 127).half()
    sig = torch.where(sig > 0, sig, torch.ones_like(sig))
    li = (lat / sig.float()[..., None]).round().clamp(-127, 127).to(torch.int8)
    rpp = torch.zeros(NP, P, GROUPS, RGP, device=lat.device)
    rpp[..., :RG] = rp.view(NP, P, GROUPS, RG)
    rpp = rpp.view(NP, P, ROPE_PAD)
    rho = (rpp.abs().amax(-1) / 127).half()
    rho = torch.where(rho > 0, rho, torch.ones_like(rho))
    ri = (rpp / rho.float()[..., None]).round().clamp(-127, 127).to(torch.int8)
    code = ((rms.float().log2() + 3.0) * (255.0 / 6.0)).round().clamp(0, 255).to(torch.uint8)
    buf = torch.empty(NP, page_bytes(R, P), dtype=torch.uint8, device=lat.device)
    o = 0
    for part in (li.view(torch.uint8), ri.view(torch.uint8), sig.view(torch.uint8), rho.view(torch.uint8),
                 code.transpose(1, 2).contiguous()):
        n = part[0].numel()
        buf[:, o:o + n] = part.reshape(NP, n)
        o += n
    assert o == buf.shape[1]
    return buf


def unpack_cache(buf, R, P):
    """Dequantize: -> lat [NP, P, R], rope [NP, P, 232], rms [NP, P, 24] (fp32, rms decoded from log8)."""
    NP = buf.shape[0]
    o = 0

    def take(n):
        nonlocal o
        t = buf[:, o:o + n]
        o += n
        return t
    li = take(P * R).view(torch.int8).reshape(NP, P, R).float()
    ri = take(P * ROPE_PAD).view(torch.int8).reshape(NP, P, GROUPS, RGP)[..., :RG].reshape(NP, P, GROUPS * RG).float()
    sig = take(2 * P).contiguous().view(torch.float16).float()
    rho = take(2 * P).contiguous().view(torch.float16).float()
    code = take(HEADS * P).reshape(NP, HEADS, P).transpose(1, 2).float()
    return li * sig[..., None], ri * rho[..., None], torch.exp2(-3.0 + code * (6.0 / 255.0))


def prep_q_ref(q, rank, sm_scale):
    """q [B_rows, 24, rank + 232] (rows 4i..4i+3 = request i's MTP rows; rope packed, head h uses slot h // 6).

    -> q8 [nreq, 24, 4, rank] int8, qs [nreq, 24, 4] fp32, qr8 [nreq, 24, 4, 256] int8 (own slot only), qrs.
    qs / qrs carry sm_scale * log2(e) so the kernel works in exp2.
    """
    B = q.shape[0]
    nreq = B // ROWS
    qf = q.float().view(nreq, ROWS, HEADS, -1).transpose(1, 2)              # [nreq, 24, 4, D]
    ql = qf[..., :rank]
    qs = ql.abs().amax(-1) / 127
    qs = torch.where(qs > 0, qs, torch.ones_like(qs))
    q8 = (ql / qs[..., None]).round().clamp(-127, 127).to(torch.int8).contiguous()
    rpk = qf[..., rank:rank + GROUPS * RG].reshape(nreq, HEADS, ROWS, GROUPS, RG)
    g = torch.arange(HEADS, device=q.device) // 6
    qr = rpk[:, torch.arange(HEADS, device=q.device), :, g]                  # [24, nreq, 4, 58]
    qr = qr.permute(1, 0, 2, 3)
    qrs = qr.abs().amax(-1) / 127
    qrs = torch.where(qrs > 0, qrs, torch.ones_like(qrs))
    qr8 = torch.zeros(nreq, HEADS, ROWS, GROUPS, RGP, dtype=torch.int8, device=q.device)
    qr8[torch.arange(nreq, device=q.device)[:, None], torch.arange(HEADS, device=q.device)[None, :], :, g[None, :], :RG] = \
        (qr / qrs[..., None]).round().clamp(-127, 127).to(torch.int8)
    k = sm_scale * LOG2E
    return q8, (qs * k).contiguous(), qr8.view(nreq, HEADS, ROWS, ROPE_PAD).contiguous(), (qrs * k).contiguous()


# ------------------------------------------------------------------------------------------------ kernels
@triton.jit
def _prep_q_kernel(Q, Q8, QS, QR8, QRS, stride_qb, stride_qh, k, R: tl.constexpr, RP: tl.constexpr):
    b = tl.program_id(0)                  # query row (request b // 4, MTP row b % 4)
    h = tl.program_id(1)
    req = b // 4
    j = b % 4
    orow = (req * 24 + h) * 4 + j
    d = tl.arange(0, RP)
    x = tl.load(Q + b * stride_qb + h * stride_qh + d, mask=d < R, other=0.0).to(tl.float32)
    s = tl.max(tl.abs(x), 0) / 127.0
    s = tl.where(s > 0, s, 1.0)
    tl.store(Q8 + orow * R + d, tl.extra.cuda.libdevice.rint(x / s).to(tl.int8), mask=d < R)
    tl.store(QS + orow, s * k)
    g = h // 6
    e = tl.arange(0, 64)
    xr = tl.load(Q + b * stride_qb + h * stride_qh + R + g * 58 + e, mask=e < 58, other=0.0).to(tl.float32)
    sr = tl.max(tl.abs(xr), 0) / 127.0
    sr = tl.where(sr > 0, sr, 1.0)
    w = tl.arange(0, 256)
    qv = tl.extra.cuda.libdevice.rint(xr / sr).to(tl.int8)
    tl.store(QR8 + orow * 256 + w, tl.zeros([256], tl.int8))
    tl.debug_barrier()
    tl.store(QR8 + orow * 256 + g * 64 + e, qv)
    tl.store(QRS + orow, sr * k)



def prep_q(q, rank, sm_scale, out=None):
    """Triton version of prep_q_ref (same outputs up to 1 LSB from division rounding); out = preallocated tuple."""
    B = q.shape[0]
    nreq = B // 4
    dev = q.device
    if out is None:
        out = (torch.empty(nreq, 24, 4, rank, dtype=torch.int8, device=dev), torch.empty(nreq, 24, 4, dtype=torch.float32, device=dev),
               torch.empty(nreq, 24, 4, 256, dtype=torch.int8, device=dev), torch.empty(nreq, 24, 4, dtype=torch.float32, device=dev))
    q8, qs, qr8, qrs = out
    _prep_q_kernel[(B, 24)](q, q8, qs, qr8, qrs, q.stride(0), q.stride(1), sm_scale * LOG2E, R=rank,
                            RP=triton.next_power_of_2(rank), num_warps=4)
    return out


@triton.jit
def _ldk(pb, tok, c: tl.constexpr, R: tl.constexpr, DC: tl.constexpr):
    dc = tl.arange(0, DC)
    return tl.load(pb + tok[:, None] * R + (c * DC + dc)[None, :])


@triton.jit
def _qk_chunk(q, pb, tok, acc, c: tl.constexpr, R: tl.constexpr, DC: tl.constexpr):
    dc = tl.arange(0, DC)
    k = tl.load(pb + tok[:, None] * R + (c * DC + dc)[None, :])
    return k, tl.dot(q, tl.trans(k), acc, out_dtype=tl.int32)


@triton.jit
def _pv_chunk(acc, k, pw, alpha, rsc, PV_INT8: tl.constexpr):
    if PV_INT8:
        t = tl.dot(pw, k, out_dtype=tl.int32).to(tl.float32) * rsc[:, None]
    else:
        t = tl.dot(pw, k.to(tl.float16), out_dtype=tl.float16).to(tl.float32) * rsc[:, None]
    return (acc.to(tl.float32) * alpha[:, None] + t).to(acc.dtype)


@triton.jit
def _store_chunk(Opart, obase, acc, linv, rvalid, c: tl.constexpr, DC: tl.constexpr):
    dc = tl.arange(0, DC)
    tl.store(Opart + obase[:, None] + (c * DC + dc)[None, :], acc.to(tl.float32) * linv[:, None], mask=rvalid[:, None])


@triton.jit
def _int8_decode_kernel(
    Q8, QS, QR8, QRS, Cache, BT, Lens, Opart, LSEpart, stride_bt,
    R: tl.constexpr, PAGE: tl.constexpr, HB: tl.constexpr, M: tl.constexpr, BN: tl.constexpr, DC: tl.constexpr,
    NC: tl.constexpr, RW: tl.constexpr, NSPLIT: tl.constexpr, PV_INT8: tl.constexpr, PSCALE: tl.constexpr,
    ACC16: tl.constexpr, EARLY: tl.constexpr, QIN: tl.constexpr,
):
    hb = tl.program_id(0)
    split = tl.program_id(1)
    req = tl.program_id(2)
    rows = tl.arange(0, M)
    hl = rows // 4
    j = rows % 4
    head = hb * HB + hl
    rvalid = (hl < HB) & (head < 24)
    L = tl.load(Lens + req)
    rlen = L - 3 + j
    per = tl.cdiv(tl.cdiv(L, NSPLIT), BN) * BN
    start = split * per
    end = tl.minimum(start + per, L)

    qrow = (req * 24 + head) * 4 + j
    dc = tl.arange(0, DC)
    q0 = tl.load(Q8 + qrow[:, None] * R + dc[None, :], mask=rvalid[:, None], other=0)
    if NC > 1:
        q1 = tl.load(Q8 + qrow[:, None] * R + (DC + dc)[None, :], mask=rvalid[:, None], other=0)
    if NC > 2:
        q2 = tl.load(Q8 + qrow[:, None] * R + (2 * DC + dc)[None, :], mask=rvalid[:, None], other=0)
    if NC > 3:
        q3 = tl.load(Q8 + qrow[:, None] * R + (3 * DC + dc)[None, :], mask=rvalid[:, None], other=0)
    if NC > 4:
        q4 = tl.load(Q8 + qrow[:, None] * R + (4 * DC + dc)[None, :], mask=rvalid[:, None], other=0)
    if NC > 5:
        q5 = tl.load(Q8 + qrow[:, None] * R + (5 * DC + dc)[None, :], mask=rvalid[:, None], other=0)
    if NC > 6:
        q6 = tl.load(Q8 + qrow[:, None] * R + (6 * DC + dc)[None, :], mask=rvalid[:, None], other=0)
    qs = tl.load(QS + qrow, mask=rvalid, other=0.0)
    wstart = tl.minimum(((hb * HB) // 6) * 64, 256 - RW)
    rw = tl.arange(0, RW)
    qr = tl.load(QR8 + qrow[:, None] * 256 + wstart + rw[None, :], mask=rvalid[:, None], other=0)
    qrs = tl.load(QRS + qrow, mask=rvalid, other=0.0)

    m_i = tl.full([M], float("-inf"), tl.float32)
    l_i = tl.zeros([M], tl.float32)
    acc0 = tl.zeros([M, DC], tl.float16 if ACC16 else tl.float32)
    if NC > 1:
        acc1 = tl.zeros([M, DC], tl.float16 if ACC16 else tl.float32)
    if NC > 2:
        acc2 = tl.zeros([M, DC], tl.float16 if ACC16 else tl.float32)
    if NC > 3:
        acc3 = tl.zeros([M, DC], tl.float16 if ACC16 else tl.float32)
    if NC > 4:
        acc4 = tl.zeros([M, DC], tl.float16 if ACC16 else tl.float32)
    if NC > 5:
        acc5 = tl.zeros([M, DC], tl.float16 if ACC16 else tl.float32)
    if NC > 6:
        acc6 = tl.zeros([M, DC], tl.float16 if ACC16 else tl.float32)
    tn = tl.arange(0, BN)
    ROPE_OFF: tl.constexpr = PAGE * R
    SIG_OFF: tl.constexpr = PAGE * (R + 256)
    RHO_OFF: tl.constexpr = SIG_OFF + 2 * PAGE
    RMS_OFF: tl.constexpr = RHO_OFF + 2 * PAGE
    PB: tl.constexpr = PAGE * (R + 256 + 4 + 24)
    for start_n in tl.range(start, end, BN):
        page = tl.load(BT + req * stride_bt + start_n // PAGE).to(tl.int64)
        pb = Cache + page * PB
        tok = (start_n % PAGE) + tn
        kpos = start_n + tn
        kin = kpos < end
        if EARLY:   # issue every chunk's load before the first dot (more bytes in flight per tile)
            k0 = _ldk(pb, tok, 0, R, DC)
            if NC > 1:
                k1 = _ldk(pb, tok, 1, R, DC)
            if NC > 2:
                k2 = _ldk(pb, tok, 2, R, DC)
            if NC > 3:
                k3 = _ldk(pb, tok, 3, R, DC)
            if NC > 4:
                k4 = _ldk(pb, tok, 4, R, DC)
            if NC > 5:
                k5 = _ldk(pb, tok, 5, R, DC)
            if NC > 6:
                k6 = _ldk(pb, tok, 6, R, DC)
            kr = tl.load(pb + ROPE_OFF + tok[:, None] * 256 + wstart + rw[None, :])
            sig = tl.load((pb + SIG_OFF).to(tl.pointer_type(tl.float16)) + tok)
            rho = tl.load((pb + RHO_OFF).to(tl.pointer_type(tl.float16)) + tok)
            code = tl.load((pb + RMS_OFF).to(tl.pointer_type(tl.uint8)) + head[:, None] * PAGE + tok[None, :],
                           mask=rvalid[:, None], other=0)
            qk = tl.dot(q0, tl.trans(k0), out_dtype=tl.int32)
            if NC > 1:
                qk = tl.dot(q1, tl.trans(k1), qk, out_dtype=tl.int32)
            if NC > 2:
                qk = tl.dot(q2, tl.trans(k2), qk, out_dtype=tl.int32)
            if NC > 3:
                qk = tl.dot(q3, tl.trans(k3), qk, out_dtype=tl.int32)
            if NC > 4:
                qk = tl.dot(q4, tl.trans(k4), qk, out_dtype=tl.int32)
            if NC > 5:
                qk = tl.dot(q5, tl.trans(k5), qk, out_dtype=tl.int32)
            if NC > 6:
                qk = tl.dot(q6, tl.trans(k6), qk, out_dtype=tl.int32)
        else:
            if QIN:   # re-read q every tile (L1/L2 hit) instead of keeping it resident in shared memory
                qp = Q8 + (start_n >> 30) + qrow[:, None] * R + dc[None, :]   # loop-variant offset (always 0)
                q0 = tl.load(qp, mask=rvalid[:, None], other=0)
                if NC > 1:
                    q1 = tl.load(qp + DC, mask=rvalid[:, None], other=0)
                if NC > 2:
                    q2 = tl.load(qp + 2 * DC, mask=rvalid[:, None], other=0)
                if NC > 3:
                    q3 = tl.load(qp + 3 * DC, mask=rvalid[:, None], other=0)
                if NC > 4:
                    q4 = tl.load(qp + 4 * DC, mask=rvalid[:, None], other=0)
                if NC > 5:
                    q5 = tl.load(qp + 5 * DC, mask=rvalid[:, None], other=0)
                if NC > 6:
                    q6 = tl.load(qp + 6 * DC, mask=rvalid[:, None], other=0)
            k0, qk = _qk_chunk(q0, pb, tok, tl.zeros([M, BN], tl.int32), 0, R, DC)
            if NC > 1:
                k1, qk = _qk_chunk(q1, pb, tok, qk, 1, R, DC)
            if NC > 2:
                k2, qk = _qk_chunk(q2, pb, tok, qk, 2, R, DC)
            if NC > 3:
                k3, qk = _qk_chunk(q3, pb, tok, qk, 3, R, DC)
            if NC > 4:
                k4, qk = _qk_chunk(q4, pb, tok, qk, 4, R, DC)
            if NC > 5:
                k5, qk = _qk_chunk(q5, pb, tok, qk, 5, R, DC)
            if NC > 6:
                k6, qk = _qk_chunk(q6, pb, tok, qk, 6, R, DC)
        if not EARLY:
            kr = tl.load(pb + ROPE_OFF + tok[:, None] * 256 + wstart + rw[None, :])
            sig = tl.load((pb + SIG_OFF).to(tl.pointer_type(tl.float16)) + tok)
            rho = tl.load((pb + RHO_OFF).to(tl.pointer_type(tl.float16)) + tok)
            code = tl.load((pb + RMS_OFF).to(tl.pointer_type(tl.uint8)) + head[:, None] * PAGE + tok[None, :],
                           mask=rvalid[:, None], other=0)
        qkr = tl.dot(qr, tl.trans(kr), out_dtype=tl.int32)
        sig = tl.where(kin, sig.to(tl.float32), 0.0)
        rho = rho.to(tl.float32)
        inv = tl.exp2(3.0 - code.to(tl.float32) * (6.0 / 255.0))
        s = (qk.to(tl.float32) * (qs[:, None] * sig[None, :]) + qkr.to(tl.float32) * (qrs[:, None] * rho[None, :])) * inv
        s = tl.where(kpos[None, :] < rlen[:, None], s, float("-inf"))
        m_new = tl.maximum(m_i, tl.max(s, 1))
        m_s = tl.where(m_new == float("-inf"), 0.0, m_new)
        alpha = tl.exp2(m_i - m_s)
        p = tl.exp2(s - m_s[:, None])
        l_new = l_i * alpha + tl.sum(p, 1)
        if ACC16:   # fp16 acc holds the normalised output (bounded by max |latent|), so it cannot overflow
            linv_t = tl.where(l_new > 0, 1.0 / l_new, 0.0)
            alpha = l_i * alpha * linv_t
        l_i = l_new
        smax = tl.max(sig, 0)
        sn = sig * tl.where(smax > 0, 1.0 / smax, 0.0)
        pw = p * sn[None, :]
        if PV_INT8:
            rmax = tl.max(pw, 1)
            pw = (pw * tl.where(rmax > 0, 127.0 / rmax, 0.0)[:, None] + 0.5).to(tl.int8)
            rsc = rmax * (smax / 127.0)
        else:
            pw = (pw * PSCALE).to(tl.float16)
            rsc = tl.zeros([M], tl.float32) + smax / PSCALE
        if ACC16:
            rsc = rsc * linv_t
        acc0 = _pv_chunk(acc0, k0, pw, alpha, rsc, PV_INT8)
        if NC > 1:
            acc1 = _pv_chunk(acc1, k1, pw, alpha, rsc, PV_INT8)
        if NC > 2:
            acc2 = _pv_chunk(acc2, k2, pw, alpha, rsc, PV_INT8)
        if NC > 3:
            acc3 = _pv_chunk(acc3, k3, pw, alpha, rsc, PV_INT8)
        if NC > 4:
            acc4 = _pv_chunk(acc4, k4, pw, alpha, rsc, PV_INT8)
        if NC > 5:
            acc5 = _pv_chunk(acc5, k5, pw, alpha, rsc, PV_INT8)
        if NC > 6:
            acc6 = _pv_chunk(acc6, k6, pw, alpha, rsc, PV_INT8)
        m_i = m_new

    has = l_i > 0
    linv = tl.where(has, 1.0 / l_i, 0.0)
    if ACC16:
        linv = tl.where(has, 1.0, 0.0)
    obase = (qrow.to(tl.int64) * NSPLIT + split) * R
    _store_chunk(Opart, obase, acc0, linv, rvalid, 0, DC)
    if NC > 1:
        _store_chunk(Opart, obase, acc1, linv, rvalid, 1, DC)
    if NC > 2:
        _store_chunk(Opart, obase, acc2, linv, rvalid, 2, DC)
    if NC > 3:
        _store_chunk(Opart, obase, acc3, linv, rvalid, 3, DC)
    if NC > 4:
        _store_chunk(Opart, obase, acc4, linv, rvalid, 4, DC)
    if NC > 5:
        _store_chunk(Opart, obase, acc5, linv, rvalid, 5, DC)
    if NC > 6:
        _store_chunk(Opart, obase, acc6, linv, rvalid, 6, DC)
    lse = tl.where(has, m_i + tl.log2(tl.where(has, l_i, 1.0)), float("-inf"))
    tl.store(LSEpart + qrow * NSPLIT + split, lse, mask=rvalid)


@triton.jit
def _int8_reduce_kernel(Opart, LSEpart, Out, LseOut, R: tl.constexpr, RP: tl.constexpr, NSPLIT: tl.constexpr,
                        NSP: tl.constexpr):
    qrow = tl.program_id(0)                       # (req, head, j) in prep_q order
    j = qrow % 4
    head = (qrow // 4) % 24
    req = qrow // 96
    sp = tl.arange(0, NSP)
    lse = tl.load(LSEpart + qrow * NSPLIT + sp, mask=sp < NSPLIT, other=float("-inf"))
    m = tl.max(lse, 0)
    m_s = tl.where(m == float("-inf"), 0.0, m)
    w = tl.exp2(lse - m_s)
    wsum = tl.sum(w, 0)
    d = tl.arange(0, RP)
    md = d < R
    acc = tl.zeros([RP], tl.float32)
    for s in range(NSPLIT):
        ws = tl.sum(tl.where(sp == s, w, 0.0), 0)
        if ws > 0:
            acc += ws * tl.load(Opart + (qrow.to(tl.int64) * NSPLIT + s) * R + d, mask=md, other=0.0)
    acc = acc / tl.where(wsum > 0, wsum, 1.0)
    orow = req * 4 + j
    tl.store(Out + (orow * 24 + head) * R + d, acc.to(Out.dtype.element_ty), mask=md)
    tl.store(LseOut + orow * 24 + head, tl.where(wsum > 0, (m_s + tl.log2(wsum)) * 0.6931471805599453, float("-inf")))


def int8_decode(q8, qs, qr8, qrs, cache, block_table, lens, rank, page, nsplit, out, lse_out, hb=8, bn=64,
                pv_int8=False, num_warps=8, num_stages=1, opart=None, lsepart=None, acc16=False, early=False, qin=False):
    """cache uint8 [NP, page_bytes]; block_table int32 [nreq, pages]; lens int32 [nreq] = length of row 3.
    out [nreq*4, 24, rank] (row 4i+j), lse_out [nreq*4, 24] (natural log)."""
    nreq = q8.shape[0]
    assert page % bn == 0 and (rank % 256 == 0 or rank <= 256)
    DC = min(256, rank)
    NC = rank // DC
    nhb = -(-HEADS // hb)
    M = triton.next_power_of_2(hb * ROWS)
    spans = [(b * hb + hb - 1) // 6 - (b * hb) // 6 + 1 for b in range(nhb)]
    RW = 64 * triton.next_power_of_2(max(spans))
    if opart is None:
        opart = torch.empty(nreq * HEADS * ROWS * nsplit * rank, dtype=torch.float32, device=q8.device)
    if lsepart is None:
        lsepart = torch.empty(nreq * HEADS * ROWS * nsplit, dtype=torch.float32, device=q8.device)
    psc = 2.0 ** math.floor(math.log2(65504.0 / (bn * 127)))
    _int8_decode_kernel[(nhb, nsplit, nreq)](
        q8, qs, qr8, qrs, cache.view(torch.int8), block_table, lens, opart, lsepart, block_table.stride(0),
        R=rank, PAGE=page, HB=hb, M=M, BN=bn, DC=DC, NC=NC, RW=RW, NSPLIT=nsplit, PV_INT8=int(pv_int8),
        PSCALE=psc, ACC16=int(acc16), EARLY=int(early), QIN=int(qin), num_warps=num_warps, num_stages=num_stages)
    _int8_reduce_kernel[(nreq * HEADS * ROWS,)](
        opart, lsepart, out, lse_out, R=rank, RP=triton.next_power_of_2(rank), NSPLIT=nsplit,
        NSP=triton.next_power_of_2(nsplit), num_warps=4)
    return out, lse_out
