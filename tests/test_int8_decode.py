# accuracy: int8-tok kernel vs exact references; FP8 kernel on the same fp32 latents
import os, sys, math, torch
sys.path.insert(0, '/home/garrett/tools/mla-vllm/src/qwen-mla-vllm'); sys.path.insert(0, '/home/garrett/tools/mla-vllm/src/qwen-mla-vllm/tests')
import qwen_mla.ops.mla_int8_decode as I
dev = torch.device('cuda'); ROPE, H = 232, 24
SM = 1 / 16.0


def make(nreq, ctx, rank, page, seed, kind='real'):
    g = torch.Generator(device=dev).manual_seed(seed)
    pps = -(-ctx // page); NP = nreq * pps + 3
    lat = torch.randn(NP, page, rank, device=dev, generator=g) * 2.0
    lat[..., :4] *= 8
    rp = torch.randn(NP, page, ROPE, device=dev, generator=g) * 1.5
    rms = torch.exp(torch.randn(NP, page, H, device=dev, generator=g) * 0.6)
    q = torch.randn(nreq * 4, H, rank + ROPE, device=dev, generator=g) * 0.05
    if kind == 'diffuse':           # near-uniform scores: tiny q, a few keys slightly favoured
        q *= 0.02
    if kind == 'peaky':
        q *= 4.0
    mask = torch.zeros(H, ROPE, device=dev)
    for h in range(H):
        mask[h, 58 * (h // 6):58 * (h // 6) + 58] = 1
    q[..., rank:] *= mask
    q = q.to(torch.bfloat16)
    perm = torch.randperm(NP, generator=torch.Generator().manual_seed(seed))[:nreq * pps].view(nreq, pps)
    return lat, rp, rms, q, perm.to(torch.int32).to(dev)


def ref(q, lat, rp, rms, bt, lens_rows, rank, page):
    """q [B,H,D] float; lat [NP,P,R]; rms [NP,P,H] (divisor); lens_rows [B]; bt per row [B, pps]."""
    out = []
    for b in range(q.shape[0]):
        L = int(lens_rows[b])
        if L <= 0:
            out.append(torch.zeros(H, rank, device=dev, dtype=torch.float64)); continue
        c = lat[bt[b].long()].reshape(-1, rank)[:L].double()
        r = rp[bt[b].long()].reshape(-1, rp.shape[-1])[:L].double()
        m = rms[bt[b].long()].reshape(-1, H)[:L].double()
        qf = q[b].double()
        s = (qf[:, :rank] @ c.T + qf[:, rank:] @ r.T) / m.T * SM
        out.append(torch.softmax(s, -1) @ c)
    return torch.stack(out)


def rel(a, b):
    return ((a.double() - b.double()).abs().max() / b.double().abs().max()).item()


def run_int8(q, cache, bt, lens, rank, page, nsplit, pv_int8, hb=8, bn=64, **kw):
    q8, qs, qr8, qrs = I.prep_q(q, rank, SM)
    nreq = bt.shape[0]
    out = torch.empty(nreq * 4, H, rank, dtype=torch.float32, device=dev)
    lse = torch.empty(nreq * 4, H, dtype=torch.float32, device=dev)
    kw.setdefault('num_stages', int(os.environ.get('ST', '1'))); I.int8_decode(q8, qs, qr8, qrs, cache, bt, lens, rank, page, nsplit, out, lse, hb=hb, bn=bn, pv_int8=pv_int8, **kw)
    return out, lse, (q8, qs, qr8, qrs)


def deq_q(q8, qs, qr8, qrs, rank):
    nreq = q8.shape[0]
    k = SM * I.LOG2E
    ql = q8.float() * (qs / k)[..., None]                      # [nreq,24,4,rank]
    qr = (qr8.float() * (qrs / k)[..., None]).view(nreq, H, 4, 4, 64)[..., :58].reshape(nreq, H, 4, 232)
    return torch.cat([ql, qr], -1).transpose(1, 2).reshape(nreq * 4, H, rank + 232)


def fp8_run(q, lat, rp, rms, btr, lens_rows, rank, page, splits=32):
    import qwen_mla.ops.mla_decode_attention as M
    import test_fp8_kv as T
    T.dev = dev
    os.environ['QWEN_MLA_2P_MIN_LV'] = '512'
    for k_, v_ in dict(QWEN_MLA_2P_BN='64', QWEN_MLA_2P_DC='128', QWEN_MLA_2P_DVC='256', QWEN_MLA_2P_STAGES_A='1', QWEN_MLA_2P_STAGES_B='1').items():
        os.environ[k_] = v_
    scale = torch.tensor(float(max(lat.abs().max(), rp.abs().max())) / 448, device=dev)
    w = rank + ROPE + H; w += (-w) % 16
    cache = T.fp8_cache(lat, rp, rms, scale, w, "log8")
    B = q.shape[0]
    o = torch.empty(B, H, rank, dtype=torch.float32, device=dev); lse = torch.empty(B, H, dtype=torch.float32, device=dev)
    lg = torch.empty(B, H, splits, rank + 1, dtype=torch.float32, device=dev)
    M.decode_attention_fwd(q, cache, cache[..., :rank], o, lse, btr, lens_rows, lg, splits, SM, page, k_scale=scale, v_scale=scale,
                           is_mla=True, rms_offset=rank + ROPE, rms_bf16=False, rms_log8=True)
    dl, dr = T.deq(lat, rp, scale)
    return o, dl, dr, T.rms_decoded(rms, scale, "log8")


def hadamard_blocks(rank, n=256):
    """Orthonormal block-diagonal Hadamard (blocks of n): models a rotation folded into kv_a_proj / W_UK / W_UV."""
    h = torch.ones(1, 1, device=dev)
    while h.shape[0] < n:
        h = torch.cat([torch.cat([h, h], 1), torch.cat([h, -h], 1)], 0)
    return torch.block_diag(*[h / math.sqrt(n)] * (rank // n))


def case(nreq, ctx, rank, page, kind='real', nsplit=8, Ls=None, do_fp8=True, seed=0, hb=8, bn=64, rot=None, **kw):
    lat, rp, rms, q, bt = make(nreq, ctx, rank, page, seed + rank, kind)
    rot = int(os.environ.get('ROT', '0')) if rot is None else rot
    if rot:
        Hb = hadamard_blocks(rank)
        lat_k, q_k = lat @ Hb, q.float().clone()
        q_k[..., :rank] = q_k[..., :rank] @ Hb
        q_k = q_k.to(torch.bfloat16)
    else:
        lat_k, q_k = lat, q
    cache = I.pack_cache(lat_k, rp, rms)
    Ls = Ls or [ctx] * nreq
    lens = torch.tensor(Ls, dtype=torch.int32, device=dev)
    lens_rows = torch.tensor([L - 3 + j for L in Ls for j in range(4)], dtype=torch.int32, device=dev)
    btr = bt.repeat_interleave(4, 0).contiguous()
    dl, dr, drms = I.unpack_cache(cache, rank, page)
    o16, lse16, qq = run_int8(q_k, cache, bt, lens, rank, page, nsplit, False, hb=hb, bn=bn, **kw)
    o8, lse8, _ = run_int8(q_k, cache, bt, lens, rank, page, nsplit, True, hb=hb, bn=bn, **kw)
    qd = deq_q(*qq, rank)
    r_arith = ref(qd, dl, dr, drms, btr, lens_rows, rank, page)            # same quantized inputs as the kernel
    r_deqc = ref(q_k.float(), dl, dr, drms, btr, lens_rows, rank, page)    # quantized cache, bf16 q
    if rot:   # outputs are in the rotated basis; W_UV would absorb Hb^T
        o16, o8 = o16 @ Hb.T, o8 @ Hb.T
        r_deqc = (r_deqc.float() @ Hb.T).double()
        r_arith = (r_arith.float() @ Hb.T).double()
    rms_t = torch.exp2(-3.0 + ((rms.log2() + 3.0) * (255.0 / 6.0)).round().clamp(0, 255) * (6.0 / 255.0))
    r_true = ref(q.float(), lat, rp, rms_t, btr, lens_rows, rank, page)    # fp32 latents (log8 rms in both paths)
    res = dict(i16_arith=rel(o16, r_arith), i8_arith=rel(o8, r_arith), i16_true=rel(o16, r_true), i8_true=rel(o8, r_true),
               qerr_int8=rel(r_deqc, r_true), qerr_int8q=rel(r_arith, r_true),
               nan=bool(torch.isnan(o16).any() or torch.isnan(o8).any() or torch.isinf(o16).any() or torch.isinf(o8).any()))
    if do_fp8:
        of, fdl, fdr, frms = fp8_run(q, lat, rp, rms, btr, lens_rows, rank, page)
        r_f = ref(q.float(), fdl, fdr, frms, btr, lens_rows, rank, page)
        res.update(fp8_arith=rel(of, r_f), fp8_true=rel(of, r_true), qerr_fp8=rel(r_f, r_true))
    return res


if __name__ == '__main__':
    # basic: the three ranks x three score shapes, plain and with a block-Hadamard rotation of the latent
    # (rot=1 models a rotation folded into kv_a_proj / W_UK / W_UV at load time; the kernel is unchanged)
    cases = os.environ.get('CASES', 'basic')
    if cases == 'basic':
        for rot in (0, 1):
            for rank, page in ((768, 1664), (256, 1664), (1792, 832)):
                kw = {1792: dict(hb=8, bn=32, acc16=True, qin=True), 768: dict(hb=8, bn=64, acc16=True),
                      256: dict(hb=8, bn=128, acc16=True)}[rank]
                for kind in ('real', 'diffuse', 'peaky'):
                    r = case(2, 5000, rank, page, kind, nsplit=4, Ls=[5000, 1700], rot=rot, **kw)
                    print(f'rot={rot}', rank, kind, ' '.join(f'{k}={v:.2e}' if isinstance(v, float) else f'{k}={v}' for k, v in r.items()), flush=True)
