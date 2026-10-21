"""ctypes loader for qwen_mla/csrc/mla_int8_decode.cu (experimental int8-tok decode, sm_86). Not used by prod.

The .so is built with the system nvcc (any CUDA 12.x; static cudart, so it does not need torch's CUDA version)
into $QWEN_MLA_INT8_BUILD (default /tmp/qwen_mla_int8) the first time it is needed or when the source changes.
Same inputs and outputs as mla_int8_decode.int8_decode (prep_q outputs, int8-tok cache, Triton split reduce).
"""
import ctypes
import hashlib
import os
import subprocess

import torch
import triton
import triton.language as tl

from . import mla_int8_decode as I

_SRC = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "csrc", "mla_int8_decode.cu")
_lib = None


def _load():
    global _lib
    if _lib is not None:
        return _lib
    src = open(_SRC, "rb").read()
    tag = hashlib.sha1(src).hexdigest()[:12]
    bdir = os.environ.get("QWEN_MLA_INT8_BUILD", "/tmp/qwen_mla_int8")
    so = os.environ.get("QWEN_MLA_INT8_SO") or os.path.join(bdir, f"libmla_int8_{tag}.so")
    if not os.path.exists(so):
        os.makedirs(bdir, exist_ok=True)
        tmp = so + f".{os.getpid()}.tmp"
        r = subprocess.run([os.environ.get("NVCC", "nvcc"), "-O3", "-arch=sm_86", "-std=c++17", "--use_fast_math", "-diag-suppress", "128",
                            "-shared", "-Xcompiler", "-fPIC", "-o", tmp, _SRC], capture_output=True, text=True, timeout=600)
        if r.returncode != 0:
            raise RuntimeError(f"nvcc failed for {_SRC}:\n{r.stderr[-4000:]}")
        os.replace(tmp, so)
    lib = ctypes.CDLL(so)
    lib.mla_int8_decode.restype = ctypes.c_int
    lib.mla_int8_decode.argtypes = [ctypes.c_void_p] * 10 + [ctypes.c_int, ctypes.c_longlong] + [ctypes.c_int] * 8 + [ctypes.c_void_p]
    _lib = lib
    return lib


@triton.jit
def _reduce_kernel(Opart, LSEpart, Out, LseOut, R: tl.constexpr, NSPLIT: tl.constexpr, NSP: tl.constexpr, DB: tl.constexpr):
    """Combine the fp16 split outputs: one program per (query row, DB columns); all splits loaded as one block."""
    qrow = tl.program_id(0)                       # (req, head, j) in prep_q order
    db = tl.program_id(1)
    j = qrow % 4
    head = (qrow // 4) % 24
    req = qrow // 96
    sp = tl.arange(0, NSP)
    lse = tl.load(LSEpart + qrow * NSPLIT + sp, mask=sp < NSPLIT, other=float("-inf"))
    m = tl.max(lse, 0)
    m_s = tl.where(m == float("-inf"), 0.0, m)
    w = tl.exp2(lse - m_s)
    wsum = tl.sum(w, 0)
    d = db * DB + tl.arange(0, DB)
    x = tl.load(Opart + (qrow.to(tl.int64) * NSPLIT + sp[:, None]) * R + d[None, :], mask=(sp[:, None] < NSPLIT) & (w[:, None] > 0),
                other=0.0).to(tl.float32)
    acc = tl.sum(x * w[:, None], 0) / tl.where(wsum > 0, wsum, 1.0)
    orow = req * 4 + j
    tl.store(Out + (orow * 24 + head) * R + d, acc.to(Out.dtype.element_ty))
    if db == 0:
        tl.store(LseOut + orow * 24 + head, tl.where(wsum > 0, (m_s + tl.log2(wsum)) * 0.6931471805599453, float("-inf")))


SMS = 82   # RTX 3090; one CTA per SM for every config except (256, bn 32, nm 2) which runs 2


def pick_config(rank, nreq, ctx):
    """(bn, nm, nsplit) from the night-of-2026-10-21 grid on the RTX 3090 (tests/bench_int8_decode_cuda.py).
    nm = row tiles of 16 per CTA (4 heads each): 3 -> 12-head blocks (2 CTAs per key tile), 2 -> 8-head, 1 -> 4-head."""
    if rank == 1792:
        bn, nm, waves = 16, 1, (3 if nreq * ctx >= 4 * 32768 else 1)
    else:
        bn, nm, waves = 32, (2 if nreq == 1 or nreq * ctx <= 32768 else 3), 1
    nhb = 24 // (4 * nm)
    return bn, nm, max(1, min(SMS * waves // (nhb * nreq), ctx // 256))


def int8_decode_cuda(q8, qs, qr8, qrs, cache, block_table, lens, rank, page, nsplit, out, lse_out, bn=None,
                     opart=None, lsepart=None, mode=1, sdbg=None, nm=None):
    """cache uint8 [NP, page_bytes]; block_table int32 [nreq, pages]; lens int32 [nreq] = length of MTP row 3.
    out [nreq*4, 24, rank], lse_out [nreq*4, 24] (natural log). nsplit / bn / nm None -> pick_config(rank, nreq,
    nsplit or the block table's capacity). mode 0 = score-only probe (sdbg [nreq*96, ld] fp32), 2 = loads only."""
    lib = _load()
    nreq = q8.shape[0]
    if nsplit is None or bn is None or nm is None:
        pbn, pnm, psp = pick_config(rank, nreq, block_table.shape[1] * page)
        bn, nm, nsplit = bn or pbn, nm or pnm, nsplit or psp
    assert page % bn == 0 and block_table.dtype == torch.int32 and lens.dtype == torch.int32
    if opart is None:
        opart = torch.empty(nreq * I.HEADS * I.ROWS * nsplit * rank, dtype=torch.float16, device=q8.device)
    if lsepart is None:
        lsepart = torch.empty(nreq * I.HEADS * I.ROWS * nsplit, dtype=torch.float32, device=q8.device)
    stream = torch.cuda.current_stream(q8.device).cuda_stream
    err = lib.mla_int8_decode(q8.data_ptr(), qs.data_ptr(), qr8.data_ptr(), qrs.data_ptr(), cache.data_ptr(),
                              block_table.data_ptr(), lens.data_ptr(), opart.data_ptr(), lsepart.data_ptr(),
                              sdbg.data_ptr() if sdbg is not None else None, sdbg.shape[1] if sdbg is not None else 0,
                              I.page_bytes(rank, page), block_table.stride(0), nsplit, page, nreq, rank, bn, nm, mode, stream)
    if err != 0:
        raise RuntimeError(f"mla_int8_decode failed: {err}")
    if mode == 0:
        return opart, lsepart
    assert opart.dtype == torch.float16
    db = 128 if rank % 128 == 0 else 64
    _reduce_kernel[(nreq * I.HEADS * I.ROWS, rank // db)](
        opart, lsepart, out, lse_out, R=rank, NSPLIT=nsplit, NSP=triton.next_power_of_2(nsplit), DB=db, num_warps=2)
    return out, lse_out
