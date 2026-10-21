"""int8-tok cache write (one launch per step): the page layout of mla_int8_decode.py, written token by token through a
slot mapping, as vLLM's do_kv_cache_update would. Byte-identical to mla_int8_decode.pack_cache. Not used by prod.

    latent int8 [PAGE, R] (scale sigma_t = absmax / 127 in fp16) | rope int8 [PAGE, 256] (4 groups x 58, padded to 64,
    scale rho_t) | sigma fp16 [PAGE] | rho fp16 [PAGE] | log8 rms code uint8 [24, PAGE]
"""
import torch
import triton
import triton.language as tl

from .mla_int8_decode import GROUPS, HEADS, RG, RGP, page_bytes


@triton.jit
def _int8_write_kernel(KV_C, K_PE, Cache, Slots, stride_c, stride_pe, stride_blk,
                       R: tl.constexpr, RP: tl.constexpr, PAGE: tl.constexpr):
    t = tl.program_id(0)
    slot = tl.load(Slots + t).to(tl.int64)
    if slot < 0:                                   # padded token
        return
    blk = slot // PAGE
    tok = slot % PAGE
    base = Cache + blk * stride_blk
    # latent: per-token absmax scale, rounded to fp16 first (the kernel multiplies by the fp16 value)
    d = tl.arange(0, RP)
    x = tl.load(KV_C + t * stride_c + d, mask=d < R, other=0.0).to(tl.float32)
    sig = (tl.max(tl.abs(x), 0) / 127.0).to(tl.float16)
    sig = tl.where(sig > 0, sig, 1.0)
    q = tl.extra.cuda.libdevice.rint(tl.math.div_rn(x, sig.to(tl.float32)))
    q = tl.minimum(tl.maximum(q, -127.0), 127.0).to(tl.int8)
    tl.store(base + tok * R + d, q.to(tl.uint8, bitcast=True), mask=d < R)
    # rope: 4 groups x 58 -> 4 x 64 (zero padded), own scale
    e = tl.arange(0, 256)
    g = e // 64
    w = e % 64
    r = tl.load(K_PE + t * stride_pe + g * 58 + w, mask=w < 58, other=0.0).to(tl.float32)
    rho = (tl.max(tl.abs(r), 0) / 127.0).to(tl.float16)
    rho = tl.where(rho > 0, rho, 1.0)
    rq = tl.extra.cuda.libdevice.rint(tl.math.div_rn(r, rho.to(tl.float32)))
    rq = tl.minimum(tl.maximum(rq, -127.0), 127.0).to(tl.int8)
    tl.store(base + PAGE * R + tok * 256 + e, rq.to(tl.uint8, bitcast=True))
    so = PAGE * (R + 256)
    tl.store((base + so).to(tl.pointer_type(tl.float16)) + tok, sig)
    tl.store((base + so + 2 * PAGE).to(tl.pointer_type(tl.float16)) + tok, rho)
    # rms divisor per head as the log8 code: round((log2(rms) + 3) * 255 / 6), clamped to [0, 255]
    h = tl.arange(0, 32)
    rms = tl.load(K_PE + t * stride_pe + 232 + h, mask=h < 24, other=1.0).to(tl.float32)
    code = tl.extra.cuda.libdevice.rint((tl.log2(rms) + 3.0) * (255.0 / 6.0))
    code = tl.minimum(tl.maximum(code, 0.0), 255.0).to(tl.uint8)
    tl.store(base + so + 4 * PAGE + h * PAGE + tok, code, mask=h < 24)


def mla_int8_cache_write(kv_c, k_pe, cache, slots, page):
    """kv_c [T, R] (kv_c_normed), k_pe [T, >= 256] (rope 4 x 58 = 232, then the rms divisor of the 24 heads),
    cache uint8 [blocks, page_bytes(R, page)], slots int [T] (-1 = skip)."""
    T, R = kv_c.shape
    assert GROUPS * RG == 232 and HEADS == 24 and RGP == 64
    assert cache.dtype == torch.uint8 and cache.shape[1] == page_bytes(R, page) and kv_c.stride(1) == 1 and k_pe.stride(1) == 1
    if T == 0:
        return
    _int8_write_kernel[(T,)](kv_c, k_pe, cache, slots, kv_c.stride(0), k_pe.stride(0), cache.stride(0),
                             R=R, RP=triton.next_power_of_2(R), PAGE=page, num_warps=4)
