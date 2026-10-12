"""FP8 cache write in one launch: [latent e4m3 | rope e4m3 | rms (log8 code or bf16 bytes) | pad] per token.

Same bytes as the torch reference in QwenMLATritonImpl.do_kv_cache_update (x * (1/scale), saturated to +-448, round
to nearest even; NaN inputs aside), which costs ~21 small kernels per layer. e4m3 is encoded with integer ops (Triton
has no fp8e4nv below SM89), so this runs on every GPU.
"""
import torch

from vllm.triton_utils import tl, triton


@triton.jit
def _e4m3_encode(x):
    """fp32 -> e4m3 code: saturate to +-448, round to nearest even (torch's c10 conversion, done in integers)."""
    x = tl.minimum(tl.maximum(x, -448.0), 448.0)
    b = x.to(tl.uint32, bitcast=True)
    a = b & 0x7FFFFFFF
    # |x| >= 2^-6 (normal): round the mantissa to 3 bits (carry runs into the exponent), rebias 127 -> 7
    normal = ((a + 0x7FFFF + ((a >> 20) & 1)) >> 20) - (120 << 3)
    # |x| < 2^-6 (subnormal): the code is round(|x| * 2^9); adding 2^23 makes the fp32 add do that rounding
    sub = (a.to(tl.float32, bitcast=True) * 512.0 + 8388608.0).to(tl.uint32, bitcast=True) - 0x4B000000
    return (tl.where(a < 0x3C800000, sub, normal) | ((b >> 24) & 0x80)).to(tl.uint8)


@triton.jit
def _fp8_write_kernel(KV_C, K_PE, Cache, Slots, k_scale, stride_c, stride_pe, stride_blk, stride_tok,
                      BLOCK_SIZE: tl.constexpr, RANK: tl.constexpr, ROPE: tl.constexpr, HEADS: tl.constexpr,
                      B_RANK: tl.constexpr, B_ROPE: tl.constexpr, B_HEADS: tl.constexpr,
                      LOG8: tl.constexpr, LOG2_LO: tl.constexpr, LOG8_STEP: tl.constexpr):
    t = tl.program_id(0)
    slot = tl.load(Slots + t).to(tl.int64)
    if slot >= 0:                                      # -1: CUDA-graph padding
        inv = tl.math.div_rn(1.0, tl.load(k_scale).to(tl.float32))
        row = Cache + (slot // BLOCK_SIZE) * stride_blk + (slot % BLOCK_SIZE) * stride_tok
        o = tl.arange(0, B_RANK)
        lat = tl.load(KV_C + t * stride_c + o, mask=o < RANK).to(tl.float32)
        tl.store(row + o, _e4m3_encode(lat * inv), mask=o < RANK)
        o = tl.arange(0, B_ROPE)
        rope = tl.load(K_PE + t * stride_pe + o, mask=o < ROPE).to(tl.float32)
        tl.store(row + RANK + o, _e4m3_encode(rope * inv), mask=o < ROPE)
        o = tl.arange(0, B_HEADS)
        rms = tl.load(K_PE + t * stride_pe + ROPE + o, mask=o < HEADS, other=1.0)
        if LOG8:
            # tl.log2 is the accurate log2f (not lg2.approx), like torch's; round to nearest even via the
            # fp32 add of 2^23 (clamping first also keeps the multiply out of an FMA)
            code = (tl.log2(tl.maximum(rms.to(tl.float32), 1e-30)) - LOG2_LO) * LOG8_STEP
            code = tl.minimum(tl.maximum(code, 0.0), 255.0) + 8388608.0 - 8388608.0
            tl.store(row + RANK + ROPE + o, code.to(tl.uint8), mask=o < HEADS)
        else:                                          # raw bf16, two bytes per head
            tl.store((row + RANK + ROPE).to(tl.pointer_type(tl.bfloat16)) + o, rms.to(tl.bfloat16), mask=o < HEADS)


def mla_fp8_cache_write(kv_c, k_pe, kv_cache, slots, k_scale, rope, heads, log8, log2_lo=-3.0, log2_hi=3.0):
    """kv_c [T, rank], k_pe [T, >= rope + heads] (rope, then rms per head), kv_cache [blocks, block_size, row]
    (any FP8/uint8 dtype), slots [n <= T] (-1 = skip), k_scale a 1-element fp32 tensor."""
    n = slots.shape[0]
    if n == 0:
        return
    rank = kv_c.shape[-1]
    assert kv_c.stride(-1) == 1 and k_pe.stride(-1) == 1 and kv_cache.stride(-1) == 1
    cache = kv_cache.view(torch.uint8)
    _fp8_write_kernel[(n,)](
        kv_c, k_pe, cache, slots, k_scale, kv_c.stride(0), k_pe.stride(0), cache.stride(0), cache.stride(1),
        BLOCK_SIZE=cache.shape[1], RANK=rank, ROPE=rope, HEADS=heads, B_RANK=triton.next_power_of_2(rank),
        B_ROPE=triton.next_power_of_2(rope), B_HEADS=triton.next_power_of_2(heads), LOG8=log8,
        LOG2_LO=log2_lo, LOG8_STEP=255.0 / (log2_hi - log2_lo), num_warps=4)
