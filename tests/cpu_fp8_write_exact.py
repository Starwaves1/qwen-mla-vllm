"""CPU only (Triton interpreter): the one-launch FP8 cache write (ops/mla_fp8_write.py) writes the same bytes as the
torch ops it replaces (QWEN_MLA_FP8_WRITE=torch in QwenMLATritonImpl.do_kv_cache_update).

  1. _e4m3_encode vs torch's x.clamp(-448, 448).to(float8_e4m3fn): every e4m3 value, every rounding midpoint
     (ties go to even) and the fp32 neighbours of each, the subnormal/normal boundary, signed zeros, values past
     448, and 2M random fp32 values over [2^-14, 2^10].
  2. Whole rows: latent + rope e4m3 and the rms tail (log8 codes and raw bf16), with CUDA-graph padding slots (-1),
     scattered slots, and a cache whose blocks are padded (stride(0) > block_size * row).
  python tests/cpu_fp8_write_exact.py
"""
import os, sys
os.environ["TRITON_INTERPRET"] = "1"; os.environ["CUDA_VISIBLE_DEVICES"] = ""
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from vllm.triton_utils import tl, triton
from qwen_mla.ops.mla_fp8_write import _e4m3_encode
import qwen_mla.mla_impl as MI

F8 = torch.float8_e4m3fn


@triton.jit
def enc(X, Y, N):
    i = tl.program_id(0) * 4096 + tl.arange(0, 4096)
    tl.store(Y + i, _e4m3_encode(tl.load(X + i, mask=i < N)), mask=i < N)


vals = torch.arange(256, dtype=torch.uint8).view(F8).float()
vals = vals[torch.isfinite(vals)].unique()                               # sorted, 253 (one zero)
mid = (vals[1:] + vals[:-1]) / 2                                         # exact in fp32: the ties
near = lambda t: torch.cat([t, torch.nextafter(t, t + 1), torch.nextafter(t, t - 1)])
g = torch.Generator().manual_seed(0)
rnd = torch.exp2(torch.rand(2_000_000, generator=g) * 24 - 14) * (torch.randint(0, 2, (2_000_000,), generator=g) * 2 - 1)
x = torch.cat([near(vals), near(mid), torch.tensor([0.0, -0.0, 2.0 ** -6, -(2.0 ** -6), 2.0 ** -10, 448.0, 449.0,
                                                   464.0, 1e6, -1e6, float("inf"), float("-inf")]), rnd]).float()
y = torch.empty(x.shape[0], dtype=torch.uint8)
enc[(triton.cdiv(x.shape[0], 4096),)](x, y, x.shape[0])
want = x.clamp(-448, 448).to(F8).view(torch.uint8)
bad = (y != want).nonzero().flatten()
ok1 = bad.numel() == 0
print(f"_e4m3_encode: {x.shape[0] - bad.numel()} of {x.shape[0]} values give torch's byte "
      f"({'ok' if ok1 else 'FAIL: ' + str([(x[i].item(), y[i].item(), want[i].item()) for i in bad[:5]])})")


def rows(mode, T=37, rank=768, rope=232, heads=24, blocks=6, bs=16, row=1024, pad=64):
    os.environ["QWEN_MLA_FP8_RMS"] = mode
    torch.manual_seed(1)
    kv_c = torch.randn(T + 3, rank).bfloat16() * 3                      # 3 CUDA-graph padding rows past the slots
    k_pe = torch.cat([torch.randn(T + 3, 1, rope) * 2,
                      torch.exp(torch.randn(T + 3, 1, heads) * 0.8), torch.zeros(T + 3, 1, 256 - rope - heads)], -1).bfloat16()
    k_pe[0, 0, :4] = torch.tensor([0.0, -0.0, 1e-9, -5000.0])           # zeros, deep subnormal, saturation
    slots = torch.randperm(blocks * bs)[:T]
    slots[[3, 10]] = -1                                                  # padding tokens
    scale = torch.tensor(0.0455)
    impl = MI.QwenMLATritonImpl.__new__(MI.QwenMLATritonImpl)
    impl.rms_offset, impl.mla_packed_rope, impl.mla_local_heads = rank + rope, rope, heads
    out = {}
    for path in ("torch", "triton"):
        os.environ["QWEN_MLA_FP8_WRITE"] = path
        # the triton path gets padded pages (stride(0) > bs * row); the torch path needs a contiguous cache
        base = torch.full((blocks, bs * row + (pad if path == "triton" else 0)), 0xAB, dtype=torch.uint8)
        cache = base[:, :bs * row].view(blocks, bs, row).view(F8)
        impl.do_kv_cache_update(kv_c, k_pe, cache, slots, "fp8", scale)
        out[path] = base
    sl = slots[slots >= 0]           # compare written slots only: torch sends padding to slot 0, triton skips it
    a = out["torch"][:, :bs * row].reshape(blocks * bs, row)[sl]
    b = out["triton"][:, :bs * row].reshape(blocks * bs, row)[sl]
    ok = torch.equal(a, b) and bool((out["triton"][:, bs * row:] == 0xAB).all())
    print(f"rows ({mode} rms): {sl.numel()} slots x {row} bytes identical to the torch path, padding untouched: "
          f"{'ok' if ok else 'FAIL'}")
    return ok


ok = ok1 & rows("log8") & rows("bf16", row=1072)
print("PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)
