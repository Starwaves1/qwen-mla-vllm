"""CPU only (Triton interpreter): the decode kernels with an sm_86 FP8 cache (uint8 codes, _e4m3_f16) against the
exact attention over the dequantized cache, for the single-pass and two-pass kernels and every rms tail format.

The output is fp32 here because the interpreter truncates fp32 -> bf16 stores (the GPU rounds to nearest), so the
comparison sees only the kernel's own arithmetic: fp16 q at 1/256, exact fp16 e4m3/256 keys and values, fp16 p.
A wrong scale, offset or layout shows up as an error near 1; the arithmetic itself stays below 1e-3.
(The interpreter cannot check the bf16-cache path: its bf16 emulation returns garbage there.)
  python tests/cpu_fp8_decode_interp.py
"""
import os, sys
os.environ["TRITON_INTERPRET"] = "1"; os.environ["CUDA_VISIBLE_DEVICES"] = ""
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE)); sys.path.insert(0, HERE)
import torch
import test_fp8_kv as T
import qwen_mla.ops.mla_decode_attention as M
from vllm.platforms import current_platform

T.dev = torch.device("cpu"); T.PAGE = 64
type(current_platform).has_device_capability = classmethod(lambda cls, c, device_id=0: c <= 86)
rel = lambda a, b: ((a.float() - b).abs().max() / b.abs().max()).item()


def run(rank, heads, rope, mode, two_pass, batch=2, ctx=200, splits=4):
    M._small_smem = lambda d: two_pass
    lat, rp, rms, q, pps = T.make(batch, heads, ctx, rank, rope, seed=rank + heads)
    bt = torch.randperm(batch * pps, generator=torch.Generator().manual_seed(1)).reshape(batch, pps).to(torch.int32)
    lens = torch.tensor([ctx - 37 * b for b in range(batch)], dtype=torch.int32)
    scale = torch.tensor(float(max(lat.abs().max(), rp.abs().max())) / 448)
    width = rank + rope + (2 if mode == "bf16" else 1) * heads
    width += (-width) % 16
    cache = T.fp8_cache(lat, rp, rms, scale, width, mode)
    o = torch.zeros(batch, heads, rank, dtype=torch.float32); lse = torch.zeros(batch, heads)
    lg = torch.zeros(batch, heads, splits, rank + 1)
    M.decode_attention_fwd(q, cache, cache[..., :rank], o, lse, bt, lens, lg, splits, 1 / 16.0, T.PAGE,
                           k_scale=scale, v_scale=scale, is_mla=True, rms_offset=rank + rope,
                           rms_bf16=mode == "bf16", rms_log8=mode == "log8")
    dl, dr = T.deq(lat, rp, scale)
    r = T.ref(q, dl, dr, T.rms_decoded(rms, scale, mode), bt, lens, rank, rms_exact=True)
    e = rel(o, r)
    ok = e < 1e-3
    print(f"{'two-pass' if two_pass else 'single  '} {mode:4s} rank {rank:4d} heads {heads}: kernel vs dequantized "
          f"ref {e:.2e} | vs bf16-KV ref {rel(o, T.ref(q, lat, rp, rms, bt, lens, rank)):.2e} (fp8 quantization)"
          f"  {'ok' if ok else 'FAIL'}", flush=True)
    return ok


ok = True
for rank, heads, rope, mode, two_pass in [(256, 24, 232, "log8", False), (768, 24, 232, "log8", False),
                                          (1792, 24, 232, "log8", True), (256, 24, 232, "bf16", False),
                                          (256, 24, 232, "e4m3", False), (768, 12, 116, "log8", False),
                                          (1792, 24, 232, "e4m3", True),
                                          (1792, 24, 232, "bf16", True)]:
    ok &= run(rank, heads, rope, mode, two_pass)
print("PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)
