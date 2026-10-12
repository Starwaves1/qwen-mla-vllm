"""CPU only: compile (not run) every Triton kernel the decode and the FP8 cache write launch for an sm_86 target with an FP8 cache, from
the real launch code (JITFunction.run forced to warmup=True, the driver reporting sm_86). With the uint8 carriage
all of them compile; NATIVE=1 hands the kernels float8 tensors as before the fix and fails (no fp8e4nv on sm_86).
  python tests/cpu_sm86_compile.py      (QWEN_MLA_FP8_DECODE=asm|hoist|triton: all compile)
"""
import os, sys
os.environ["CUDA_VISIBLE_DEVICES"] = ""
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch, triton
from triton.backends.compiler import GPUTarget
from triton.runtime import driver
from triton.runtime.jit import JITFunction
from triton.backends.nvidia.driver import CudaDriver
drv = CudaDriver()
driver.set_active(drv)
drv.get_current_target = lambda *a, **k: GPUTarget("cuda", 86, 32)
drv.get_current_device = lambda *a, **k: 0
drv.get_current_stream = lambda *a, **k: 0
seen = []
orig = JITFunction.run
def run(self, *a, **k):
    k["warmup"] = True
    r = orig(self, *a, **k)
    seen.append(self.fn.__name__)
    return r
JITFunction.run = run
import qwen_mla.ops.mla_decode_attention as M
from vllm.platforms import current_platform
type(current_platform).has_device_capability = classmethod(lambda cls, c, device_id=0: c <= 86 or os.environ.get("NATIVE") == "1")
import test_fp8_kv as T
T.dev = torch.device("cpu")
for small in (True, False):
    M._small_smem = lambda d, s=small: s
    for mode in ("log8", "bf16", "e4m3"):
        for rank in (256, 768, 1792):
            batch, heads, ctx, rope = 2, 24, 1600, 232
            lat, rp, rms, q, pps = T.make(batch, heads, ctx, rank, rope, seed=0)
            scale = torch.tensor(0.05)
            width = rank + rope + (2 if mode == "bf16" else 1) * heads; width += (-width) % 16
            cache = T.fp8_cache(lat, rp, rms, scale, width, mode)
            bt = torch.arange(batch * pps, dtype=torch.int32).reshape(batch, pps)
            lens = torch.tensor([ctx] * batch, dtype=torch.int32)
            o = torch.zeros(batch, heads, rank, dtype=torch.bfloat16); lse = torch.zeros(batch, heads, dtype=torch.bfloat16)
            lg = torch.zeros(batch, heads, 8, rank + 1, dtype=torch.float32)
            try:
                M.decode_attention_fwd(q, cache, cache[..., :rank], o, lse, bt, lens, lg, 8, 1 / 16.0, T.PAGE,
                                       k_scale=scale, v_scale=scale, is_mla=True, rms_offset=rank + rope,
                                       rms_bf16=mode == "bf16", rms_log8=mode == "log8")
            except Exception as e:
                print(f"two-pass={small} {mode} {rank}: FAILED {type(e).__name__}: {str(e)[:300]}"); sys.exit(1)
from qwen_mla.ops.mla_fp8_write import mla_fp8_cache_write
for log8 in (True, False):                              # the one-launch FP8 cache write (no fp8 types at all)
    cache = torch.zeros(4, 16, 1072, dtype=torch.float8_e4m3fn)
    mla_fp8_cache_write(torch.zeros(8, 768, dtype=torch.bfloat16), torch.zeros(8, 256, dtype=torch.bfloat16), cache,
                        torch.arange(8), torch.tensor(0.05), 232, 24, log8=log8)
print("compiled for sm_86 with a uint8 cache:", sorted(set(seen)), f"({len(seen)} launches)",
      "decode mode:", M._FP8_MODE)
