"""CPU only (Triton interpreter): ops.mla_decode_attention._fp8 decodes every finite e4m3 code exactly.
The interpreter cannot run the decode kernels themselves (the upstream bf16 path also returns garbage there).
  python tests/cpu_fp8_decode_exact.py
"""
import os, sys
os.environ["TRITON_INTERPRET"] = "1"; os.environ["CUDA_VISIBLE_DEVICES"] = ""
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from vllm.triton_utils import tl, triton
import qwen_mla.ops.mla_decode_attention as M
from vllm.platforms import current_platform
print("platform", type(current_platform).__name__, "has sm89:", current_platform.has_device_capability(89))

@triton.jit
def k(X, Y):
    i = tl.arange(0, 256)
    tl.store(Y + i, M._fp8(tl.load(X + i)))
x = torch.arange(256, dtype=torch.uint8); y = torch.empty(256, dtype=torch.float32)
k[(1,)](x, y)
ref = x.view(torch.float8_e4m3fn).float()
fin = torch.isfinite(ref)
print("e4m3 decode: exact on", int((y[fin] == ref[fin]).sum()), "of", int(fin.sum()), "finite codes; NaN codes:", (~fin).nonzero().flatten().tolist())

ok = bool((y[fin] == ref[fin]).all())
print("PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)
