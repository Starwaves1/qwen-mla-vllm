"""CPU only: the sm_86 e4m3 decode (ops.mla_decode_attention._e4m3_f16) is bit-exact against torch's float8_e4m3fn.

  1. The Triton form, run in the interpreter: every finite code gives fp16 bits equal to value / 256 (and _fp8,
     the fp32 form, equals the value), signed zeros included.
  2. The PTX form (the GPU default), emulated instruction by instruction in numpy: prmt with sign replication,
     shl, and -- every code in every byte position of the packed register. Only the element order of Triton's
     pack=4 needs the GPU (tests/gpu_fp8_bench.py checks the real kernel bit for bit).
  python tests/cpu_fp8_decode_exact.py
"""
import os, sys
os.environ["TRITON_INTERPRET"] = "1"; os.environ["CUDA_VISIBLE_DEVICES"] = ""
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import torch
from vllm.triton_utils import tl, triton
import qwen_mla.ops.mla_decode_attention as M


@triton.jit
def k(X, Y16, Y32):
    i = tl.arange(0, 256)
    x = tl.load(X + i)
    tl.store(Y16 + i, M._e4m3_f16(x))
    tl.store(Y32 + i, M._fp8(x))


x = torch.arange(256, dtype=torch.uint8)
y16, y32 = torch.empty(256, dtype=torch.float16), torch.empty(256, dtype=torch.float32)
k[(1,)](x, y16, y32)
ref = x.view(torch.float8_e4m3fn).float()
fin = torch.isfinite(ref)
want = (ref / 256).half()                              # exact: every e4m3 value / 256 is an fp16 value
ok16 = torch.equal(y16[fin].view(torch.int16), want[fin].view(torch.int16))      # bits, so -0 counts
ok32 = torch.equal(y32[fin].view(torch.int32), ref[fin].view(torch.int32))
print(f"triton form: fp16 bits exact on {int((y16[fin].view(torch.int16) == want[fin].view(torch.int16)).sum())}"
      f" of {int(fin.sum())} finite codes ({'ok' if ok16 else 'FAIL'}); fp32 _fp8 {'ok' if ok32 else 'FAIL'};"
      f" NaN codes {(~fin).nonzero().flatten().tolist()} -> {y32[~fin].tolist()}")


def prmt(a, b, sel):                                   # PTX prmt.b32 default mode, a/b uint32 arrays
    src = (b.astype(np.uint64) << 32) | a.astype(np.uint64)
    out = np.zeros_like(a, dtype=np.uint32)
    for i in range(4):
        n = (sel >> (4 * i)) & 0xF
        byte = ((src >> np.uint64(8 * (n & 7))) & np.uint64(0xFF)).astype(np.uint32)
        if n & 8:                                      # replicate the selected byte's sign bit
            byte = np.where(byte & 0x80, 0xFF, 0).astype(np.uint32)
        out |= byte << np.uint32(8 * i)
    return out


def ptx(w):                                            # the asm block in _e4m3_f16, on packed codes
    a, b = prmt(w, np.zeros_like(w), 0x9180), prmt(w, np.zeros_like(w), 0xB3A2)
    a, b = (a << np.uint32(7)) & np.uint32(0xBF80BF80), (b << np.uint32(7)) & np.uint32(0xBF80BF80)
    return np.stack([a & 0xFFFF, a >> 16, b & 0xFFFF, b >> 16], -1).astype(np.uint16)   # elements 0..3


codes = np.arange(256, dtype=np.uint32)
rng = np.random.default_rng(0)
okp = True
for pos in range(4):                                   # each code in each byte, random neighbours
    other = rng.integers(0, 256, size=(256, 4)).astype(np.uint32)
    other[:, pos] = codes
    w = other[:, 0] | (other[:, 1] << 8) | (other[:, 2] << 16) | (other[:, 3] << 24)
    h = ptx(w)
    for e in range(4):
        got = torch.from_numpy(h[:, e].astype(np.int16).view(np.int16).copy())
        exp = (torch.from_numpy(other[:, e].astype(np.uint8)).view(torch.float8_e4m3fn).float() / 256).half()
        f = torch.isfinite(exp)
        okp &= torch.equal(got[f], exp[f].view(torch.int16))
print(f"ptx form (numpy emulation, 4 byte positions x 256 codes): {'ok' if okp else 'FAIL'}")

ok = ok16 and ok32 and okp
print("PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)
