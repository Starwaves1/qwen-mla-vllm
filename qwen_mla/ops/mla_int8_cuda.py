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
        subprocess.run([os.environ.get("NVCC", "nvcc"), "-O3", "-arch=sm_86", "-std=c++17", "--use_fast_math", "-shared", "-Xcompiler", "-fPIC",
                        "-o", tmp, _SRC], check=True, timeout=600)
        os.replace(tmp, so)
    lib = ctypes.CDLL(so)
    lib.mla_int8_decode.restype = ctypes.c_int
    lib.mla_int8_decode.argtypes = [ctypes.c_void_p] * 10 + [ctypes.c_int, ctypes.c_longlong] + [ctypes.c_int] * 8 + [ctypes.c_void_p]
    _lib = lib
    return lib


def int8_decode_cuda(q8, qs, qr8, qrs, cache, block_table, lens, rank, page, nsplit, out, lse_out, bn=32,
                     opart=None, lsepart=None, mode=1, sdbg=None, nm=3):
    """cache uint8 [NP, page_bytes]; block_table int32 [nreq, pages]; lens int32 [nreq] = length of MTP row 3.
    out [nreq*4, 24, rank], lse_out [nreq*4, 24] (natural log). mode 0 = score-only probe (sdbg [nreq*96, ld] fp32)."""
    lib = _load()
    nreq = q8.shape[0]
    assert page % bn == 0 and block_table.dtype == torch.int32 and lens.dtype == torch.int32
    if opart is None:
        opart = torch.empty(nreq * I.HEADS * I.ROWS * nsplit * rank, dtype=torch.float32, device=q8.device)
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
    I._int8_reduce_kernel[(nreq * I.HEADS * I.ROWS,)](
        opart, lsepart, out, lse_out, R=rank, RP=triton.next_power_of_2(rank), NSPLIT=nsplit,
        NSP=triton.next_power_of_2(nsplit), num_warps=4)
    return out, lse_out
