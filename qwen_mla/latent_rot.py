"""Block-Hadamard rotation of the cached latent, folded into the weights at load time (QWEN_MLA_LATENT_ROT).

The latent c = W_a x is cached and read through linear maps only: the up-projections (kv_b_proj, so W_UK / W_UV, and
the raw k_up kept for the rms divisor) and, at decode, the absorbed query q . c. With an orthonormal R,
    c' = R c = (R W_a) x,   K c = (K R^T) c',   so   W_a <- R W_a,  K <- K R^T  for every up-projection K
leaves every attention output unchanged (the latent has no layernorm in this model, so nothing sits between W_a
and the cache). What changes is the cached vector: R = blockdiag(H_256 / 16) spreads each latent's outliers over
256 dims, which is what a per-token int8 scale (int8-tok cache) needs. Every rank in use is a multiple of 256.
"""
from __future__ import annotations

import os

import torch

BLOCK = 256


def mla_latent_rot(int8_kv: bool) -> bool:
    """QWEN_MLA_LATENT_ROT=1 / 0 forces it; unset follows QWEN_MLA_INT8_KV."""
    v = os.environ.get("QWEN_MLA_LATENT_ROT")
    return int8_kv if v is None else v == "1"


def hadamard(n: int = BLOCK, device=None) -> torch.Tensor:
    """Orthonormal Sylvester Hadamard [n, n] in fp32 (symmetric, so R^T = R)."""
    h = torch.ones(1, 1, dtype=torch.float32, device=device)     # explicit: vLLM loads under a bf16 default dtype
    while h.shape[0] < n:
        h = torch.cat([torch.cat([h, h], 1), torch.cat([h, -h], 1)], 0)
    return h / h.shape[0] ** 0.5


def rotate_rows(w: torch.Tensor) -> torch.Tensor:
    """R @ w for w [r, ...] (the latent projection's output rows), computed in fp32."""
    r = w.shape[0]
    assert r % BLOCK == 0, r
    h = hadamard(BLOCK, w.device)
    return torch.einsum("ij,bj...->bi...", h, w.float().reshape(r // BLOCK, BLOCK, *w.shape[1:])).reshape(w.shape)


def rotate_cols(w: torch.Tensor) -> torch.Tensor:
    """w @ R^T for w [..., r] (an up-projection's input columns), computed in fp32."""
    r = w.shape[-1]
    assert r % BLOCK == 0, r
    h = hadamard(BLOCK, w.device)
    return (w.float().reshape(*w.shape[:-1], r // BLOCK, BLOCK) @ h.T).reshape(w.shape)


@torch.no_grad()
def fold_latent_rotation(attn) -> None:
    """Rotate one Qwen3_5MLAAttention's latent in place: kv_a_proj rows; kv_b_proj and _k_up_raw columns.
    Must run before vLLM derives W_UK_T / W_UV from kv_b_proj (MLAAttention.process_weights_after_loading)."""
    for name in ("kv_a_proj", "kv_b_proj"):
        w = getattr(getattr(attn, name), "weight", None)
        if not isinstance(w, torch.Tensor) or not w.dtype.is_floating_point or w.dtype.itemsize < 2:
            raise NotImplementedError(f"QWEN_MLA_LATENT_ROT needs an unquantized {name} (got {type(w)})")
    a, b, k = attn.kv_a_proj.weight, attn.kv_b_proj.weight, attn._k_up_raw
    assert a.shape[0] == attn.kv_lora_rank == b.shape[1] == k.shape[1], (a.shape, b.shape, k.shape)
    a.copy_(rotate_rows(a).to(a.dtype))
    b.copy_(rotate_cols(b).to(b.dtype))
    k.copy_(rotate_cols(k).to(k.dtype))
