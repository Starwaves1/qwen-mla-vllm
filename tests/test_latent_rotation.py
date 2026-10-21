"""Block-Hadamard latent rotation (latent_rot.py) leaves one MLA layer's attention unchanged in bf16.

Folds R into a real checkpoint layer's kv_a_proj / k_up / v_up exactly as fold_latent_rotation does at load (fp32
products, rounded back to bf16), then runs the absorbed decode attention (scores q.c, rms divisor from the raw k_up,
values W_UV o) in bf16 with and without the rotation, against an fp32 unrotated reference. Also prints how much the
rotation shrinks the per-token int8 quantization error of the latent.

    MLA_CKPT=/path/to/checkpoint MLA_LAYER=11 python tests/test_latent_rotation.py
"""
import json
import os
import types

import torch
from safetensors import safe_open

from qwen_mla.latent_rot import fold_latent_rotation, rotate_cols, rotate_rows

CKPT = os.environ.get("MLA_CKPT", "/home/garrett/models/TelperionAI-Qwen3.8-27B-MLA-exl3-4.50bpw-H4")
LAYER = int(os.environ.get("MLA_LAYER", 11))


def load(name):
    idx = json.load(open(os.path.join(CKPT, "model.safetensors.index.json")))["weight_map"]
    key = next(k for k in idx if k.endswith(f"layers.{LAYER}.self_attn.{name}.weight"))
    with safe_open(os.path.join(CKPT, idx[key]), "pt", device="cuda") as f:
        return f.get_tensor(key)


def attention(wa, k_up, v_up, x, q_nope, dt):
    """Absorbed MLA decode for 24 heads over T tokens (latent part only, the rope term is unchanged by R)."""
    nh = q_nope.shape[0]
    c = (x.to(dt) @ wa.to(dt).T)                                  # [T, r]
    K = k_up.to(dt).view(nh, -1, wa.shape[0])                     # [h, nope, r]
    V = v_up.to(dt).view(nh, -1, wa.shape[0])                     # [h, 256, r]
    rms = (torch.einsum("tr,hnr->thn", c, K).float().pow(2).mean(-1) + 1e-6).sqrt()   # raw-k_up rms, per (token, head)
    ql = torch.einsum("hn,hnr->hr", q_nope.to(dt), K)             # absorbed query
    s = (ql @ c.T).float() / rms.T / 16.0
    o = torch.softmax(s, -1).to(dt) @ c                           # [h, r] latent output
    return torch.einsum("hr,hdr->hd", o, V).float(), c.float()


def int8_err(c):
    s = c.abs().amax(-1, keepdim=True) / 127
    return ((c / s).round().clamp(-127, 127) * s - c).norm() / c.norm()


def main():
    torch.manual_seed(0)
    wa, k_up, v_up = load("kv_a_proj"), load("k_up_proj"), load("v_up_proj")
    r = wa.shape[0]
    x = torch.randn(4096, wa.shape[1], device="cuda")
    q_nope = torch.randn(24, k_up.shape[0] // 24, device="cuda")
    # the fold as done at load, through the plugin's function, on a stand-in layer
    attn = types.SimpleNamespace(kv_a_proj=types.SimpleNamespace(weight=wa.clone()), kv_lora_rank=r,
                                 kv_b_proj=types.SimpleNamespace(weight=torch.cat([k_up, v_up]).clone()),
                                 _k_up_raw=k_up.clone())
    fold_latent_rotation(attn)
    wa_r, kv_b_r = attn.kv_a_proj.weight, attn.kv_b_proj.weight
    k_r, v_r = kv_b_r[: k_up.shape[0]], kv_b_r[k_up.shape[0]:]
    assert torch.equal(attn._k_up_raw, k_r)
    ref, c32 = attention(wa, k_up, v_up, x, q_nope, torch.float32)
    base, _ = attention(wa, k_up, v_up, x, q_nope, torch.bfloat16)
    rot, c_rot = attention(wa_r, k_r, v_r, x, q_nope, torch.bfloat16)
    rel = lambda a, b: ((a - b).abs().max() / b.abs().max()).item()
    print(f"layer {LAYER} rank {r}: bf16 unrotated vs fp32 {rel(base, ref):.2e} | bf16 rotated vs fp32 {rel(rot, ref):.2e}"
          f" | rotated vs unrotated (bf16) {rel(rot, base):.2e}")
    # the rotated latent is R c (to bf16 rounding)
    print(f"latent: |R c - c_rot| / |c| = {((rotate_cols(c32) - c_rot).norm() / c32.norm()).item():.2e};"
          f" per-token int8 error {int8_err(c32):.3e} -> {int8_err(c_rot):.3e} rotated")
    assert rel(rot, ref) < 2 * rel(base, ref) + 1e-3, "rotation changed the attention output beyond bf16 rounding"
    assert torch.allclose(rotate_rows(rotate_rows(wa.float())), wa.float(), atol=1e-4)   # R R = I


if __name__ == "__main__":
    main()
