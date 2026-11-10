"""SnapRef anima_ref_kv/1 inference, adapted from ArcAnima/SnapRef (Apache-2.0).

Reference and target LoRAs stay separate. The reference runs at sigma zero with
its own text and mode embedding; target LoRA also runs on reference-free calls.

Other extensions are reached only through ComfyUI's contracts, never their internals:
  * the reference stream calls the embedder, norm and block modules' forward directly, so
    module hooks that steer the target (e.g. Spectrum Mod Guidance) do not reach it; hooks on
    their submodules, such as this adapter's LoRA hooks, still run;
  * the model call keeps its batch whole, so per-row state set by APPLY_MODEL wrappers stays
    aligned; in a mixed CFG batch the rows without the reference skip it inside self-attention.
"""
from __future__ import annotations

import json
import math
import os
import weakref

import torch
import torch.nn.functional as F
from safetensors import safe_open
from safetensors.torch import load_file

import comfy.patcher_extension
import comfy.utils

from . import _adapter_block_map, _rope_from_positions, _frame_positions

WRAPPER_KEY = "anima_ref_snapref"
STATE_ATTR = "_anima_ref_snapref_state"
NO_REF_KEY = "anima_ref_off"
_STATE = {"adapter": None, "stream": None}
_adapters = {}


def metadata(path):
    if not path.lower().endswith(".safetensors"):
        return None
    with safe_open(path, framework="pt") as handle:
        fields = handle.metadata() or {}
    raw = fields.get("snapref") or fields.get("anima_ref_kv")
    if raw is None:
        return None
    meta = json.loads(raw)
    if meta.get("format") != "anima_ref_kv/1":
        raise ValueError(f"Unsupported SnapRef format: {meta.get('format')}")
    return meta


class Adapter:
    def __init__(self, path):
        self.meta = metadata(path)
        if self.meta is None:
            raise ValueError("Not a SnapRef anima_ref_kv/1 adapter")
        tensors = load_file(path)
        self.ref_t = int(self.meta.get("ref_t", 10))
        self.modes = self.meta.get("modes", ["cs"])
        self.mode_embed = tensors.get("anima_ref_kv.mode_embed")
        self.lora = {}
        for key in tensors:
            if not key.endswith(".lora_down"):
                continue
            root = key[:-len("lora_down")]
            _, stream, name = root[:-1].split(".", 2)
            if stream not in ("ref", "tgt"):
                raise ValueError(f"Unknown SnapRef stream: {stream}")
            weights = []
            for suffix in ("lora_down", "lora_up"):
                value = tensors[root + suffix]
                if root + suffix + "_scale" in tensors:
                    value = value.float() * tensors[root + suffix + "_scale"]
                weights.append(value.to(torch.bfloat16))
            self.lora.setdefault(name, {})[stream] = [*weights, float(tensors[root + "alpha"]) / weights[0].shape[0]]
        self.block_indices = {int(n.split(".")[1]) for n in self.lora if n.startswith("blocks.")}
        self.hooked = weakref.WeakSet()
        self.cache = {}
        self.on_device = {}  # (name, stream) -> device copies; the CPU weights above are never replaced

    def delta(self, name, stream, x):
        weights = self.on_device.get((name, stream))
        if weights is None or weights[0].device != x.device or weights[0].dtype != x.dtype:
            down, up, scale = self.lora[name][stream]
            weights = self.on_device[(name, stream)] = (down.to(x), up.to(x), scale)
        return F.linear(F.linear(x, weights[0]), weights[1]) * weights[2]

    def release(self):
        """Drop the reference K/V and the device copies of the LoRA weights (ComfyUI does not track either)."""
        self.cache.clear()
        self.on_device.clear()

    def install(self, dm):
        block_map = _adapter_block_map(self.block_indices, dm)
        if dm in self.hooked:
            return
        targets = []
        for name in self.lora:
            path = name
            if name.startswith("blocks."):
                _, index, suffix = name.split(".", 2)
                path = f"blocks.{block_map[int(index)]}.{suffix}"
            targets.append((dm.get_submodule(path), name))
        for module, name in targets:
            def hook(module, inputs, output, name=name):
                if _STATE["adapter"] is not self or _STATE["stream"] not in self.lora[name]:
                    return output
                return output + self.delta(name, _STATE["stream"], inputs[0])
            module.register_forward_hook(hook)
        for index, block in enumerate(dm.blocks):
            attn = block.self_attn
            if hasattr(attn, STATE_ATTR):
                continue
            setattr(attn, STATE_ATTR, None)
            original = attn.forward
            def forward(x, context=None, rope_emb=None, transformer_options=None,
                        attn=attn, index=index, original=original):
                options = transformer_options or {}
                state = getattr(attn, STATE_ATTR)
                if state is None:
                    return original(x, context, rope_emb=rope_emb, transformer_options=options)
                q, k, v = attn.compute_qkv(x, context, rope_emb=rope_emb, transformer_options=options)
                if "record" in state:
                    state["record"][index] = (k, v)
                    return attn.compute_attention(q, k, v, transformer_options=options)
                kr, vr = state["cache"][index]
                rows = state["rows"]
                if rows is None:
                    return _attend_reference(attn, q, k, v, kr, vr, state["bias"], options)
                plain = state["plain"]
                with_ref = _attend_reference(attn, q[rows], k[rows], v[rows], kr, vr, state["bias"], options)
                without = attn.compute_attention(q[plain], k[plain], v[plain], transformer_options=options)
                out = with_ref.new_empty((q.shape[0], *with_ref.shape[1:]))
                out[rows], out[plain] = with_ref, without
                return out
            attn.forward = forward
        self.hooked.add(dm)


def _attend_reference(attn, q, k, v, kr, vr, bias, options):
    """Self-attention over [reference; own] keys; bias (log of ref_weight) is added to the reference logits."""
    count = kr.shape[1]
    k = torch.cat([kr.expand(k.shape[0], *kr.shape[1:]).to(k), k], dim=1)
    v = torch.cat([vr.expand(v.shape[0], *vr.shape[1:]).to(v), v], dim=1)
    if bias == 0:
        return attn.compute_attention(q, k, v, transformer_options=options)
    mask = q.new_zeros(1, 1, 1, k.shape[1])
    mask[..., :count] = bias
    out = F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), attn_mask=mask)
    return attn.output_dropout(attn.output_proj(out.transpose(1, 2).flatten(2)))


def _set_state(dm, state):
    for block in dm.blocks:
        setattr(block.self_attn, STATE_ATTR, state)


@torch.no_grad()
def _encode_reference(adapter, dm, ref, context, target_hw, mode):
    x, _, _ = dm.prepare_embedded_sequence(ref, fps=None, padding_mask=None)
    _, _, hp, wp, _ = x.shape
    h0 = round((target_hw[0] // 2) / 2 - hp / 2)
    w0 = round((target_hw[1] // 2) / 2 - wp / 2)
    rope = _rope_from_positions(dm.pos_embedder, *_frame_positions(adapter.ref_t, hp, wp, h0, w0, ref.device))
    # .forward, not __call__: hooks other extensions put on these modules steer the target, not the reference
    sinusoid = dm.t_embedder[0].forward(torch.zeros(ref.shape[0], 1, device=ref.device))
    emb, adaln = dm.t_embedder[1].forward(sinusoid.to(x.dtype))
    emb = dm.t_embedding_norm.forward(emb)
    if adapter.mode_embed is not None:
        emb = emb + adapter.mode_embed[adapter.modes.index(mode)].to(emb)[None, None, :]
    record = [None] * len(dm.blocks)
    previous = _STATE.copy()
    _set_state(dm, {"record": record})
    _STATE.update(adapter=adapter, stream="ref")
    try:
        x = x.float() if x.dtype == torch.float16 else x
        for block in dm.blocks:
            x = block.forward(x, emb, context, rope_emb_L_1_1_D=rope.unsqueeze(1).unsqueeze(0),
                              adaln_lora_B_T_3D=adaln, extra_per_block_pos_emb=None, transformer_options={})
    finally:
        _set_state(dm, None)
        _STATE.update(previous)
    return record


def _reference_kv(adapter, dm, latent, conditioning, mode, x):
    """Per-block reference K/V for this target size, encoded once per sampling run."""
    key = (dm, id(latent), id(conditioning), mode, tuple(x.shape[-2:]), x.dtype, x.device)
    if key not in adapter.cache:
        adapter.cache.clear()
        cross, meta = conditioning
        ids, weights = meta.get("t5xxl_ids"), meta.get("t5xxl_weights")
        ctx = dm.preprocess_text_embeds(cross.to(x), None if ids is None else ids.unsqueeze(0).to(x.device),
                                        None if weights is None else weights.unsqueeze(0).unsqueeze(-1).to(x))
        adapter.cache[key] = _encode_reference(adapter, dm, latent.to(x), ctx, x.shape[-2:], mode)
    return adapter.cache[key]


def make_wrapper(adapter, latent, conditioning, mode, weight, in_uncond):
    bias = math.log(weight)

    def wrapper(executor, x, timesteps, context, fps=None, padding_mask=None, **kwargs):
        options = kwargs.get("transformer_options", {})
        if "multigpu_thread_device" in options:
            raise ValueError("SnapRef does not support core MultiGPU sampling")
        dm = executor.class_obj
        adapter.install(dm)
        flags = options.get("cond_or_uncond") or [0]
        use = [not options.get(NO_REF_KEY) and (f == 0 or in_uncond) for f in flags]
        state = None
        if any(use):
            state = {"cache": _reference_kv(adapter, dm, latent, conditioning, mode, x), "bias": bias, "rows": None}
            if not all(use):
                n = x.shape[0] // len(flags)
                row_use = [use[r // n] for r in range(x.shape[0])]
                state["rows"] = torch.tensor([r for r, u in enumerate(row_use) if u], device=x.device)
                state["plain"] = torch.tensor([r for r, u in enumerate(row_use) if not u], device=x.device)
        previous = _STATE.copy()
        _set_state(dm, state)
        _STATE.update(adapter=adapter, stream="tgt")
        try:
            return executor(x, timesteps, context, fps, padding_mask, **kwargs)
        finally:
            _set_state(dm, None)
            _STATE.update(previous)
    return wrapper


def outer_wrapper(adapter):
    def wrapper(executor, *args, **kwargs):
        try:
            return executor(*args, **kwargs)
        finally:
            adapter.release()
    return wrapper


def apply(model, clip, vae, image, path, mode, caption, weight, megapixels, guidance):
    """weight: linear weight of the reference keys in attention (1 = trained, 0 = the model unchanged)."""
    if weight == 0:
        return (model,)
    if clip is None:
        raise ValueError("SnapRef requires the Anima CLIP input; connect the text encoder to clip")
    key = (os.path.abspath(path), os.stat(path).st_mtime_ns)
    if key not in _adapters:
        _adapters[key] = Adapter(path)
    adapter = _adapters[key]
    if mode not in adapter.modes:
        raise ValueError(f"SnapRef mode {mode!r} is not supported by this adapter ({adapter.modes})")
    _adapter_block_map(adapter.block_indices, model.get_model_object("diffusion_model"))
    pixels = image[:1].movedim(-1, 1)
    h, w = pixels.shape[-2:]
    scale = min(1., (megapixels * 1024 * 1024 / (h * w)) ** .5)
    width, height = max(16, int(w * scale) // 16 * 16), max(16, int(h * scale) // 16 * 16)
    pixels = comfy.utils.common_upscale(pixels, width, height, "lanczos", "disabled").movedim(1, -1)
    latent = vae.encode(pixels[..., :3])
    if latent.ndim == 4:
        latent = latent.unsqueeze(2)
    latent = model.get_model_object("process_latent_in")(latent)
    cond = clip.encode_from_tokens_scheduled(clip.tokenize(caption.strip()))[0]
    m = model.clone()
    m.add_wrapper_with_key(comfy.patcher_extension.WrappersMP.DIFFUSION_MODEL, WRAPPER_KEY,
                           make_wrapper(adapter, latent, cond, mode, weight, guidance))
    m.add_wrapper_with_key(comfy.patcher_extension.WrappersMP.OUTER_SAMPLE, WRAPPER_KEY, outer_wrapper(adapter))
    return (m,)
