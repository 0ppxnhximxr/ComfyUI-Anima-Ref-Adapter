"""SnapRef anima_ref_kv/1 inference, adapted from ArcAnima/SnapRef (Apache-2.0).

Reference and target LoRAs stay separate. The reference runs at sigma zero with
its own text and mode embedding; target LoRA also runs on reference-free calls.
"""
from __future__ import annotations

import json
import math
import os
import weakref
from contextlib import contextmanager

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


@contextmanager
def reference_context(dm):
    # Spectrum Mod Guidance steers the target batch, never the clean reference.
    attrs = ((getattr(dm, "t_embedding_norm", None), "_anima_mod_delta"), (dm, "_anima_mod_block_state"))
    saved = [(obj, name, getattr(obj, name)) for obj, name in attrs if hasattr(obj, name)]
    try:
        for obj, name, _ in saved:
            setattr(obj, name, None)
        yield
    finally:
        for obj, name, value in saved:
            setattr(obj, name, value)


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
                else:
                    kr, vr = state["cache"][index]
                    count = kr.shape[1]
                    k = torch.cat([kr.expand(k.shape[0], *kr.shape[1:]).to(k), k], dim=1)
                    v = torch.cat([vr.expand(v.shape[0], *vr.shape[1:]).to(v), v], dim=1)
                    rows = state.get("rows")
                    if state["bias"] != 0 or rows is not None:
                        mask = q.new_zeros(q.shape[0] if rows is not None else 1, 1, 1, k.shape[1])
                        mask[..., :count] = state["bias"] if rows is None else float("-inf")
                        if rows is not None:
                            mask[rows, ..., :count] = state["bias"]
                        out = F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2),
                                                            v.transpose(1, 2), attn_mask=mask)
                        out = out.transpose(1, 2).flatten(2)
                        return attn.output_dropout(attn.output_proj(out))
                return attn.compute_attention(q, k, v, transformer_options=options)
            attn.forward = forward
        self.hooked.add(dm)


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
    emb, adaln = dm.t_embedder[1](dm.t_embedder[0](torch.zeros(ref.shape[0], 1, device=ref.device)).to(x.dtype))
    emb = dm.t_embedding_norm(emb)
    if adapter.mode_embed is not None:
        emb = emb + adapter.mode_embed[adapter.modes.index(mode)].to(emb)[None, None, :]
    record = [None] * len(dm.blocks)
    previous = _STATE.copy()
    _set_state(dm, {"record": record})
    _STATE.update(adapter=adapter, stream="ref")
    try:
        x = x.float() if x.dtype == torch.float16 else x
        for block in dm.blocks:
            x = block(x, emb, context, rope_emb_L_1_1_D=rope.unsqueeze(1).unsqueeze(0),
                      adaln_lora_B_T_3D=adaln, extra_per_block_pos_emb=None, transformer_options={})
    finally:
        _set_state(dm, None)
        _STATE.update(previous)
    return record


def _slice_batch(value, sl, batch):
    if isinstance(value, torch.Tensor):
        return value[sl] if value.ndim and value.shape[0] == batch else value
    if isinstance(value, dict):
        return {k: _slice_batch(v, sl, batch) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return type(value)(_slice_batch(v, sl, batch) for v in value)
    return value


def make_wrapper(adapter, latent, conditioning, mode, strength, in_uncond):
    def wrapper(executor, x, timesteps, context, fps=None, padding_mask=None, **kwargs):
        if "multigpu_thread_device" in kwargs.get("transformer_options", {}):
            raise ValueError("SnapRef does not support core MultiGPU sampling")
        dm = executor.class_obj
        adapter.install(dm)
        key = (dm, id(latent), id(conditioning), mode, tuple(x.shape[-2:]), x.dtype, x.device)
        if key not in adapter.cache:
            adapter.cache.clear()
            cross, meta = conditioning
            ids, weights = meta.get("t5xxl_ids"), meta.get("t5xxl_weights")
            ctx = dm.preprocess_text_embeds(cross.to(x), None if ids is None else ids.unsqueeze(0).to(x.device),
                                           None if weights is None else weights.unsqueeze(0).unsqueeze(-1).to(x))
            with reference_context(dm):
                adapter.cache[key] = _encode_reference(adapter, dm, latent.to(x), ctx, x.shape[-2:], mode)
        cache = adapter.cache[key]
        options = kwargs.get("transformer_options", {})
        flags = options.get("cond_or_uncond") or [0]
        use = [not options.get(NO_REF_KEY) and (f == 0 or in_uncond) for f in flags]

        def run(xs, ts, cs, fs, mask, with_ref, kw, rows=None):
            previous = _STATE.copy()
            _set_state(dm, {"cache": cache, "bias": math.log(strength), "rows": rows} if with_ref else None)
            _STATE.update(adapter=adapter, stream="tgt")
            try:
                return executor(xs, ts, cs, fs, mask, **kw)
            finally:
                _set_state(dm, None)
                _STATE.update(previous)
        if all(use) or not any(use):
            return run(x, timesteps, context, fps, padding_mask, use[0], kwargs)
        n = x.shape[0] // len(flags)
        if getattr(getattr(dm, "t_embedding_norm", None), "_anima_mod_delta", None) is not None \
                or getattr(dm, "_anima_mod_block_state", None) is not None:
            rows = [r for r in range(x.shape[0]) if use[r // n]]
            return run(x, timesteps, context, fps, padding_mask, True, kwargs, rows)
        outputs = []
        for i, with_ref in enumerate(use):
            sl = slice(i * n, (i + 1) * n)
            kw = _slice_batch(kwargs, sl, x.shape[0])
            kw["transformer_options"] = {**kw.get("transformer_options", {}), "cond_or_uncond": [flags[i]]}
            outputs.append(run(x[sl], _slice_batch(timesteps, sl, x.shape[0]), context[sl],
                               _slice_batch(fps, sl, x.shape[0]), _slice_batch(padding_mask, sl, x.shape[0]), with_ref, kw))
        return torch.cat(outputs)
    return wrapper


def outer_wrapper(adapter):
    def wrapper(executor, *args, **kwargs):
        try:
            return executor(*args, **kwargs)
        finally:
            adapter.release()
    return wrapper


def apply(model, clip, vae, image, path, mode, caption, strength, megapixels, guidance):
    if strength == 0:
        return (model,)
    if clip is None:
        raise ValueError("SnapRef requires the Anima CLIP input; connect the text encoder to clip")
    if not math.isfinite(strength) or strength < 0:
        raise ValueError("SnapRef strength must be finite and nonnegative")
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
                           make_wrapper(adapter, latent, cond, mode, strength, guidance))
    m.add_wrapper_with_key(comfy.patcher_extension.WrappersMP.OUTER_SAMPLE, WRAPPER_KEY, outer_wrapper(adapter))
    return (m,)
