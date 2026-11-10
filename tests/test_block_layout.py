"""CPU regression tests for real adapter hooks; no ComfyUI server or checkpoints required."""
import ast
from pathlib import Path
import unittest
from types import SimpleNamespace

import torch
import torch.nn.functional as F

SOURCE = Path(__file__).resolve().parents[1] / "__init__.py"


class AnimaBase(torch.nn.Module):
    pass


class Attention(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.q_proj = torch.nn.Linear(2, 2, bias=False)
        torch.nn.init.zeros_(self.q_proj.weight)

    def forward(self, x, **kwargs):
        return self.q_proj(x)

    def compute_qkv(self, x, context, **kwargs):
        return x, x, x

    def compute_attention(self, q, k, v, **kwargs):
        return q


class TinyAnima(AnimaBase):
    def __init__(self, count):
        super().__init__()
        self.blocks = torch.nn.ModuleList()
        for _ in range(count):
            block = torch.nn.Module()
            block.self_attn = Attention()
            self.blocks.append(block)


def load_production(source):
    tree = ast.parse(source)
    names = {"_adapter_block_map", "_Adapter", "_set_attn_state", "_encode_ref_asym", "_asym_forward"}
    nodes = [n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name in names]
    ns = {"torch": torch, "F": F, "AnimaModel": AnimaBase,
          "_INSERTED_29B": frozenset((2, 5, 8, 11, 14, 17, 21, 24, 27, 30, 33, 36)),
          "HOOKS_ATTR": "_anima_ref_adapter_hooks", "_active": {"key": None},
          "_STATE": {"adapter": None, "stream": None}}
    module = ast.Module(body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)] + nodes,
                        type_ignores=[])
    exec(compile(ast.fix_missing_locations(module), str(SOURCE), "exec"), ns)
    return ns


NS = load_production(SOURCE.read_text(encoding="utf-8"))
# Explicit expected destinations, independent of the production map construction.
EXPECTED = (0, 1, 3, 4, 6, 7, 9, 10, 12, 13, 15, 16, 18, 19,
            20, 22, 23, 25, 26, 28, 29, 31, 32, 34, 35, 37, 38, 39)


def adapter(count=28):
    obj = object.__new__(NS["_Adapter"])
    weights = {}
    for i in range(count):
        entry = [torch.ones(1, 2), torch.full((2, 1), float(i + 1)), 0.5]
        weights[f"blocks.{i}.self_attn.q_proj"] = entry
    obj.block_indices = set(range(count))
    obj.weights, obj.key = weights, ("adapter", 1.0)
    obj.ref_cache, obj.on_device = {}, {}
    return obj


def active(obj, enabled=True, stream="tgt"):
    NS["_active"]["key"] = obj.key if enabled else None


def result(model, i):
    return model.blocks[i].self_attn.q_proj(torch.ones(1, 2))


class BlockLayoutTests(unittest.TestCase):
    def test_native_layouts(self):
        for count in (28, 40):
            self.assertEqual(NS["_adapter_block_map"](range(count), TinyAnima(count)), tuple(range(count)))

    def test_expansion_map(self):
        self.assertEqual(NS["_adapter_block_map"](range(28), TinyAnima(40)), EXPECTED)

    def test_unknown_and_incomplete_layouts(self):
        for indices, count in ((range(28), 36), (range(40), 28), (range(27), 40), ([], 40)):
            with self.assertRaises(ValueError):
                NS["_adapter_block_map"](indices, TinyAnima(count))

    def test_other_model_type_not_guessed(self):
        with self.assertRaises(ValueError):
            NS["_adapter_block_map"](range(28), SimpleNamespace(blocks=[None] * 40))


class HookTests(unittest.TestCase):
    def tearDown(self):
        NS["_STATE"].update(adapter=None, stream=None)
        NS["_active"]["key"] = None

    def test_native_hooks_preserve_layer_values(self):
        model, ad = TinyAnima(28), adapter()
        ad.install(model)
        active(ad)
        for i in range(28):
            torch.testing.assert_close(result(model, i), torch.full((1, 2), float(i + 1)))

    def test_expanded_hooks_land_on_original_layers(self):
        model, ad = TinyAnima(40), adapter()
        ad.install(model)
        active(ad)
        for src, dst in enumerate(EXPECTED):
            torch.testing.assert_close(result(model, dst), torch.full((1, 2), float(src + 1)))
        for i in set(range(40)) - set(EXPECTED):
            torch.testing.assert_close(result(model, i), torch.zeros(1, 2))

    def test_repeated_install_does_not_double_delta(self):
        model, ad = TinyAnima(40), adapter()
        ad.install(model)
        ad.install(model)
        active(ad)
        torch.testing.assert_close(result(model, 39), torch.full((1, 2), 28.0))

    def test_one_adapter_can_serve_both_models_without_mutating_keys(self):
        ad = adapter()
        before = tuple(ad.weights)
        base, expanded = TinyAnima(28), TinyAnima(40)
        ad.install(expanded)
        ad.install(base)
        active(ad)
        torch.testing.assert_close(result(base, 2), torch.full((1, 2), 3.0))
        torch.testing.assert_close(result(expanded, 3), torch.full((1, 2), 3.0))
        self.assertEqual(before, tuple(ad.weights))

    def test_device_copies_are_released_and_cpu_weights_kept(self):
        model, ad = TinyAnima(28), adapter()
        cpu = {name: entry[1] for name, entry in ad.weights.items()}
        ad.install(model)
        active(ad)
        result(model, 0)
        self.assertTrue(ad.on_device)
        ad.release()
        self.assertFalse(ad.on_device)
        for name, entry in ad.weights.items():
            self.assertIs(entry[1], cpu[name])
        torch.testing.assert_close(result(model, 0), torch.full((1, 2), 1.0))

    def test_inactive_adapter_does_not_change_model(self):
        model, ad = TinyAnima(40), adapter()
        ad.install(model)
        active(ad, False)
        for i in range(40):
            torch.testing.assert_close(result(model, i), torch.zeros(1, 2))

    def test_native_40_block_weights_are_not_remapped(self):
        model, ad = TinyAnima(40), adapter(40)
        ad.install(model)
        active(ad)
        torch.testing.assert_close(result(model, 2), torch.full((1, 2), 3.0))
        torch.testing.assert_close(result(model, 39), torch.full((1, 2), 40.0))

    def test_incompatible_layout_does_not_install_hooks(self):
        model, ad = TinyAnima(36), adapter()
        with self.assertRaises(ValueError):
            ad.install(model)
        self.assertFalse(any(b.self_attn.q_proj._forward_hooks for b in model.blocks))

    def test_missing_target_does_not_leave_partial_hooks(self):
        model, ad = TinyAnima(40), adapter()
        ad.block_indices = set(range(28))
        weights = ad.weights
        weights["blocks.27.missing"] = next(iter(weights.values()))
        with self.assertRaises(AttributeError):
            ad.install(model)
        self.assertFalse(any(b.self_attn.q_proj._forward_hooks for b in model.blocks))
        del weights["blocks.27.missing"]
        ad.install(model)
        active(ad)
        torch.testing.assert_close(result(model, 39), torch.full((1, 2), 28.0))


class ReferenceCacheTests(unittest.TestCase):
    def test_reference_encode_and_target_read_keep_40_cache_slots(self):
        model = TinyAnima(40)
        seen = []
        indices = {id(block): i for i, block in enumerate(model.blocks)}
        def frame_embed(dm, latent, *args):
            return latent, None, None, None
        def block_forward(block, xt, xr, *args):
            i = indices[id(block)]
            kv = args[-2] if xt is not None else (torch.full((1, 1), float(i)), torch.full((1, 1), float(i)))
            if xt is not None:
                self.assertEqual(kv[0].item(), i)
                seen.append(i)
            return xt, xr, kv
        saved = {k: NS.get(k) for k in ("_frame_embed", "_asym_block", "comfy")}
        NS["_frame_embed"], NS["_asym_block"] = frame_embed, block_forward
        NS["comfy"] = SimpleNamespace(ldm=SimpleNamespace(common_dit=SimpleNamespace(pad_to_patch_size=lambda x, p: x)))
        model.patch_temporal = model.patch_spatial = 1
        model.final_layer = lambda x, *args, **kwargs: x
        model.unpatchify = lambda x: x
        try:
            x = torch.zeros(1, 2, 1, 1, 1)
            cache = NS["_encode_ref_asym"](model, x, (0, 0))
            self.assertEqual(len(cache), 40)
            NS["_asym_forward"](model, x, torch.ones(1), torch.zeros(1), cache, {})
            self.assertEqual(seen, list(range(40)))
        finally:
            NS.update(saved)


if __name__ == "__main__":
    unittest.main()
