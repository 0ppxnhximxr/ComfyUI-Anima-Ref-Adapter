"""CPU integration checks; ComfyUI imports are stubbed, tensor math and files are real."""
import importlib.util
import json
import math
from pathlib import Path
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

import torch
import torch.nn.functional as F
from safetensors.torch import load_file, save_file


ROOT = Path(__file__).resolve().parents[1]


def load_nodes():
    names = ("comfy", "comfy.conds", "comfy.ldm", "comfy.ldm.common_dit", "comfy.ldm.anima",
             "comfy.ldm.anima.model", "comfy.model_management", "comfy.patcher_extension",
             "comfy.samplers", "comfy.utils", "folder_paths", "node_helpers")
    modules = {name: ModuleType(name) for name in names}
    for name, module in modules.items():
        if "." in name:
            parent, attr = name.rsplit(".", 1)
            setattr(modules[parent], attr, module)
    modules["comfy.ldm.anima.model"].Anima = torch.nn.Module
    modules["comfy.samplers"].CFGGuider = object
    modules["comfy.patcher_extension"].WrappersMP = SimpleNamespace(DIFFUSION_MODEL="diffusion", OUTER_SAMPLE="outer")
    modules["comfy.utils"].load_torch_file = lambda path, **kw: load_file(path)
    modules["comfy.utils"].common_upscale = lambda x, w, h, *a: F.interpolate(x, size=(h, w))
    modules["folder_paths"].get_filename_list = lambda _: ["renamed.safetensors"]
    spec = importlib.util.spec_from_file_location("anima_ref_test", ROOT / "__init__.py",
                                                 submodule_search_locations=[str(ROOT)])
    node = importlib.util.module_from_spec(spec)
    modules[spec.name] = node
    with patch.dict(sys.modules, modules):
        spec.loader.exec_module(node)
        if (ROOT / "snapref.py").exists():
            backend_spec = importlib.util.spec_from_file_location("anima_ref_test.snapref", ROOT / "snapref.py")
            backend = importlib.util.module_from_spec(backend_spec)
            modules[backend_spec.name] = backend
            with patch.dict(sys.modules, modules):
                backend_spec.loader.exec_module(backend)
            node.snapref = backend
    sys.modules[spec.name] = node
    return node


nodes = load_nodes()


class Attention(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.q_proj = torch.nn.Linear(2, 2, bias=False)
        torch.nn.init.zeros_(self.q_proj.weight)

    def forward(self, x, context=None, **kwargs):
        return self.q_proj(x)


class Model(torch.nn.Module):
    def __init__(self, count=1):
        super().__init__()
        self.blocks = torch.nn.ModuleList()
        for _ in range(count):
            block = torch.nn.Module()
            block.self_attn = Attention()
            self.blocks.append(block)


def write_adapter(path, fp8=False):
    tensors = {"anima_ref_kv.mode_embed": torch.tensor([[1., 2.], [3., 4.]])}
    for stream, value in (("ref", 2.), ("tgt", 3.)):
        root = f"anima_ref_kv.{stream}.blocks.0.self_attn.q_proj."
        tensors[root + "lora_down"] = torch.ones(1, 2)
        tensors[root + "lora_up"] = torch.full((2, 1), value)
        tensors[root + "alpha"] = torch.tensor(.5)
        if fp8:
            for suffix in ("lora_down", "lora_up"):
                tensors[root + suffix] = tensors[root + suffix].to(torch.float8_e4m3fn)
                tensors[root + suffix + "_scale"] = torch.full((tensors[root + suffix].shape[0], 1), .5)
    save_file(tensors, str(path), metadata={"anima_ref_kv": json.dumps({
        "format": "anima_ref_kv/1", "modes": ["cs", "char"], "ref_t": 10})})


class IntegrationTests(unittest.TestCase):
    def test_existing_inputs_stay_in_order_and_clip_is_optional(self):
        schema = nodes.AnimaRefAdapter.INPUT_TYPES()
        self.assertEqual(list(schema["required"]),
                         ["model", "vae", "image", "lora_name", "ref_guidance", "correction"])
        self.assertEqual(list(schema["optional"])[:3], ["strength", "ref_end", "ref_megapixels"])
        self.assertIn("clip", schema["optional"])

    def test_workflow_widget_values_still_bind_to_legacy_inputs(self):
        schema = nodes.AnimaRefAdapter.INPUT_TYPES()
        for filename in ("anima_ref_adapter.json", "anima_ref_adapter_advanced.json", "anima_ref_adapter_two_characters.json"):
            path = ROOT / "workflows" / filename
            for node in json.loads(path.read_text())["nodes"]:
                if node["type"] == "AnimaRefAdapter":
                    widget_names = [k for group in schema.values() for k, v in group.items()
                                    if isinstance(v[0], list) or v[0] in ("FLOAT", "INT", "STRING", "BOOLEAN")]
                    self.assertEqual(widget_names[:6],
                                     ["lora_name", "ref_guidance", "correction", "strength", "ref_end", "ref_megapixels"])
                    self.assertEqual(node["widgets_values"][:3], ["anima_ref_adapter_v2_r64.safetensors", 4.5, "apg"])

    def test_renamed_snapref_reports_missing_clip_before_encoding(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "renamed.safetensors"
            write_adapter(path)
            with patch.object(nodes, "_lora_path", return_value=str(path)):
                with self.assertRaisesRegex(ValueError, "CLIP"):
                    nodes.AnimaRefAdapter().apply(None, None, None, path.name, 4.5, "apg")

    def test_zero_snapref_strength_is_a_true_bypass(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "renamed.safetensors"
            write_adapter(path)
            model = object()
            with patch.object(nodes, "_lora_path", return_value=str(path)):
                result = nodes.AnimaRefAdapter().apply(model, None, None, path.name, 4.5, "apg", strength=0.)
            self.assertIs(result[0], model)


class BackendTests(unittest.TestCase):
    def setUp(self):
        self.assertTrue(hasattr(nodes, "snapref"), "SnapRef backend is missing")
        self.backend = nodes.snapref
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name) / "adapter.safetensors"
        write_adapter(self.path)
        self.adapter = self.backend.Adapter(str(self.path))
        self.addCleanup(self.backend._STATE.update, adapter=None, stream=None)

    def test_stream_hooks_are_isolated_and_reload_without_duplicate_deltas(self):
        for _ in range(2):
            model = Model()
            self.adapter.install(model)
            self.adapter.install(model)
            for stream, expected in (("ref", 2.), ("tgt", 3.), (None, 0.)):
                self.backend._STATE.update(adapter=self.adapter, stream=stream)
                actual = model.blocks[0].self_attn.q_proj(torch.ones(1, 2))
                torch.testing.assert_close(actual, torch.full((1, 2), expected))

    def test_fp8_scales_are_applied_before_lora_delta(self):
        write_adapter(self.path, fp8=True)
        adapter = self.backend.Adapter(str(self.path))
        model = Model()
        adapter.install(model)
        self.backend._STATE.update(adapter=adapter, stream="tgt")
        torch.testing.assert_close(model.blocks[0].self_attn.q_proj(torch.ones(1, 2)), torch.full((1, 2), .75))

    def test_incompatible_model_installs_no_partial_hooks(self):
        model = Model(2)
        with self.assertRaises(ValueError):
            self.adapter.install(model)
        self.assertFalse(model.blocks[0].self_attn.q_proj._forward_hooks)

    def test_sampling_exit_clears_cache_even_on_error(self):
        self.adapter.cache["sample"] = object()
        def fail():
            raise RuntimeError("sampling failed")
        with self.assertRaisesRegex(RuntimeError, "sampling failed"):
            self.backend.outer_wrapper(self.adapter)(fail)
        self.assertFalse(self.adapter.cache)

    def test_core_multigpu_is_rejected_before_touching_the_model(self):
        wrapper = self.backend.make_wrapper(self.adapter, None, None, 'cs', 1., False)
        with self.assertRaisesRegex(ValueError, 'MultiGPU'):
            wrapper(None, None, None, None, transformer_options={'multigpu_thread_device': 'cuda:1'})

    def test_mixed_cfg_slices_extras_and_keeps_target_lora_without_reference(self):
        model = Model()
        model.preprocess_text_embeds = lambda cross, ids, weights: cross
        seen = []
        backend = self.backend
        class Executor:
            class_obj = model
            def __call__(self, x, t, context, fps, mask, **kw):
                state = getattr(model.blocks[0].self_attn, backend.STATE_ATTR)
                flag = kw['transformer_options']['cond_or_uncond']
                seen.append((flag, kw['control']['input'][0].tolist(), mask.tolist(), state is not None))
                return model.blocks[0].self_attn.q_proj(x) + (100 if state else 0)
        wrapper = backend.make_wrapper(self.adapter, torch.ones(1, 2), (torch.ones(1, 1, 2), {}), 'cs', 1., False)
        with patch.object(backend, '_encode_reference', return_value=[(torch.ones(1), torch.ones(1))]):
            out = wrapper(Executor(), torch.ones(2, 2), torch.ones(2), torch.ones(2, 1, 2),
                          padding_mask=torch.tensor([10, 20]), control={'input': [torch.tensor([30, 40])]},
                          transformer_options={'cond_or_uncond': [0, 1]})
        torch.testing.assert_close(out, torch.tensor([[103., 103.], [3., 3.]]))
        self.assertEqual(seen, [([0], [30], [10], True), ([1], [40], [20], False)])
        self.assertIsNone(getattr(model.blocks[0].self_attn, backend.STATE_ATTR))
        self.assertEqual(backend._STATE, {'adapter': None, 'stream': None})

    def test_reference_free_call_retains_target_lora_and_cleans_up_on_error(self):
        model = Model()
        model.preprocess_text_embeds = lambda cross, ids, weights: cross
        backend = self.backend
        class Executor:
            class_obj = model
            def __call__(self, x, *a, **kw):
                self_output = model.blocks[0].self_attn.q_proj(x)
                torch.testing.assert_close(self_output, torch.full((1, 2), 3.))
                if getattr(model.blocks[0].self_attn, backend.STATE_ATTR) is not None:
                    raise AssertionError('reference was not disabled')
                raise RuntimeError('target failed')
        wrapper = backend.make_wrapper(self.adapter, torch.ones(1, 2), (torch.ones(1, 1, 2), {}), 'cs', 1., True)
        with patch.object(backend, '_encode_reference', return_value=[(torch.ones(1), torch.ones(1))]):
            with self.assertRaisesRegex(RuntimeError, 'target failed'):
                wrapper(Executor(), torch.ones(1, 2), torch.ones(1), torch.ones(1, 1, 2),
                        transformer_options={backend.NO_REF_KEY: True})
        self.assertIsNone(getattr(model.blocks[0].self_attn, backend.STATE_ATTR))
        self.assertEqual(backend._STATE, {'adapter': None, 'stream': None})

    def test_mod_guidance_keeps_cfg_batch_and_is_suspended_only_for_reference(self):
        model = Model()
        delta = torch.tensor([[[10., 10.]], [[20., 20.]]])
        block_state = object()
        model.t_embedding_norm = SimpleNamespace(_anima_mod_delta=delta)
        model._anima_mod_block_state = block_state
        model.preprocess_text_embeds = lambda cross, ids, weights: cross
        backend = self.backend
        calls = []
        case = self
        class Executor:
            class_obj = model
            def __call__(self, x, *a, **kw):
                case.assertEqual(x.shape[0], 2, 'Mod Guidance owns a two-row CFG batch')
                case.assertIs(model.t_embedding_norm._anima_mod_delta, delta)
                case.assertIs(model._anima_mod_block_state, block_state)
                state = getattr(model.blocks[0].self_attn, backend.STATE_ATTR)
                calls.append(state['rows'])
                return x
        def encode(*args):
            case.assertIsNone(model.t_embedding_norm._anima_mod_delta)
            case.assertIsNone(model._anima_mod_block_state)
            return [(torch.ones(1), torch.ones(1))]
        wrapper = backend.make_wrapper(self.adapter, torch.ones(1, 2), (torch.ones(1, 1, 2), {}), 'cs', 1., False)
        with patch.object(backend, '_encode_reference', side_effect=encode):
            result = wrapper(Executor(), torch.ones(2, 2), torch.ones(2), torch.ones(2, 1, 2),
                             transformer_options={'cond_or_uncond': [0, 1]})
        self.assertEqual(result.shape[0], 2)
        self.assertEqual(calls, [[0]])

    def test_reference_mod_guidance_state_is_restored_on_failure(self):
        model = Model()
        delta, block_state = object(), object()
        model.t_embedding_norm = SimpleNamespace(_anima_mod_delta=delta)
        model._anima_mod_block_state = block_state
        model.preprocess_text_embeds = lambda *a: torch.ones(1)
        wrapper = self.backend.make_wrapper(self.adapter, torch.ones(1), (torch.ones(1), {}), 'cs', 1., False)
        with patch.object(self.backend, '_encode_reference', side_effect=RuntimeError('reference failed')):
            with self.assertRaisesRegex(RuntimeError, 'reference failed'):
                wrapper(SimpleNamespace(class_obj=model), torch.ones(2, 2), None, None)
        self.assertIs(model.t_embedding_norm._anima_mod_delta, delta)
        self.assertIs(model._anima_mod_block_state, block_state)

    def test_mixed_reference_mask_keeps_negative_rows_reference_free(self):
        model = Model()
        self.adapter.install(model)
        attn = model.blocks[0].self_attn
        q = torch.zeros(2, 1, 1, 2)
        attn.compute_qkv = lambda *a, **kw: (q, q, q)
        attn.output_proj = attn.output_dropout = torch.nn.Identity()
        self.backend._set_state(model, {'cache': [(torch.zeros(1, 1, 1, 2), torch.full((1, 1, 1, 2), 2.))],
                                        'bias': math.log(.5), 'rows': [0]})
        self.addCleanup(self.backend._set_state, model, None)
        actual = attn(torch.zeros(2, 1, 2))
        torch.testing.assert_close(actual, torch.tensor([[[2/3, 2/3]], [[0., 0.]]]))


if __name__ == "__main__":
    unittest.main()
