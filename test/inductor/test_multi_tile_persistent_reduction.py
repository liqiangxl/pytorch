# Owner(s): ["module: inductor"]

"""Tests for multi-tile persistent reductions.

When a fused reduction+epilogue kernel shares source reads between the
reduction and epilogue, the reduction dimension can be split into tiles
so shared inputs stay in registers across both phases.  These tests
verify correctness, generated code shape, and fallback behaviour.
"""

import sys
import unittest

import torch
import torch._inductor.config as inductor_config
from torch._inductor.test_case import run_tests, TestCase
from torch._inductor.utils import run_and_get_code
from torch.testing._internal.inductor_utils import GPU_TYPE, HAS_GPU_AND_TRITON


# Config to enable multi-tile persistent reductions with 2 tiles for 8192 rnumel
multi_tile_config = {
    "triton.register_tiled_persistent_reductions": True,
    "triton.register_tiled_persistent_reduction_tile_size": 4096,
    "triton.register_tiled_persistent_reduction_max_tiles": 4,
    "triton.register_tiled_persistent_reduction_min_numel": 2049,
}


@unittest.skipUnless(HAS_GPU_AND_TRITON, "requires GPU and Triton")
class TestMultiTilePersistentReduction(TestCase):
    def _run_and_check(self, fn, *args, atol=1e-4, rtol=1e-4):
        """Compile fn with multi-tile config, check correctness against eager."""
        expected = fn(*args)
        compiled = torch.compile(fn)
        with inductor_config.patch(multi_tile_config):
            actual, code = run_and_get_code(compiled, *args)
        self.assertTrue(
            torch.allclose(expected.float(), actual.float(), atol=atol, rtol=rtol),
            f"max diff: {(expected.float() - actual.float()).abs().max().item()}",
        )
        return code

    # ---- fp32 tests ----

    def test_simple_div_sum_fp32(self):
        """x / x.sum(dim=-1, keepdim=True) with fp32."""

        def fn(x):
            return x / x.sum(dim=-1, keepdim=True)

        x = torch.randn(16, 8192, device=GPU_TYPE)
        code = self._run_and_check(fn, x)

        self.assertIn("persistent_reduction", code[0])
        self.assertIn("tl.static_range(NUM_TILES)", code[0])
        self.assertNotIn("tl.range(", code[0])

    def test_simple_div_sum_bf16(self):
        """x / x.sum(dim=-1, keepdim=True) with bf16."""

        def fn(x):
            return x / x.sum(dim=-1, keepdim=True)

        x = torch.randn(16, 8192, device=GPU_TYPE, dtype=torch.bfloat16)
        code = self._run_and_check(fn, x, atol=1e-2, rtol=1e-2)

        self.assertIn("persistent_reduction", code[0])
        self.assertIn("tl.static_range(NUM_TILES)", code[0])

    def test_rmsnorm_fp32(self):
        """RMSNorm-like pattern with fp32."""

        def fn(x, weight):
            variance = x.pow(2).mean(-1, keepdim=True)
            x_normed = x * torch.rsqrt(variance + 1e-6)
            return x_normed * weight

        x = torch.randn(32, 8192, device=GPU_TYPE)
        weight = torch.randn(8192, device=GPU_TYPE)
        self._run_and_check(fn, x, weight)

    def test_rmsnorm_bf16(self):
        """RMSNorm-like pattern with bf16 input and weight."""

        def fn(x, weight):
            variance = x.to(torch.float32).pow(2).mean(-1, keepdim=True)
            x_normed = x * torch.rsqrt(variance + 1e-6)
            return x_normed * weight

        x = torch.randn(32, 8192, device=GPU_TYPE, dtype=torch.bfloat16)
        weight = torch.randn(8192, device=GPU_TYPE, dtype=torch.bfloat16)
        self._run_and_check(fn, x, weight, atol=1e-2, rtol=1e-2)

    def test_four_tiles_fp32(self):
        """rnumel=16384 with tile_size=4096 gives 4 tiles, fp32."""

        def fn(x):
            return x / x.sum(dim=-1, keepdim=True)

        x = torch.randn(8, 16384, device=GPU_TYPE)
        code = self._run_and_check(fn, x)

        self.assertIn("tl.static_range(NUM_TILES)", code[0])
        self.assertIn("'persistent_reduction_num_tiles': 4", code[0])

    def test_four_tiles_bf16(self):
        """rnumel=16384 with tile_size=4096 gives 4 tiles, bf16."""

        def fn(x):
            return x / x.sum(dim=-1, keepdim=True)

        x = torch.randn(8, 16384, device=GPU_TYPE, dtype=torch.bfloat16)
        code = self._run_and_check(fn, x, atol=1e-2, rtol=1e-2)

        self.assertIn("tl.static_range(NUM_TILES)", code[0])
        self.assertIn("'persistent_reduction_num_tiles': 4", code[0])

    # ---- retained-load test ----

    def test_no_epilogue_reload(self):
        """Epilogue should reuse retained loads, not emit a second tl.load for x."""

        def fn(x):
            return x / x.sum(dim=-1, keepdim=True)

        x = torch.randn(16, 8192, device=GPU_TYPE)
        code = self._run_and_check(fn, x)

        load_count = code[0].count("tl.load")
        self.assertEqual(load_count, 1, f"Expected 1 tl.load (inside loop), got {load_count}")

    # ---- fallback tests ----

    def test_small_rnumel_stays_single_tile(self):
        """rnumel=1024 is below min_numel — should use standard persistent, no tiling."""

        def fn(x):
            return x / x.sum(dim=-1, keepdim=True)

        x = torch.randn(16, 1024, device=GPU_TYPE)
        compiled = torch.compile(fn)
        with inductor_config.patch(multi_tile_config):
            _, code = run_and_get_code(compiled, x)

        self.assertIn("persistent_reduction", code[0])
        self.assertNotIn("tl.static_range", code[0])
        self.assertNotIn("persistent_reduction_num_tiles", code[0])

    def test_non_power_of_two_falls_back(self):
        """rnumel=6144 is not power-of-two — should not use multi-tile."""

        def fn(x):
            return x / x.sum(dim=-1, keepdim=True)

        x = torch.randn(16, 6144, device=GPU_TYPE)
        compiled = torch.compile(fn)
        with inductor_config.patch(multi_tile_config):
            _, code = run_and_get_code(compiled, x)

        self.assertNotIn("tl.static_range", code[0])

    def test_welford_falls_back(self):
        """torch.var_mean uses welford reduction — unsupported, should fall back."""

        def fn(x):
            var, mean = torch.var_mean(x, dim=-1, keepdim=True)
            return x * mean / (var + 1e-6)

        x = torch.randn(16, 8192, device=GPU_TYPE)
        expected = fn(x)
        compiled = torch.compile(fn)
        with inductor_config.patch(multi_tile_config):
            actual = compiled(x)
        self.assertTrue(
            torch.allclose(expected, actual, atol=1e-4, rtol=1e-4),
        )

    # ---- metadata / gating tests ----

    def test_metadata_emitted(self):
        """Check that inductor_meta contains tile metadata."""

        def fn(x):
            return x / x.sum(dim=-1, keepdim=True)

        x = torch.randn(16, 8192, device=GPU_TYPE)
        compiled = torch.compile(fn)
        with inductor_config.patch(multi_tile_config):
            _, code = run_and_get_code(compiled, x)

        self.assertIn("persistent_reduction_num_tiles", code[0])
        self.assertIn("persistent_reduction_tile_size", code[0])

    def test_feature_disabled_by_default(self):
        """Without the config flag, multi-tile should not activate."""

        def fn(x):
            return x / x.sum(dim=-1, keepdim=True)

        x = torch.randn(16, 8192, device=GPU_TYPE)
        compiled = torch.compile(fn)
        _, code = run_and_get_code(compiled, x)
        self.assertNotIn("tl.static_range", code[0])
        self.assertNotIn("persistent_reduction_num_tiles", code[0])


if __name__ == "__main__":
    run_tests()
