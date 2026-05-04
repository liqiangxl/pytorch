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
    "triton.register_tiled_persistent_reduction_max_tiles": 8,
    "triton.register_tiled_persistent_reduction_min_numel": 2049,
}


@unittest.skipUnless(HAS_GPU_AND_TRITON, "requires GPU and Triton")
class TestMultiTilePersistentReduction(TestCase):
    def _run_and_check(self, fn, *args):
        """Compile fn with multi-tile config, check correctness against eager."""
        expected = fn(*args)
        compiled = torch.compile(fn)
        with inductor_config.patch(multi_tile_config):
            actual, code = run_and_get_code(compiled, *args)
        self.assertTrue(
            torch.allclose(expected, actual, atol=1e-4, rtol=1e-4),
            f"max diff: {(expected - actual).abs().max().item()}",
        )
        return code

    def test_simple_div_sum(self):
        """x / x.sum(dim=-1, keepdim=True) — the canonical shared-read pattern."""

        def fn(x):
            return x / x.sum(dim=-1, keepdim=True)

        x = torch.randn(16, 8192, device=GPU_TYPE)
        code = self._run_and_check(fn, x)

        # Should use persistent_reduction heuristic (not plain reduction)
        self.assertIn("persistent_reduction", code[0])
        # Should have static tile offsets (no tl.range reduction loop)
        self.assertIn("r0_offset = 0", code[0])
        self.assertIn("r0_offset = 4096", code[0])
        self.assertNotIn("tl.range", code[0])

    def test_rmsnorm_pattern(self):
        """RMSNorm-like: x * rsqrt(mean(x^2)) — the motivating use case."""

        def fn(x, weight):
            variance = x.to(torch.float32).pow(2).mean(-1, keepdim=True)
            x_normed = x * torch.rsqrt(variance + 1e-6)
            return x_normed * weight

        x = torch.randn(32, 8192, device=GPU_TYPE, dtype=torch.bfloat16)
        weight = torch.randn(8192, device=GPU_TYPE, dtype=torch.bfloat16)
        expected = fn(x, weight)
        compiled = torch.compile(fn)
        with inductor_config.patch(multi_tile_config):
            actual = compiled(x, weight)
        self.assertTrue(
            torch.allclose(expected.float(), actual.float(), atol=1e-2, rtol=1e-2),
            f"max diff: {(expected.float() - actual.float()).abs().max().item()}",
        )

    def test_four_tiles(self):
        """rnumel=16384 with tile_size=4096 gives 4 tiles."""

        def fn(x):
            return x / x.sum(dim=-1, keepdim=True)

        x = torch.randn(8, 16384, device=GPU_TYPE)
        code = self._run_and_check(fn, x)

        # 4 tile offsets: 0, 4096, 8192, 12288
        self.assertIn("r0_offset = 0", code[0])
        self.assertIn("r0_offset = 4096", code[0])
        self.assertIn("r0_offset = 8192", code[0])
        self.assertIn("r0_offset = 12288", code[0])

    def test_no_epilogue_reload(self):
        """Epilogue should reuse retained loads, not emit a second tl.load for x."""

        def fn(x):
            return x / x.sum(dim=-1, keepdim=True)

        x = torch.randn(16, 8192, device=GPU_TYPE)
        code = self._run_and_check(fn, x)

        # Count tl.load calls — should be 2 (one per tile), not 4
        load_count = code[0].count("tl.load")
        self.assertEqual(load_count, 2, f"Expected 2 loads (one per tile), got {load_count}")

    def test_small_rnumel_stays_single_tile(self):
        """rnumel=1024 is below min_numel — should use standard persistent, no tiling."""

        def fn(x):
            return x / x.sum(dim=-1, keepdim=True)

        x = torch.randn(16, 1024, device=GPU_TYPE)
        compiled = torch.compile(fn)
        with inductor_config.patch(multi_tile_config):
            _, code = run_and_get_code(compiled, x)

        # Should be persistent but without multi-tile offsets
        self.assertIn("persistent_reduction", code[0])
        self.assertNotIn("r0_offset = 4096", code[0])
        self.assertNotIn("persistent_reduction_num_tiles", code[0])

    def test_non_power_of_two_falls_back(self):
        """rnumel=6144 is not power-of-two — should not use multi-tile."""

        def fn(x):
            return x / x.sum(dim=-1, keepdim=True)

        x = torch.randn(16, 6144, device=GPU_TYPE)
        compiled = torch.compile(fn)
        with inductor_config.patch(multi_tile_config):
            _, code = run_and_get_code(compiled, x)

        # Should not have tile offsets
        self.assertNotIn("r0_offset = 4096", code[0])

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
        # No config patch — feature is off
        _, code = run_and_get_code(compiled, x)
        self.assertNotIn("r0_offset = 4096", code[0])
        self.assertNotIn("persistent_reduction_num_tiles", code[0])


if __name__ == "__main__":
    run_tests()
