"""Tests for GluonTMAKernel codegen (torch/_inductor/codegen/gluon_kernel.py).

Requires SM>=90 (Hopper+) and triton.experimental.gluon availability.
"""
import unittest

import torch
import torch._dynamo
import torch._inductor.config as inductor_config
from torch.testing._internal.common_utils import (
    TestCase,
)
from torch.testing._internal.inductor_utils import HAS_GPU, GPU_TYPE


def _has_gluon():
    try:
        from triton.experimental import gluon  # noqa: F401
        return True
    except ImportError:
        return False


def _is_sm90():
    if not HAS_GPU:
        return False
    props = torch.cuda.get_device_properties(0)
    return props.major >= 9


requires_gluon_tma = unittest.skipUnless(
    HAS_GPU and _has_gluon() and _is_sm90(),
    "Requires CUDA SM>=90 and triton.experimental.gluon",
)


def rmsnorm(x, weight, eps=1e-6):
    variance = x.pow(2).mean(-1, keepdim=True)
    x_normed = x * torch.rsqrt(variance + eps)
    return x_normed * weight


def mixed_reused_input(x, y):
    scale = (x.float() * y).mean(-1, keepdim=True)
    return scale * y


@requires_gluon_tma
class TestGluonTMAKernel(TestCase):
    def setUp(self):
        super().setUp()
        torch._dynamo.reset()
        inductor_config.triton.gluon_tma_reductions = True
        inductor_config.force_disable_caches = True

    def tearDown(self):
        inductor_config.triton.gluon_tma_reductions = False
        inductor_config.force_disable_caches = False
        super().tearDown()

    def _check_rmsnorm(self, *shape, rtol=0.05, atol=0.05):
        x = torch.randn(*shape, device="cuda", dtype=torch.bfloat16)
        w = torch.randn(shape[-1], device="cuda", dtype=torch.bfloat16)
        compiled = torch.compile(rmsnorm)
        result = compiled(x, w)
        expected = rmsnorm(x, w)
        torch.testing.assert_close(result, expected, rtol=rtol, atol=atol)

    def test_rmsnorm_4096(self):
        self._check_rmsnorm(8, 4096)

    def test_rmsnorm_8192(self):
        self._check_rmsnorm(32, 8192)

    def test_rmsnorm_16384(self):
        self._check_rmsnorm(32, 16384)

    def test_rmsnorm_large_batch(self):
        self._check_rmsnorm(8192, 4096)

    def test_rmsnorm_3d(self):
        self._check_rmsnorm(2, 3, 4096)

    def test_fallback_fp32(self):
        """fp32 input should fall back to normal TritonKernel (not Gluon)."""
        x = torch.randn(8, 4096, device="cuda", dtype=torch.float32)
        w = torch.randn(4096, device="cuda", dtype=torch.float32)
        compiled = torch.compile(rmsnorm)
        result = compiled(x, w)
        expected = rmsnorm(x, w)
        torch.testing.assert_close(result, expected, rtol=1e-4, atol=1e-4)

    def test_fallback_small_n(self):
        """N < 4096 should fall back to normal TritonKernel."""
        x = torch.randn(8, 2048, device="cuda", dtype=torch.bfloat16)
        w = torch.randn(2048, device="cuda", dtype=torch.bfloat16)
        compiled = torch.compile(rmsnorm)
        result = compiled(x, w)
        expected = rmsnorm(x, w)
        torch.testing.assert_close(result, expected, rtol=0.01, atol=0.01)

    def test_fallback_non_power_of_two_n(self):
        x = torch.randn(8, 6144, device="cuda", dtype=torch.bfloat16)
        w = torch.randn(6144, device="cuda", dtype=torch.bfloat16)
        compiled = torch.compile(rmsnorm)
        result = compiled(x, w)
        expected = rmsnorm(x, w)
        torch.testing.assert_close(result, expected, rtol=0.01, atol=0.01)

    def test_fallback_reused_input_not_bf16(self):
        x = torch.randn(8, 4096, device="cuda", dtype=torch.bfloat16)
        y = torch.randn(8, 4096, device="cuda", dtype=torch.float32)
        compiled = torch.compile(mixed_reused_input)
        result = compiled(x, y)
        expected = mixed_reused_input(x, y)
        torch.testing.assert_close(result, expected, rtol=1e-3, atol=1e-3)

    def test_gluon_kernel_is_used(self):
        """Verify the GluonTMAKernel is actually dispatched."""
        from torch._inductor.codegen.gluon_kernel import GluonTMAKernel
        from torch._inductor.codegen import triton as triton_codegen

        created_gluon = []
        original = triton_codegen.TritonScheduling.create_kernel_choices

        def patched(self, kernel_features, kernel_args, kernel_kwargs):
            result = original(self, kernel_features, kernel_args, kernel_kwargs)
            for k in result:
                if isinstance(k, GluonTMAKernel):
                    created_gluon.append(True)
            return result

        triton_codegen.TritonScheduling.create_kernel_choices = patched
        try:
            x = torch.randn(8, 4096, device="cuda", dtype=torch.bfloat16)
            w = torch.randn(4096, device="cuda", dtype=torch.bfloat16)
            compiled = torch.compile(rmsnorm)
            compiled(x, w)
            self.assertTrue(len(created_gluon) > 0, "GluonTMAKernel was not created")
        finally:
            triton_codegen.TritonScheduling.create_kernel_choices = original


if __name__ == "__main__":
    unittest.main()
