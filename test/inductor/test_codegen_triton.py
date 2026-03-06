# Owner(s): ["module: inductor"]
import contextlib

import sympy

import torch
import torch._inductor.config as inductor_config
from torch._inductor.codegen import triton_utils
from torch._inductor.codegen.common import SizeArg
from torch._inductor.graph import GraphLowering
from torch._inductor.test_case import TestCase as InductorTestCase
from torch._inductor.virtualized import V
from torch.testing._internal.inductor_utils import HAS_CPU, HAS_GPU


class TestCodegenTriton(InductorTestCase):
    def setUp(self):
        super().setUp()

        class DummyModule(torch.nn.Module):
            def forward(self, x):
                return x * 2

        self._gm = torch.fx.symbolic_trace(DummyModule())
        self._graph = GraphLowering(self._gm)

        self._stack = contextlib.ExitStack()
        self._stack.enter_context(V.set_graph_handler(self._graph))

    def tearDown(self):
        self._stack.close()
        super().tearDown()

    @inductor_config.patch("triton.divisible_by_16", True)
    def test_config_of_sizearg(self):
        from torch._inductor.utils import (
            get_triton_attrs_descriptor_version,
            TritonAttrsDescriptorVersion,
        )

        two = sympy.Integer(2)
        eight = sympy.Integer(8)
        sixteen = sympy.Integer(16)
        s0 = sympy.Symbol("s0", positive=True, integer=True)
        s1 = sympy.Symbol("s1", positive=True, integer=True)

        def _check_divisibility(expected_divisible_indices, config):
            if get_triton_attrs_descriptor_version() in {
                TritonAttrsDescriptorVersion.V1_COMPILER,
                TritonAttrsDescriptorVersion.V0_NO_TRITON,
            }:
                self.assertEqual(expected_divisible_indices, config.divisible_by_16)
            elif get_triton_attrs_descriptor_version() in {
                TritonAttrsDescriptorVersion.V2_BACKENDS,
                TritonAttrsDescriptorVersion.V3_BACKENDS_TUPLE,
            }:
                self.assertEqual(expected_divisible_indices, config.divisibility_16)
            else:
                if (
                    get_triton_attrs_descriptor_version()
                    != TritonAttrsDescriptorVersion.V4_DICT
                ):
                    raise AssertionError
                self.assertIsInstance(config, dict)

                for idx in expected_divisible_indices:
                    # config is in the form
                    # {(idx,): [["tt.divisibility", 16]]}
                    # where (idx,) is a tuple in order to support tuple inputs to triton kernels.
                    self.assertTrue((idx,) in config)
                    self.assertTrue(["tt.divisibility", 16] in config[(idx,)])

        _check_divisibility(
            (2,),
            triton_utils.config_of(
                [
                    SizeArg("A", two),  # no
                    SizeArg("B", eight),  # no
                    SizeArg("C", sixteen),  # yes
                    SizeArg("D", s0),  # no
                    SizeArg("E", s1),  # no
                ]
            ),
        )

        _check_divisibility(
            (0, 2, 4, 5, 6),
            triton_utils.config_of(
                [
                    SizeArg("A", two * eight),  # 0: yes
                    SizeArg("B", eight * s0),  # 1: no
                    SizeArg("C", two * eight * s0),  # 2: yes
                    SizeArg("D", s0 * s1),  # 3: no
                    SizeArg("E", sixteen * s0),  # 4: yes
                    SizeArg("F", sixteen * eight * s0 * s1),  # 5: yes
                    SizeArg("G", two * eight * s0 * s1),  # 6: yes
                ]
            ),
        )

    @inductor_config.patch("triton.divisible_by_16", True)
    def test_config_of_dynamic_shapes_hint_divisibility(self):
        """Guard-based tt.divisibility=16 for backed symbolic SizeArgs.

        When dynamic=True, numel args are symbolic but have concrete backing
        values. If the backing value is divisible by 16, config_of() should
        install a guard and emit the hint, recovering vectorization.
        """
        from torch._dynamo.source import ConstantSource
        from torch.fx.experimental.symbolic_shapes import ShapeEnv

        shape_env = ShapeEnv()
        s0 = shape_env.create_symbol(512, source=ConstantSource("s0"))
        s1 = shape_env.create_symbol(2048, source=ConstantSource("s1"))
        s2 = shape_env.create_symbol(13, source=ConstantSource("s2"))

        gm = torch.fx.symbolic_trace(
            type("M", (torch.nn.Module,), {"forward": lambda self, x: x * 2})()
        )
        graph = GraphLowering(gm, shape_env=shape_env)

        with V.set_graph_handler(graph):
            cfg = triton_utils.config_of(
                [
                    SizeArg("xnumel", s0 * s1),  # 0: 1048576, div by 16
                    SizeArg("r0_numel", s0),  # 1: 512, div by 16
                    SizeArg("r1_numel", s2),  # 2: 13, NOT div by 16
                    SizeArg("ks0", s0 * s2),  # 3: 6656, div by 16
                ]
            )

        # cfg is a dict like {(idx,): [["tt.divisibility", 16]], ...}
        if not isinstance(cfg, dict):
            self.skipTest("Test only applies to dict-based Triton attrs API")
        for idx in (0, 1, 3):
            self.assertIn(["tt.divisibility", 16], cfg[(idx,)])
        self.assertNotIn((2,), cfg)


if __name__ == "__main__":
    from torch._inductor.test_case import run_tests

    if HAS_CPU or HAS_GPU:
        run_tests("sympy")
