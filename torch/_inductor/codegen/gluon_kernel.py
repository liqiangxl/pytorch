"""
GluonTMAKernel keeps TritonKernel's generated body structure and routes one
selected input through a single Gluon TMA row load.

The TMA buffer is held in shared memory using the input dtype. Each reduction
slice is gathered from shared memory and converted to fp32 only for compute,
which keeps the cached row compact and avoids promoting the whole row into
registers.
"""

from __future__ import annotations

from functools import lru_cache
from textwrap import dedent
from typing import TYPE_CHECKING

import sympy

import torch
from torch._inductor.codegen.simd import IterationRangesRoot
from torch._inductor.codegen.triton import (
    FixedTritonConfig,
    TritonCSEVariable,
    TritonKernel,
    TritonKernelOverrides,
    TritonSymbols,
)
from torch._inductor.utils import IndentedBuffer, upcast_compute_type
from torch._inductor.virtualized import V


if TYPE_CHECKING:
    from torch._inductor.ir import IRNode


_GLUON_ROW_LAYOUT = (
    "NVMMASharedLayout(swizzle_byte_width=0, element_bitwidth=16, "
    "rank=2, transposed=False, fp4_padded=False, cga_layout=[])"
)

_GLUON_TYPE_MAPPING = {
    torch.float64: "_ttgl.float64",
    torch.float32: "_ttgl.float32",
    torch.float16: "_ttgl.float16",
    torch.bfloat16: "_ttgl.bfloat16",
    torch.int64: "_ttgl.int64",
    torch.int32: "_ttgl.int32",
    torch.int16: "_ttgl.int16",
    torch.int8: "_ttgl.int8",
    torch.uint64: "_ttgl.uint64",
    torch.uint32: "_ttgl.uint32",
    torch.uint16: "_ttgl.uint16",
    torch.uint8: "_ttgl.uint8",
    torch.bool: "_ttgl.int1",
}


def gluon_type(dtype: torch.dtype) -> str:
    return _GLUON_TYPE_MAPPING[dtype]


class GluonKernelOverrides(TritonKernelOverrides):
    @staticmethod
    def to_dtype(
        x,
        dtype: torch.dtype,
        src_dtype: torch.dtype | None = None,
        use_compute_types=True,
    ):
        if dtype == torch.bool:
            return f"({x} != 0)"
        out_dtype = upcast_compute_type(dtype) if use_compute_types else dtype
        return f"{x}.to({V.kernel.codegen_dtype(out_dtype)})"

    @staticmethod
    def _shaped_constant(value, dtype, shape):
        type_ = torch._prims_common.dtype_to_type(dtype)
        triton_val = repr(type_(value))
        return V.kernel.codegen_full(str(shape), triton_val, upcast_compute_type(dtype))

    @staticmethod
    def where(a, b, c):
        return f"_ttgl.where({a}, {b}, {c})"


class GluonTMAKernel(TritonKernel):
    """
    TritonKernel variant with one selected input cached in shared memory by TMA.
    """

    overrides = GluonKernelOverrides  # type: ignore[assignment]

    def __init__(
        self,
        *args,
        tma_buffer_name: str,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.tma_buffer_name = tma_buffer_name
        self._gluon_tma_desc_arg: str | None = None
        self._gluon_tma_load_seen = False
        self._gluon_tma_preamble_emitted = False

        r_numel = int(self.features.reduction_numel)
        r_block, num_warps = self._select_tiling(r_numel)
        assert r_numel % r_block == 0
        self._gluon_inner = r_block
        self._gluon_num_warps = num_warps
        self.fixed_config = FixedTritonConfig(
            {
                "XBLOCK": 1,
                "R0_BLOCK": self._gluon_inner,
                "num_warps": self._gluon_num_warps,
                "num_stages": 1,
            }
        )

    @staticmethod
    def _is_power_of_two(value: int) -> bool:
        return value > 0 and (value & (value - 1)) == 0

    @classmethod
    def _select_tiling(cls, r_numel: int) -> tuple[int, int]:
        if not cls._is_power_of_two(r_numel):
            return 1024, 8
        if r_numel >= 16384:
            return 2048, 16
        if r_numel >= 8192:
            return 1024, 8
        return 512, 4

    def _jit_decorator(self) -> str:
        return "@_gluon.jit"

    @property
    def _constexpr_type(self) -> str:
        return "_ttgl.constexpr"

    @classmethod
    @lru_cache(None)
    def gen_common_triton_imports(cls) -> str:
        base = TritonKernel.gen_common_triton_imports()
        return base + dedent("""
            from triton.experimental import gluon as _gluon
            from triton.experimental.gluon import language as _ttgl
            from triton.experimental.gluon.language.nvidia.hopper import mbarrier as _mbarrier
            from triton.experimental.gluon.language.nvidia.hopper import tma as _tma
        """)

    def want_no_x_dim(self) -> bool:
        return True

    def codegen_dtype(self, dtype: torch.dtype) -> str:
        return gluon_type(dtype)

    def codegen_cast(self, value: str, dtype: torch.dtype) -> str:
        return f"{value}.to({self.codegen_dtype(dtype)})"

    def codegen_program_id(self, dim: int) -> str:
        return f"_ttgl.program_id({dim})"

    def codegen_num_programs(self, dim: int) -> str:
        return f"_ttgl.num_programs({dim})"

    def iteration_ranges_ranges_code(self, entry: IterationRangesRoot) -> str:
        assert entry.tensor_dim is not None
        size = self.indexing_size_str(entry.tensor_dim)
        index_dtype = self.get_index_dtype_as_torch_dtype()
        suffix = (
            f".to({self.codegen_dtype(index_dtype)})"
            if index_dtype != torch.int32
            else ""
        )
        return (
            f"_ttgl.arange(0, {self.kexpr(entry.block_size())}, _gluon_r_layout)"
            f"{size}{suffix}"
        )

    def iteration_ranges_scalar_code(
        self, entry: IterationRangesRoot, value
    ) -> str:
        if entry.prefix == "x":
            return value
        ndim = self.triton_tensor_ndim()
        size = [1] * ndim
        return self.codegen_full(
            str(size), str(value), self.get_index_dtype_as_torch_dtype()
        )

    def iteration_ranges_get_pid(self, entry: IterationRangesRoot) -> str:
        assert entry.grid_dim is not None
        key = self.codegen_program_id(entry.grid_dim)
        pid = entry.pid_cache.get(key, key)
        index_dtype = self.get_index_dtype_as_torch_dtype()
        if index_dtype != torch.int32:
            return self.codegen_cast(pid, index_dtype)
        return pid

    def reduction_resize(self, value) -> str:
        ndims = self.triton_tensor_ndim()
        if ndims == 1:
            return f"({value})"
        return super().reduction_resize(value)

    def reduction_resize_and_shape(self, value, shape):
        ndims = self.triton_tensor_ndim()
        if ndims == 1:
            return f"({value})", shape
        return super().reduction_resize_and_shape(value, shape)

    def codegen_full(self, size: str, value: str, dtype: torch.dtype) -> str:
        return f"_ttgl.full({size}, {value}, {self.codegen_dtype(dtype)}, _gluon_r_layout)"

    def codegen_zeros(self, size: str, dtype: torch.dtype) -> str:
        return f"_ttgl.zeros({size}, {self.codegen_dtype(dtype)}, _gluon_r_layout)"

    def create_constant_mask(self, entry: IterationRangesRoot) -> str:
        if entry.tensor_dim is None:
            mask = self.codegen_full(self.dense_size_str(), "True", torch.bool)
            return f"{entry.mask_name()} = {mask}"
        sizes = ["None"] * self.triton_tensor_ndim()
        sizes[entry.tensor_dim] = ":"
        suffix = ", ".join(sizes)
        mask = self.codegen_full(f"[{entry.block_size_str()}]", "True", torch.bool)
        return f"{entry.mask_name()} = {mask}[{suffix}]"

    def codegen_static_numels(self, code: IndentedBuffer) -> None:
        code.writeline(f"_gluon_nw: {self._constexpr_type} = _ttgl.num_warps()")
        code.splice(
            dedent(
                """
                _gluon_r_layout: _ttgl.constexpr = _ttgl.BlockedLayout(
                    size_per_thread=[R0_BLOCK // (_gluon_nw * 32)],
                    threads_per_warp=[32],
                    warps_per_cta=[_gluon_nw],
                    order=[0],
                )
                """
            )
        )
        super().codegen_static_numels(code)

    def load(self, name: str, index: sympy.Expr):
        if name != self.tma_buffer_name:
            return super().load(name, index)

        self.args.input(name)
        self._load_counts[name] += 1
        self._gluon_tma_load_seen = True
        dtype = torch.float32
        indexing = self.indexing(index, block_ptr=False)
        line = "_gluon_x_smem.gather(rindex, 0).to(_ttgl.float32)"
        shape = indexing.expand_shape or TritonSymbols.get_block_shape(indexing.index)
        load_buffer = self.get_load_buffer(indexing)
        result_var = self.cse.newvar(dtype=dtype, shape=shape)
        load_buffer.writeline(f"{result_var} = {line}")
        assert isinstance(result_var, TritonCSEVariable)
        result_var.mask_vars = indexing.mask_vars  # type: ignore[assignment]
        if not self.inside_reduction or not indexing.has_rmask():
            self.outside_loop_vars.add(result_var)
        return result_var

    def _gluon_tma_descriptor_arg(self) -> str:
        if self._gluon_tma_desc_arg is None:
            source_arg = self.args.input(self.tma_buffer_name)
            outer_name = f"gluon_tma_descriptor_{source_arg}"
            self._gluon_tma_desc_arg = self.args.tma_descriptor(
                outer_name=outer_name,
                source_name=self.tma_buffer_name,
                api_type="gluon",
                block_shape=[1, int(self.features.reduction_numel)],
                dtype=torch.bfloat16,
                layout=_GLUON_ROW_LAYOUT,
            )
        return self._gluon_tma_desc_arg

    def _gluon_tma_descriptor_call(self, source_name: str) -> str:
        r_numel = int(self.features.reduction_numel)
        row_stride = (
            f"({r_numel} * {source_name}.stride(-1) "
            f"if {source_name}.dim() == 1 else {source_name}.stride(-2))"
        )
        return (
            "triton.tools.tensor_descriptor.TensorDescriptor"
            f"({source_name}, [{source_name}.numel() // {r_numel}, {r_numel}], "
            f"[{row_stride}, {source_name}.stride(-1)], [1, {r_numel}])"
        )

    def _emit_host_tma_descriptors(self) -> None:
        wrapper = V.graph.wrapper_code
        for outer_name, arg in self.args.tma_descriptor_args.items():
            if arg.api_type != "gluon":
                continue
            assert arg.source_name is not None
            wrapper.writeline(
                f"{outer_name} = {self._gluon_tma_descriptor_call(arg.source_name)}"
            )

    def _tma_preamble_code(self) -> str:
        r_numel = int(self.features.reduction_numel)
        constexpr = self._constexpr_type
        tma_desc = self._gluon_tma_descriptor_arg()
        code = IndentedBuffer()
        code.writeline(f"_GLUON_N: {constexpr} = {r_numel}")
        code.writeline(
            f"_gluon_smem_layout: {constexpr} = "
            "_ttgl.NVMMASharedLayout(swizzle_byte_width=0, element_bitwidth=16)"
        )
        code.splice(dedent(f"""
            _gluon_bar = _ttgl.allocate_shared_memory(_ttgl.int64, [1], _mbarrier.MBarrierLayout())
            _gluon_smem_2d = _ttgl.allocate_shared_memory(_ttgl.bfloat16, [1, _GLUON_N], _gluon_smem_layout)
            _mbarrier.init(_gluon_bar, count=1)
            _mbarrier.expect(_gluon_bar, _GLUON_N * 2)
            _tma.async_copy_global_to_shared({tma_desc}, [xoffset, 0], _gluon_bar, _gluon_smem_2d)
            _mbarrier.wait(_gluon_bar, phase=0)
            _mbarrier.invalidate(_gluon_bar)
            _gluon_x_smem = _gluon_smem_2d.reshape([_GLUON_N])
        """))
        return code.getvalue()

    def _emit_tma_preamble_once(self) -> None:
        if self._gluon_tma_load_seen and not self._gluon_tma_preamble_emitted:
            self.body.splice(self._tma_preamble_code())
            self._gluon_tma_preamble_emitted = True

    def call_kernel(
        self, name: str, node: "IRNode | None" = None, deallocate_ws: bool = True
    ):
        self._gluon_tma_descriptor_arg()
        wrapper = V.graph.wrapper_code
        wrapper.write_triton_header_once()
        self._emit_host_tma_descriptors()
        return super().call_kernel(name, node=node, deallocate_ws=deallocate_ws)

    def codegen_body(self):
        self._emit_tma_preamble_once()
        return super().codegen_body()
