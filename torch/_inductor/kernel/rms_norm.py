# mypy: allow-untyped-defs
import functools
import logging

import sympy

import torch
from torch._inductor.kernel.mm_common import load_kernel_template

from .. import ir, lowering as L
from ..lowering import fallback_handler, register_lowering
from ..select_algorithm import (
    autotune_select_algorithm,
    GluonTritonTemplate,
    SymbolicGridFn,
)
from ..utils import use_triton_template
from ..virtualized import V


log = logging.getLogger(__name__)
aten = torch.ops.aten


@functools.cache
def has_gluon_tma() -> bool:
    try:
        from triton.experimental import gluon  # noqa: F401
        from triton.experimental.gluon import language as ttgl  # noqa: F401
        from triton.experimental.gluon.nvidia.hopper import TensorDescriptor  # noqa: F401
    except ImportError:
        return False
    return True


GLUON_RMS_NORM_LAYOUT = (
    "NVMMASharedLayout(swizzle_byte_width=0, element_bitwidth=16, "
    "rank=2, transposed=False, fp4_padded=False, cga_layout=[])"
)


@SymbolicGridFn
def rms_norm_grid(m, n, meta):
    return (m, 1, 1)


gluon_rms_norm_template = GluonTritonTemplate(
    name="gluon_rms_norm",
    grid=rms_norm_grid,
    source=load_kernel_template("gluon_rms_norm"),
    cache_codegen_enabled_for_template=True,
)


def _valid_r_blocks(n: int) -> list[int]:
    return [rb for rb in (256, 512, 1024, 2048, 4096) if rb <= n and n % rb == 0]


def _static_int(expr) -> int | None:
    if isinstance(expr, int):
        return expr
    if isinstance(expr, sympy.Integer):
        return int(expr)
    return None


def _is_sm90_or_later(device: torch.device) -> bool:
    if not torch.cuda.is_available():
        return False
    try:
        major, _ = torch.cuda.get_device_capability(device)
    except (AssertionError, RuntimeError):
        return False
    return major >= 9


def _normalized_shape_matches_last_dim(x, normalized_shape) -> bool:
    if len(normalized_shape) != 1:
        return False
    norm_n = _static_int(normalized_shape[0])
    x_n = _static_int(x.get_size()[-1])
    return norm_n is not None and x_n is not None and norm_n == x_n


def can_use_gluon_rms_norm(x, weight, normalized_shape, eps) -> bool:
    if V.graph.cpp_wrapper:
        return False
    if not has_gluon_tma():
        return False
    if weight is None or eps is None or eps != 1e-6:
        return False
    if x.get_device() is None or x.get_device().type != "cuda":
        return False
    if not _is_sm90_or_later(x.get_device()):
        return False
    if x.get_dtype() is not torch.bfloat16 or weight.get_dtype() not in (
        torch.bfloat16,
        torch.float32,
    ):
        return False
    if len(x.get_size()) != 2 or len(weight.get_size()) != 1:
        return False
    if not _normalized_shape_matches_last_dim(x, normalized_shape):
        return False
    n = _static_int(x.get_size()[1])
    if n is None or not _valid_r_blocks(n):
        return False
    layout = ir.FixedLayout(x.get_device(), x.get_dtype(), x.get_size())
    return use_triton_template(layout, check_max_autotune=False)


def _get_autotune_configs(n: int):
    """Generate (r_block, num_warps) autotune configs for Gluon RMSNorm."""
    configs = []
    for r_block in _valid_r_blocks(n):
        for num_warps in (4, 8, 16):
            if r_block < num_warps * 32:
                continue
            configs.append((r_block, num_warps))
    return configs


def _best_config(n: int) -> tuple[int, int]:
    """Pick best (r_block, num_warps) heuristically based on N.

    Empirically from standalone benchmark on GB200:
    - N <= 4096: r_block=1024, warps=4
    - N = 8192: r_block=1024, warps=8
    - N >= 16384: r_block=4096, warps=16 (or r_block=2048, warps=16)
    """
    valid = _valid_r_blocks(n)
    if n >= 16384:
        for rb in (4096, 2048):
            if rb in valid:
                return (rb, 16)
    if n >= 8192:
        for rb in (1024, 2048):
            if rb in valid:
                return (rb, 8)
    for rb in (1024, 512, 256):
        if rb in valid:
            return (rb, 4)
    return (valid[0], 4)


def tuned_rms_norm(x, weight, *, eps: float = 1e-6, layout=None):
    x = ir.ExternKernel.require_contiguous(x)
    weight = ir.ExternKernel.require_contiguous(weight)

    if layout is None:
        layout = ir.FixedLayout(x.get_device(), x.get_dtype(), x.get_size())

    n = _static_int(x.get_size()[1])
    if n is None:
        raise NotImplementedError("Gluon RMSNorm requires static N")

    x_desc = ir.GluonTensorDescriptor(
        x,
        [1, n],
        GLUON_RMS_NORM_LAYOUT,
    )

    configs = _get_autotune_configs(n)
    if not configs:
        raise NotImplementedError("no valid Gluon RMSNorm TMA-smem configs")

    choices: list[ir.ChoiceCaller] = []
    for r_block, num_warps in configs:
        gluon_rms_norm_template.maybe_append_choice(
            choices,
            input_nodes=(x_desc, weight),
            layout=layout,
            num_stages=3,
            num_warps=num_warps,
            prefix_args=1,
            BLOCK_N=n,
            R_BLOCK=r_block,
            EPS=eps,
            ACC_TYPE="tl.float32",
        )

    if not choices:
        raise NotImplementedError("no valid Gluon RMSNorm TMA-smem configs")

    result, _ = autotune_select_algorithm(
        "gluon_rms_norm",
        choices,
        [x_desc, weight],
        layout,
    )
    return result


_fallback_fused_rms_norm = fallback_handler(
    aten._fused_rms_norm.default, add_to_fallback_set=False
)


def _compute_rstd(x, normalized_shape, eps: float):
    x_fp32 = L.to_dtype(x, torch.float32)
    squared = L.mul(x_fp32, x_fp32)
    summed = L.sum_(squared, axis=-1, keepdims=True, dtype=torch.float32)
    n = _static_int(normalized_shape[0])
    assert n is not None
    mean = L.div(summed, n)
    shifted = L.add(mean, eps)
    return L.rsqrt(shifted)


@register_lowering(aten._fused_rms_norm.default, type_promotion_kind=None)
def fused_rms_norm(input, normalized_shape, weight, eps):
    if not can_use_gluon_rms_norm(input, weight, normalized_shape, eps):
        return _fallback_fused_rms_norm(input, normalized_shape, weight, eps)

    rstd = _compute_rstd(input, normalized_shape, eps)
    out = tuned_rms_norm(input, weight, eps=eps)
    return out, rstd
