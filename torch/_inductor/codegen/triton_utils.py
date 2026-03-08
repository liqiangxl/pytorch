# mypy: allow-untyped-defs
import dataclasses
import enum
from typing import Any

import sympy

import torch
from torch.utils._sympy.symbol import symbol_is_type, SymT

from .. import config
from ..runtime.hints import AttrsDescriptorWrapper
from ..utils import _type_of, expr_fits_within_32bit, triton_version_uses_attrs_dict
from ..virtualized import V
from .common import (
    ArgName,
    ConstexprArg,
    KernelArgType,
    SizeArg,
    TensorArg,
    TMADescriptorArg,
    WorkspaceArg,
)


class HintCheckType(enum.Enum):
    """Type of runtime check for a speculative hint."""

    DIVISIBLE = "divisible"  # arg % value == 0


@dataclasses.dataclass(frozen=True)
class SpeculativeHint:
    """A speculative divisibility annotation for a kernel SizeArg.

    When dynamic=True, SizeArgs may not be statically provable as divisible by 16,
    preventing Triton from emitting vectorized loads (LDG.E.128). A SpeculativeHint
    records that a SizeArg *could* be divisible, enabling AOT compilation of a fast
    variant with the divisibility annotation applied.

    Fields:
        arg_index:  Position in the kernel's flattened signature (used by apply_hints
                    to add to the divisible_by_16 tuple for Triton config).
        arg_name:   SizeArg.name (e.g. "ks0", "xnumel") — for debugging only.
        check_type: What runtime check to emit (currently only DIVISIBLE).
        check_value: The divisor (16 for Triton's tt.divisibility=16).
        sympy_expr: The symbolic expression from SizeArg.expr — used to deduplicate
                    runtime conditions when multiple SizeArgs share the same symbol
                    (e.g. ks0 and r0_numel both mapping to s1).
    """

    arg_index: int
    arg_name: str
    check_type: HintCheckType
    check_value: int
    sympy_expr: sympy.Expr


def should_unwrap_unspec_arg(name: str):
    if V.graph.is_unspec_arg(name):
        # Unwrap on all devices except CPU
        if V.graph.get_current_device_or_throw().type != "cpu":
            return True
        # Only unwrap on CPU if the input is not used as an output
        if name not in V.graph.mutated_buffers:
            return True
    return False


def signature_of(arg: KernelArgType, *, size_dtype: str | None) -> str:
    if isinstance(arg, TensorArg):
        # TODO: Remove fp8 special handling when Triton supports PyTorch fp8 dtypes.
        # Related PR: https://github.com/triton-lang/triton/pull/2279/
        if arg.dtype == torch.float8_e4m3fn:
            typ = "*fp8e4nv"
        elif arg.dtype == torch.float8_e5m2:
            typ = "*fp8e5"
        elif arg.dtype == torch.float8_e4m3fnuz:
            typ = "*fp8e4b8"
        elif arg.dtype == torch.float8_e5m2fnuz:
            typ = "*fp8e5b16"
        else:
            typ = _type_of(arg.dtype)
        if should_unwrap_unspec_arg(arg.buffer):
            # had unwrapped 0d tensor as scalar
            new_typ = typ.lstrip("*")
            if new_typ in ["fp16", "bf16"]:
                return "fp32"
            else:
                return new_typ
        else:
            return typ
    if isinstance(arg, SizeArg):
        if arg.expr is None:
            if triton_version_uses_attrs_dict():
                # In newer versions of Triton, the signature includes "None" args
                # and their type is marked as "constexpr"
                return "constexpr"
            else:
                # In older versions of Triton...
                # From triton/runtime/jit.py
                # `None` is nullptr.  Implicitly convert to *i8.
                return "*i8"
        elif _arg_equals_1(arg) and triton_version_uses_attrs_dict():
            # In new versions of Triton, if we have an equal-to-1 arg that's marked as a constant,
            # it should be marked as "constexpr" in the signature.
            return "constexpr"
        elif isinstance(arg.expr, (float, sympy.Float)):
            # Python floats are natively fp64, so use fp64 to preserve precision
            return "fp64" if config._use_fp64_for_unbacked_floats else "fp32"
        elif isinstance(arg.expr, sympy.Symbol) and symbol_is_type(
            arg.expr, (SymT.UNBACKED_FLOAT)
        ):
            # Unbacked floats from .item() should preserve fp64 precision
            return "fp64" if config._use_fp64_for_unbacked_floats else "fp32"
        elif isinstance(arg.expr, bool):
            return "i1"

        # if this is a integer
        if size_dtype == "tl.int32":
            return "i32"
        elif size_dtype == "tl.int64":
            return "i64"
        elif size_dtype is None:
            # no hint: we'll see if we know that this is a 32-bit int, and guard if possible.
            int_max = torch.iinfo(torch.int32).max
            if expr_fits_within_32bit(arg.expr):
                V.graph.sizevars.check_leq(arg.expr, int_max)
                return "i32"
            else:
                return "i64"
        else:
            raise NotImplementedError(f"unhandled size_dtype {size_dtype}")
    if isinstance(arg, WorkspaceArg):
        return _type_of(arg.dtype)
    if isinstance(arg, TMADescriptorArg):
        if arg.api_type == "experimental":
            return "nvTmaDesc"
        else:
            # https://github.com/triton-lang/triton/blob/9695baed9b46cf957e08b157bb4133f4a4b331c5/python/triton/runtime/jit.py#L360-L363
            assert arg.api_type == "stable"
            assert arg.block_shape is not None
            assert arg.dtype is not None
            inner = _type_of(arg.dtype)[1:]  # strip the `*`: *fp32 -> fp32
            return f"tensordesc<{inner}{list(arg.block_shape)}>"
    if isinstance(arg, ConstexprArg):
        return "constexpr"
    raise NotImplementedError(f"unhandled {type(arg)}: {arg}")


def non_constexpr_signature(signature):
    new_signature = []
    for arg in signature:
        if not isinstance(arg, ConstexprArg):
            new_signature.append(arg)

    return new_signature


def signature_to_meta(
    signature: list[KernelArgType],
    *,
    size_dtype: str | None,
    argdefs: list[ArgName],
    indices: list[int] | None = None,
    is_template: bool = False,
) -> dict[str, str]:
    if indices is None:
        indices = list(range(len(signature)))

    def _decide_tl_dtype(arg):
        # Even if the ks0 symbol itself is within tl.int32 range, it's
        # risky to use tl.int32 dtype since we may have ks0*ks1 later
        # for kernels like torch.mean when dynamic shape is enabled.
        #
        # Check config.triton.use_block_ptr, since Triton block pointer
        # does not support 64bit indexing:
        # https://gist.github.com/shunting314/6a41c776171720ce4561f202dcde0ad6
        #
        # If the triton metadata is for a template, don't use tl.int64 index.
        # Templates like flex attention/decoding uses block pointers which
        # does not support 64 bit indexing.
        if (
            not config.triton.use_block_ptr
            and not is_template
            and isinstance(arg, SizeArg)
            and arg.name.startswith("ks")
        ):
            return "tl.int64"
        return size_dtype

    return {
        argdefs[i].name: signature_of(arg, size_dtype=_decide_tl_dtype(arg))
        for i, arg in zip(indices, signature)
    }


def is_unaligned_buffer(arg: TensorArg):
    buf_name = arg.buffer
    if buf_name in V.graph.unaligned_buffers:
        return True

    if buf_name in V.graph.graph_inputs:
        # See Note: [Input Alignment handling in Inductor]
        # For graph inputs that is not recorded in V.graph.unaligned_buffers,
        # we know for sure the tensor is aligned.
        return False

    if buf_name in V.graph.constants:
        # all constants are assumed to be aligned
        return False

    if V.graph.scheduler:
        layout = V.graph.scheduler.get_buffer_layout(buf_name)
    else:
        buffer = V.graph.try_get_buffer(buf_name)
        # output arg
        if not buffer:
            assert buf_name == V.kernel.output_node.name
            layout = V.kernel.output_node.layout
        else:
            layout = buffer.get_layout()

    if isinstance(layout, torch._inductor.ir.NonOwningLayout):
        return not layout.maybe_guard_aligned()
    else:
        return False


def _arg_equals_1(arg: KernelArgType) -> bool:
    return (
        isinstance(arg, SizeArg)
        and isinstance(arg.expr, (int, sympy.Integer))
        and V.graph.sizevars.statically_known_equals(arg.expr, 1)  # type: ignore[arg-type]
    )


def equal_1_arg_indices(
    args: list[KernelArgType],
    *,
    indices: list[int] | None = None,
) -> tuple[int, ...]:
    if indices is None:
        indices = list(range(len(args)))

    equal_to_1 = tuple(i for i, arg in zip(indices, args) if _arg_equals_1(arg))

    return equal_to_1


def _is_aligned(x: KernelArgType, alignment: int, include_tensor: bool) -> bool:
    """Check if a kernel arg is statically provable as aligned to `alignment` bytes.

    For TensorArgs: checks pointer offset alignment and buffer alignment.
    For SizeArgs: checks if the symbolic expression is a known multiple of `alignment`.
    For WorkspaceArgs: always aligned (we control the allocation).

    Mirrors Triton's alignment logic:
    https://github.com/triton-lang/triton/blob/5282ed890d453e10b9ee30076ef89115dd197761/python/triton/runtime/jit.py#L208-L222
    """
    if isinstance(x, TensorArg):
        if include_tensor:
            offset_aligned = V.graph.sizevars.statically_known_multiple_of(
                x.offset * x.dtype.itemsize,
                alignment,  # type: ignore[arg-type]
            )
            return offset_aligned and not is_unaligned_buffer(x)
        else:
            return False
    if isinstance(x, SizeArg):
        # TODO(voz): These are kinda redundant, if we can solve out statically_known_multiple_of with
        # _maybe_evaluate_static...
        if x.name.startswith("load_seed_offset"):
            return False
        if x.expr is None:
            return False
        if isinstance(x.expr, float):
            return False
        return V.graph.sizevars.statically_known_multiple_of(x.expr, alignment)  # type: ignore[arg-type]
    if isinstance(x, WorkspaceArg):
        # We allocate the workspace ourselves, so it is always aligned
        return True
    if isinstance(x, (TMADescriptorArg, ConstexprArg)):
        return False
    raise NotImplementedError(f"unhandled {type(x)}: {x}")


def divisible_by_16_indices(
    args: list[KernelArgType],
    *,
    indices: list[int] | None = None,
) -> tuple[int, ...]:
    """Compute which arg indices are statically provable as divisible by 16."""
    if not config.triton.divisible_by_16:
        return ()
    if indices is None:
        indices = list(range(len(args)))
    return tuple(
        i
        for i, arg in zip(indices, args)
        if _is_aligned(arg, alignment=16, include_tensor=True)
    )


def config_of(
    args: list[KernelArgType],
    *,
    indices: list[int] | None = None,
) -> Any:
    if indices is None:
        indices = list(range(len(args)))

    divisible_by_16 = divisible_by_16_indices(args, indices=indices)
    equal_to_1 = equal_1_arg_indices(args, indices=indices)

    # pyrefly: ignore [bad-argument-type]
    return AttrsDescriptorWrapper(divisible_by_16, equal_to_1)


def _get_divisible_by_16(attrs_config: Any) -> set[int]:
    """Extract divisible-by-16 arg indices from an AttrsDescriptorWrapper.

    AttrsDescriptorWrapper has different formats across Triton versions:
      - V0 (no triton) / V1 (triton.compiler): namedtuple with .divisible_by_16
      - V2/V3 (triton.backends): AttrsDescriptor with .divisibility_16
      - V4 (2025 dict): {(idx,): [["tt.divisibility", 16]], ...}
    """
    if hasattr(attrs_config, "divisible_by_16"):
        return set(attrs_config.divisible_by_16)
    if hasattr(attrs_config, "divisibility_16"):
        return set(attrs_config.divisibility_16)
    if isinstance(attrs_config, dict):
        return {k[0] for k in attrs_config if isinstance(k, tuple)}
    return set()


def speculative_hints(
    args: list[KernelArgType],
    base_config: Any,
    *,
    indices: list[int] | None = None,
) -> list[SpeculativeHint]:
    """Find SizeArgs that could benefit from speculative divisibility-by-16 annotation.

    Returns a SpeculativeHint for each symbolic SizeArg whose divisibility by 16
    is not statically provable. These hints are used to AOT-compile a fast kernel
    variant (with tt.divisibility=16 applied) alongside a general fallback, with
    runtime if/else dispatch based on the actual values.

    This is a companion to config_of() — config_of() computes what is statically
    proven, speculative_hints() identifies what could additionally be true at runtime.
    """
    if not config.triton.speculative_divisibility:
        return []

    if indices is None:
        indices = list(range(len(args)))

    # Indices already statically proven divisible by 16 — no need to speculate
    proven_div16 = _get_divisible_by_16(base_config)
    hints: list[SpeculativeHint] = []

    for i, arg in zip(indices, args):
        if i in proven_div16:
            continue
        if not isinstance(arg, SizeArg):
            continue
        if arg.expr is None or isinstance(arg.expr, float):
            continue
        if arg.name.startswith("load_seed_offset"):
            continue

        # Any symbolic SizeArg not statically proven div16 gets a speculative hint.
        # We don't check the concrete size_hint value — the runtime if/else handles
        # both div16 and non-div16 shapes regardless of which is seen first.
        hints.append(
            SpeculativeHint(
                arg_index=i,
                arg_name=arg.name,
                check_type=HintCheckType.DIVISIBLE,
                check_value=16,
                sympy_expr=arg.expr,
            )
        )

    return hints


def apply_hints(
    args: list[KernelArgType],
    hints: list[SpeculativeHint],
    *,
    indices: list[int] | None = None,
) -> Any:
    """Build an AttrsDescriptorWrapper with speculative divisibility hints applied.

    Merges the statically-proven divisible_by_16 indices (from _is_aligned) with
    the speculative hint indices to produce a config where all hinted args are
    annotated as divisible by 16. Used for the fast kernel variant.
    """
    base_div16 = divisible_by_16_indices(args, indices=indices)
    equal_to_1 = equal_1_arg_indices(args, indices=indices)
    speculative_div16 = tuple(
        h.arg_index for h in hints if h.check_type == HintCheckType.DIVISIBLE
    )
    merged_div16 = tuple(sorted(set(base_div16) | set(speculative_div16)))
    return AttrsDescriptorWrapper(merged_div16, equal_to_1)
