#pragma once

#include <ATen/core/Tensor.h>

namespace at::native {

void tma_scatter_add_dispatch(
    const Tensor& self,
    int64_t dim,
    const Tensor& index,
    const Tensor& src,
    int rows_per_block);

} // namespace at::native
