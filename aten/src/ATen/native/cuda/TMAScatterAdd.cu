#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/native/cuda/TMAScatterAdd.h>
#include <ATen/core/Tensor.h>
#include <ATen/Dispatch.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

#if !defined(USE_ROCM) && !defined(_WIN32) && defined(CUDA_VERSION) && CUDA_VERSION >= 12080

#include <cuda/ptx>
#include <cuda_bf16.h>
#include <cuda_fp16.h>

namespace at::native {

namespace ptx = ::cuda::ptx;

namespace {

template <typename T, typename index_t, int RowsPerBlock>
__global__ void __launch_bounds__(64)
tma_scatter_add_kernel(
    T* __restrict__ dst,
    const T* __restrict__ src,
    const index_t* __restrict__ index,
    int64_t N,
    int64_t D,
    int64_t index_size) {
#if __CUDA_ARCH__ >= 900
  extern __shared__ char smem_raw[];
  T* smem = reinterpret_cast<T*>(smem_raw);
  uint64_t* mbar = reinterpret_cast<uint64_t*>(smem_raw + (size_t)RowsPerBlock * D * sizeof(T));

  const int warp_id = threadIdx.x / 32;
  const int lane_id = threadIdx.x % 32;
  const int64_t row_base = (int64_t)blockIdx.x * RowsPerBlock;
  const int num_rows = static_cast<int>(min((int64_t)RowsPerBlock, N - row_base));

  if (threadIdx.x == 0) {
    for (int s = 0; s < RowsPerBlock; s++) {
      ptx::mbarrier_init(&mbar[s], 1u);
    }
  }
  __syncthreads();

  if (warp_id == 0) {
    for (int i = 0; i < num_rows; i++) {
      if (lane_id == 0) {
        int64_t row = row_base + i;
        uint32_t size = D * sizeof(T);
        ptx::mbarrier_arrive_expect_tx(
            ptx::sem_release, ptx::scope_cta, ptx::space_shared, &mbar[i], size);
        ptx::cp_async_bulk(
            ptx::space_shared, ptx::space_global,
            smem + i * D,
            src + (int64_t)row * D,
            size,
            &mbar[i]);
      }
    }
  } else {
    for (int i = 0; i < num_rows; i++) {
      if (lane_id == 0) {
        while (!ptx::mbarrier_try_wait_parity(&mbar[i], 0u)) {}
      }
      __syncwarp();

      if (lane_id == 0) {
        int64_t row = row_base + i;
        auto idx_dim = (int64_t)index[row];
        CUDA_KERNEL_ASSERT(idx_dim >= 0 && idx_dim < index_size
          && "scatter gather kernel index out of bounds");
        T* dst_ptr = dst + idx_dim * D;
        ptx::fence_proxy_async(ptx::space_shared);
        ptx::cp_reduce_async_bulk(
            ptx::space_global, ptx::space_shared, ptx::op_add,
            dst_ptr,
            smem + i * D,
            static_cast<uint32_t>(D * sizeof(T)));
        ptx::cp_async_bulk_commit_group();
      }
    }

    if (lane_id == 0) {
      ptx::cp_async_bulk_wait_group_read(ptx::n32_t<0>{});
    }
  }

  __syncthreads();
#endif // __CUDA_ARCH__ >= 900
}

template <typename T, typename index_t, int RowsPerBlock>
void launch_tma_scatter_add(
    T* dst,
    const T* src,
    const index_t* index,
    int64_t N,
    int64_t D,
    int64_t index_size) {
  if (N == 0) return;

  auto stream = at::cuda::getCurrentCUDAStream();
  int64_t grid = (N + RowsPerBlock - 1) / RowsPerBlock;
  TORCH_INTERNAL_ASSERT(grid <= at::cuda::getCurrentDeviceProperties()->maxGridSize[0],
      "TMA scatter_add grid size exceeds device limit");

  size_t smem_bytes = RowsPerBlock * D * sizeof(T)
                    + RowsPerBlock * sizeof(uint64_t);

  int default_smem = at::cuda::getCurrentDeviceProperties()->sharedMemPerBlock;
  if (smem_bytes > static_cast<size_t>(default_smem)) {
    C10_CUDA_CHECK(cudaFuncSetAttribute(
        tma_scatter_add_kernel<T, index_t, RowsPerBlock>,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        smem_bytes));
  }

  tma_scatter_add_kernel<T, index_t, RowsPerBlock><<<static_cast<unsigned int>(grid), 64, smem_bytes, stream>>>(
      dst, src, index, N, D, index_size);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template <int RowsPerBlock>
void tma_scatter_add_impl(
    const Tensor& self,
    int64_t dim,
    const Tensor& index,
    const Tensor& src) {
  // 2D only: self[M, D].scatter_add_(0, index[N, D], src[N, D])
  //   dst_row = index[row]
  TORCH_INTERNAL_ASSERT(dim == 0 && self.dim() == 2);

  const int64_t N = src.size(0);
  const int64_t D = self.size(1);
  const int64_t index_size = self.size(0);

  // Index has stride 0 on dim 1 (broadcast), so just take column 0.
  Tensor index_flat = index.select(1, 0).contiguous();

  auto launch = [&](auto* idx_ptr) {
    using index_t = std::remove_const_t<std::remove_pointer_t<decltype(idx_ptr)>>;
    switch (self.scalar_type()) {
      case at::ScalarType::BFloat16:
        launch_tma_scatter_add<__nv_bfloat16, index_t, RowsPerBlock>(
            reinterpret_cast<__nv_bfloat16*>(self.data_ptr()),
            reinterpret_cast<const __nv_bfloat16*>(src.data_ptr()),
            idx_ptr, N, D, index_size);
        break;
      case at::ScalarType::Half:
        launch_tma_scatter_add<__half, index_t, RowsPerBlock>(
            reinterpret_cast<__half*>(self.data_ptr()),
            reinterpret_cast<const __half*>(src.data_ptr()),
            idx_ptr, N, D, index_size);
        break;
      case at::ScalarType::Float:
        launch_tma_scatter_add<float, index_t, RowsPerBlock>(
            reinterpret_cast<float*>(self.data_ptr()),
            reinterpret_cast<const float*>(src.data_ptr()),
            idx_ptr, N, D, index_size);
        break;
      default:
        TORCH_INTERNAL_ASSERT(false, "Unsupported dtype for TMA scatter_add");
    }
  };

  if (index_flat.scalar_type() == at::kInt) {
    launch(index_flat.data_ptr<int32_t>());
  } else {
    launch(index_flat.data_ptr<int64_t>());
  }
}

} // anonymous namespace

void tma_scatter_add_dispatch(
    const Tensor& self,
    int64_t dim,
    const Tensor& index,
    const Tensor& src,
    int rows_per_block) {
  switch (rows_per_block) {
    case 16: tma_scatter_add_impl<16>(self, dim, index, src); break;
    case 8:  tma_scatter_add_impl<8>(self, dim, index, src); break;
    case 4:  tma_scatter_add_impl<4>(self, dim, index, src); break;
    case 2:  tma_scatter_add_impl<2>(self, dim, index, src); break;
    case 1:  tma_scatter_add_impl<1>(self, dim, index, src); break;
    default:
      TORCH_INTERNAL_ASSERT(false, "Unsupported rows_per_block: ", rows_per_block);
  }
}

} // namespace at::native

#else // !USE_ROCM && !_WIN32 && CUDA_VERSION >= 12080

namespace at::native {

void tma_scatter_add_dispatch(
    const Tensor& self,
    int64_t dim,
    const Tensor& index,
    const Tensor& src,
    int rows_per_block) {
  TORCH_CHECK(false,
      "TMA scatter_add requires CUDA 12.8+ and is not supported on ROCm or Windows");
}

} // namespace at::native

#endif
