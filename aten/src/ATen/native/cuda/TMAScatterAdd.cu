#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/native/cuda/TMAScatterAdd.h>
#include <ATen/core/Tensor.h>
#include <ATen/Dispatch.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/ops/arange.h>
#include <c10/cuda/CUDAGuard.h>

#if !defined(USE_ROCM) && !defined(_WIN32) && defined(CUDA_VERSION) && CUDA_VERSION >= 12080

#include <cuda/ptx>
#include <cuda_bf16.h>
#include <cuda_fp16.h>

namespace at::native {

namespace ptx = ::cuda::ptx;

namespace {

template <typename T, int RowsPerBlock>
__global__ void __launch_bounds__(64)
tma_scatter_add_kernel(
    T* __restrict__ dst,
    const T* __restrict__ src,
    const int64_t* __restrict__ index,
    int N,
    int D) {
#if __CUDA_ARCH__ >= 900
  extern __shared__ char smem_raw[];
  T* smem = reinterpret_cast<T*>(smem_raw);
  // Row buffer is 16-byte aligned (D * sizeof(T) % 16 == 0), so mbarriers
  // that follow are naturally 8-byte aligned.
  uint64_t* mbar = reinterpret_cast<uint64_t*>(smem_raw + (size_t)RowsPerBlock * D * sizeof(T));

  const int warp_id = threadIdx.x / 32;
  const int lane_id = threadIdx.x % 32;
  const int row_base = blockIdx.x * RowsPerBlock;
  const int num_rows = min(RowsPerBlock, N - row_base);

  if (threadIdx.x == 0) {
    for (int s = 0; s < RowsPerBlock; s++) {
      ptx::mbarrier_init(&mbar[s], 1u);
    }
  }
  __syncthreads();

  if (warp_id == 0) {
    for (int i = 0; i < num_rows; i++) {
      if (lane_id == 0) {
        int row = row_base + i;
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
        int row = row_base + i;
        T* dst_ptr = dst + index[row] * D;
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

template <typename T, int RowsPerBlock>
void launch_tma_scatter_add(
    T* dst,
    const T* src,
    const int64_t* index,
    int N,
    int D) {
  if (N == 0) return;

  auto stream = at::cuda::getCurrentCUDAStream();
  int grid = (N + RowsPerBlock - 1) / RowsPerBlock;

  // Row buffer is already 16-byte aligned (dispatch guard ensures
  // D * sizeof(T) % 16 == 0), so mbarrier array is naturally 8-byte aligned.
  size_t smem_bytes = RowsPerBlock * D * sizeof(T)
                    + RowsPerBlock * sizeof(uint64_t);

  // Dynamic smem beyond the device default requires opt-in via cudaFuncSetAttribute.
  int default_smem = at::cuda::getCurrentDeviceProperties()->sharedMemPerBlock;
  if (smem_bytes > static_cast<size_t>(default_smem)) {
    C10_CUDA_CHECK(cudaFuncSetAttribute(
        tma_scatter_add_kernel<T, RowsPerBlock>,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        smem_bytes));
  }

  tma_scatter_add_kernel<T, RowsPerBlock><<<grid, 64, smem_bytes, stream>>>(
      dst, src, index, N, D);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template <int RowsPerBlock>
void tma_scatter_add_impl(
    const Tensor& self,
    int64_t dim,
    const Tensor& index,
    const Tensor& src) {
  auto ndim = self.dim();

  int64_t batch_size = 1;
  for (int d = 0; d < dim; d++) {
    batch_size *= self.size(d);
  }
  int64_t scatter_size = self.size(dim);
  int64_t J = src.size(dim);
  int64_t trailing_size = 1;
  for (int d = dim + 1; d < ndim; d++) {
    trailing_size *= self.size(d);
  }

  // Extract core index: one value per (batch, scatter_idx) combination.
  // Select index 0 along each trailing dim (all have stride 0 so values are identical).
  Tensor idx_core = index;
  for (int d = ndim - 1; d > dim; d--) {
    idx_core = idx_core.select(d, 0);
  }
  idx_core = idx_core.contiguous();

  Tensor index_1d;
  if (batch_size == 1) {
    index_1d = idx_core.reshape({-1});
  } else {
    auto idx_2d = idx_core.reshape({batch_size, J});
    auto batch_offsets = at::arange(
        batch_size, index.options().dtype(at::kLong));
    batch_offsets.mul_(scatter_size);
    index_1d = idx_2d.add(batch_offsets.unsqueeze(1)).reshape({-1}).contiguous();
  }

  const int N = static_cast<int>(batch_size * J);
  const int D = static_cast<int>(trailing_size);

  switch (self.scalar_type()) {
    case at::ScalarType::BFloat16:
      launch_tma_scatter_add<__nv_bfloat16, RowsPerBlock>(
          reinterpret_cast<__nv_bfloat16*>(self.data_ptr()),
          reinterpret_cast<const __nv_bfloat16*>(src.data_ptr()),
          index_1d.data_ptr<int64_t>(), N, D);
      break;
    case at::ScalarType::Half:
      launch_tma_scatter_add<__half, RowsPerBlock>(
          reinterpret_cast<__half*>(self.data_ptr()),
          reinterpret_cast<const __half*>(src.data_ptr()),
          index_1d.data_ptr<int64_t>(), N, D);
      break;
    case at::ScalarType::Float:
      launch_tma_scatter_add<float, RowsPerBlock>(
          reinterpret_cast<float*>(self.data_ptr()),
          reinterpret_cast<const float*>(src.data_ptr()),
          index_1d.data_ptr<int64_t>(), N, D);
      break;
    default:
      TORCH_INTERNAL_ASSERT(false, "Unsupported dtype for TMA scatter_add");
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
