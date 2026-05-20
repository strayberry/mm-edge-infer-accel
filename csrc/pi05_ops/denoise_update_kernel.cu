#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

namespace {

__global__ void denoise_update_kernel(
    const float* __restrict__ x,
    const float* __restrict__ v,
    float* __restrict__ out,
    float dt,
    int64_t n) {
  int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx < n) {
    out[idx] = x[idx] + dt * v[idx];
  }
}

}  // namespace

torch::Tensor denoise_update_cuda(torch::Tensor x, torch::Tensor v, double dt) {
  auto out = torch::empty_like(x);
  const int64_t n = x.numel();
  if (n == 0) {
    return out;
  }

  constexpr int threads = 256;
  const int blocks = static_cast<int>((n + threads - 1) / threads);
  denoise_update_kernel<<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
      x.data_ptr<float>(),
      v.data_ptr<float>(),
      out.data_ptr<float>(),
      static_cast<float>(dt),
      n);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}
