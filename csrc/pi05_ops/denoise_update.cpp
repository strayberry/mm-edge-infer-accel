#include <torch/extension.h>

torch::Tensor denoise_update_cuda(torch::Tensor x, torch::Tensor v, double dt);

torch::Tensor denoise_update(torch::Tensor x, torch::Tensor v, double dt) {
  TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
  TORCH_CHECK(v.is_cuda(), "v must be a CUDA tensor");
  TORCH_CHECK(x.scalar_type() == torch::kFloat32, "x must be float32");
  TORCH_CHECK(v.scalar_type() == torch::kFloat32, "v must be float32");
  TORCH_CHECK(x.sizes() == v.sizes(), "x and v must have the same shape");
  TORCH_CHECK(x.is_contiguous(), "x must be contiguous");
  TORCH_CHECK(v.is_contiguous(), "v must be contiguous");
  return denoise_update_cuda(x, v, dt);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("denoise_update", &denoise_update, "Pi0.5 fused denoise update (CUDA)");
}
