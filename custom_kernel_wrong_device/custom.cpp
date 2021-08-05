#include <torch/extension.h>
#include <iostream>

void c_copy(const at::Tensor& a, at::Tensor& b);

at::Tensor custom_copy(
    torch::Tensor input) {
#if false
  auto cuda_input = input.cuda();
  auto cuda_output = torch::rand_like(cuda_input);
  c_copy(cuda_input, cuda_output);
  return cuda_output.to(input.options()).set_requires_grad(input.requires_grad());
#else
  auto output = torch::empty_like(input);
  c_copy(input, output);
  return output;
#endif
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("custom_copy", &custom_copy, "Simply copy tensor (CUDA)");
}
