#include <torch/extension.h>

#include <cuda.h>
#include <cuda_runtime.h>

template <typename scalar_t>
__global__ void copy_kernel(scalar_t *inp, scalar_t *out, size_t N) {
  for (size_t i = 0; i < N; ++i) {
    out[i] = inp[i];
  }
}

void c_copy(const at::Tensor& a, at::Tensor& b) {
  AT_DISPATCH_FLOATING_TYPES(a.type(), "custom copy kernel", ([&] {
    copy_kernel<scalar_t><<<1, 1>>>(a.data<scalar_t>(), b.data<scalar_t>(), a.numel());
  }));
}
