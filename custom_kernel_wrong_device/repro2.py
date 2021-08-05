import torch
import lazy_tensor_core
import lazy_tensor_core.debug.metrics as metrics
import lazy_tensor_core.core.lazy_model as ltm

lazy_tensor_core._LAZYC._ltc_init_ts_backend()

from torch.utils.cpp_extension import load
custom_cpp = load(name="custom_cpp", sources=["custom.cpp", "custom.cu"])

torch.manual_seed(42)

device = 'lazy'
dtype = torch.float32

x = torch.randn(16, 4, device=device, dtype=dtype)

def f(x):
  return custom_cpp.custom_copy(x)

o = f(x)
ltm.mark_step()

print(metrics.metrics_report())
