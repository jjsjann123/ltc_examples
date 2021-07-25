import torch
import lazy_tensor_core
import lazy_tensor_core.debug.metrics as metrics
import lazy_tensor_core.core.lazy_model as ltm

lazy_tensor_core._LAZYC._ltc_init_ts_backend()
#lazy_tensor_core._LAZYC._initialize_aten_bindings()

torch._C._jit_set_nvfuser_enabled(False)
torch._C._jit_set_texpr_fuser_enabled(False)
torch._C._jit_set_profiling_executor(True)
torch._C._jit_set_profiling_mode(True)
torch._C._jit_override_can_fuse_on_cpu(True)
torch._C._jit_override_can_fuse_on_gpu(True)
torch._C._jit_set_bailout_depth(20)

device = 'xla'
dtype = torch.float32

x1 = torch.randn(2, 4, device=device, dtype=dtype)

x2 = torch.randn(2, 8, 8, device=device, dtype=dtype)

def f(x):
  o = x + 1.0
  o = torch.nn.functional.gelu(o)
  return o

for i in range(2):
  o = f(x1)
  #print(o)
  ltm.mark_step()

o = f(x2)
ltm.mark_step()

print(metrics.metrics_report())
