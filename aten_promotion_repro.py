import torch
import lazy_tensor_core
import lazy_tensor_core.debug.metrics as metrics
import lazy_tensor_core.core.lazy_model as ltm

lazy_tensor_core._LAZYC._ltc_init_ts_backend()
lazy_tensor_core._LAZYC._initialize_aten_bindings()

torch.manual_seed(42)

device = 'xla'
#device = 'cuda'
dtype = torch.float32
ITER = 3

x = torch.randn(4, 4, device=device, dtype=torch.float16).requires_grad_()
w = torch.randn(4, 4, device=device, dtype=torch.float16).requires_grad_()
y = torch.randn(4, 4, device=device, dtype=dtype).requires_grad_()
ww = torch.randn(4, 4, device=device, dtype=dtype).requires_grad_()

#@torch.jit.script
def f(x, y, w, ww):
  o1 = torch.mm(x, w)
  o2 = torch.mm(y, ww)
  o = o1 + o2
  return o

for i in range(ITER):
  o = f(x, y, w, ww)
  o.sum().backward()
  ltm.mark_step()

print(metrics.metrics_report())
