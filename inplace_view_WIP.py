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
ITER = 2

x = torch.randn(2, 4, 4, device=device, dtype=dtype)
y = torch.randn(2, 4, 4, device=device, dtype=dtype)
z = torch.randn(2, 4, 4, device=device, dtype=dtype)

class MyModule(torch.nn.Module):
  def __init__(self):
    super(MyModule, self).__init__()
    self.bn = torch.nn.BatchNorm1d(4)

  def forward(self, x, y, z):
    o = x * y
    #o = torch.tanh_(o)
    o = self.bn(o)
    o = o * z
    return o

module = MyModule().to(device=device, dtype=dtype)
#print(z)

with torch.jit.fuser("fuser2"):
  for i in range(ITER):
    o = module(x, y, z)
    ltm.mark_step()

#print(z)

#print(metrics.metrics_report())
