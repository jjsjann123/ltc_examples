import torch
import lazy_tensor_core
import lazy_tensor_core.debug.metrics as metrics
import lazy_tensor_core.core.lazy_model as ltm

lazy_tensor_core._LAZYC._ltc_init_ts_backend()
lazy_tensor_core._LAZYC._initialize_aten_bindings()

device = 'xla'
dtype = torch.float32
ITER = 2

x = torch.randn(2, 4, device=device, dtype=dtype)

class MyModule(torch.nn.Module):
  def __init__(self):
    super(MyModule, self).__init__()
    self.linear = torch.nn.Linear(4, 4)
    self.gelu = torch.nn.functional.gelu
  def forward(self, x):
    o = self.linear(x)
    o = self.gelu(o)
    return o

module = MyModule().to(device=device, dtype=dtype)
optim = torch.optim.SGD(module.parameters(), lr = 1e-2, momentum = 0.0)

for i in range(ITER):
  o = module(x)
  o.sum().backward()
  optim.step()
  ltm.mark_step()

print(metrics.metrics_report())
