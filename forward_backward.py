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
ITER = 1

x = torch.randn(2, 16, 4, device=device, dtype=dtype)

class MyModule(torch.nn.Module):
  def __init__(self):
    super(MyModule, self).__init__()
    self.softmax = torch.nn.Softmax()
    self.linear = torch.nn.Linear(4, 4)
    self.linear2 = torch.nn.Linear(4, 4)
    #self.softmax = torch.nn.functional.gelu
  def forward(self, x):
    o = self.linear2(x)
    o = self.softmax(o)
    o = self.linear(o)
    #o += 1.0
    o = o + 1.0
    return o

module = MyModule().to(device=device, dtype=dtype)
for p in module.parameters():
  print(p)
optim = torch.optim.SGD(module.parameters(), lr = 1e-2, momentum = 0.0)

#print("before: ", module.linear.bias)

with torch.jit.fuser("fuser2"):
  for i in range(ITER):
    o = module(x)
    o.sum().backward()
    optim.step()
    ltm.mark_step()
    #for p in module.parameters():
    #  print(p)

#print("after: ", module.linear.bias)

print(metrics.metrics_report())
