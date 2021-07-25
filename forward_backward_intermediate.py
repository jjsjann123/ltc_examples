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

x = torch.randn(2, 4, device=device, dtype=dtype)
y = torch.empty(2, dtype=torch.long, device=device).random_(4)

class MyModule(torch.nn.Module):
  def __init__(self):
    super(MyModule, self).__init__()
    self.softmax = torch.nn.Softmax()
    self.linear = torch.nn.Linear(4, 4)
    self.gelu = torch.nn.functional.gelu
  def forward(self, x):
    o = self.linear(x)
    o = o + 1.0
    o = self.softmax(o)
    o = self.gelu(o)
    return o

module = MyModule().to(device=device, dtype=dtype)
#for p in module.parameters():
#  print(p)
optim = torch.optim.SGD(module.parameters(), lr = 1e-2, momentum = 0.0)

loss = torch.nn.CrossEntropyLoss()

use_loss = False

with torch.jit.fuser("fuser2"):
  for i in range(ITER):
    o = module(x)
    
    if use_loss:
      loss(o, y).backward()
    else:
      o.sum().backward()
    optim.step()
    ltm.mark_step()
    #for p in module.parameters():
    #  print(p)

#print("after: ", module.linear.bias)

print(metrics.metrics_report())
