import torch
import lazy_tensor_core
import lazy_tensor_core.debug.metrics as metrics
import lazy_tensor_core.core.lazy_model as ltm

lazy_tensor_core._LAZYC._ltc_init_ts_backend()

from torch.utils.cpp_extension import load
custom_cpp = load(name="custom_cpp", sources=["custom.cpp", "custom.cu"])

torch.manual_seed(42)

device = 'xla'
#device = 'cuda'
dtype = torch.float32
ITER = 5

x = torch.randn(16, 4, device=device, dtype=dtype).requires_grad_()
y = torch.empty(16, device=device, dtype=torch.long).random_(4)

class CustomLayer(torch.autograd.Function):
  @staticmethod
  def forward(ctx, x):
    return custom_cpp.custom_copy(x)

  @staticmethod
  def backward(ctx, x):
    return custom_cpp.custom_copy(x)

class MyModule(torch.nn.Module):
  def __init__(self):
    super(MyModule, self).__init__()
    self.softmax = torch.nn.Softmax()
    self.gelu = torch.nn.functional.gelu
    self.relu = torch.nn.functional.relu
  def forward(self, x):
    o = self.softmax(x)
    o = self.gelu(o)
    o = self.relu(o)
    o = CustomLayer.apply(o)
    return o

module = MyModule().to(device=device, dtype=dtype)
loss_fn = torch.nn.CrossEntropyLoss()

with torch.jit.fuser("fuser2"):
  for i in range(ITER):
    o = module(x)
    loss_fn(o, y).backward()
    ltm.mark_step()

print(metrics.metrics_report())
