import torch
import lazy_tensor_core
import lazy_tensor_core.core.lazy_model as ltm

lazy_tensor_core._LAZYC._ltc_init_ts_backend()
lazy_tensor_core._LAZYC._initialize_aten_bindings()

device = 'xla'
dtype = torch.float32

x = torch.randn(2, 16, 4, device=device, dtype=dtype).requires_grad_()

class MyModule(torch.nn.Module):
  def __init__(self):
    super(MyModule, self).__init__()
    self.softmax = torch.nn.Softmax()
    self.gelu = torch.nn.functional.gelu
    self.relu = torch.nn.functional.relu
  def forward(self, x):
    o = self.softmax(x)
    o = o + 1.0
    o = self.relu(o)
    return o

module = MyModule().to(device=device, dtype=dtype)
o = module(x)
o.sum().backward()
ltm.mark_step()
