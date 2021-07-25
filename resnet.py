import torch
import lazy_tensor_core
import lazy_tensor_core.debug.metrics as metrics
import lazy_tensor_core.core.lazy_model as ltm

import torch.optim as optim
import torchvision.models as models

lazy_tensor_core._LAZYC._ltc_init_ts_backend()
lazy_tensor_core._LAZYC._initialize_aten_bindings()

torch.manual_seed(42)

device = 'xla'
#device = 'cuda'
dtype = torch.float32
ITER = 2

model = models.resnet50().to(device=device)
inputs = torch.randn(32, 3, 224, 224).to(device=device, dtype=dtype)
loss = torch.nn.CrossEntropyLoss()

#for p in module.parameters():
#  print(p)
optimizer = optim.Adam(model.parameters())

with torch.jit.fuser("fuser2"):
  for i in range(ITER):
    pred = model(inputs)
    y = torch.empty(pred.shape[0], dtype=torch.long, device=device).random_(pred.shape[1])
    loss(pred, y).backward()
    optimizer.step()
    ltm.mark_step()
    #for p in module.parameters():
    #  print(p)

#print("after: ", module.linear.bias)

print(metrics.metrics_report())
