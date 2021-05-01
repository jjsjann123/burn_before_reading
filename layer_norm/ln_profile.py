import torch
import torch.nn.functional as F
import itertools

from torch.autograd.profiler import profile as profile

from typing_extensions import Final
from typing import List

import timeit

torch._C._debug_set_autodiff_subgraph_inlining(False)

class MyModule(torch.nn.Module):
  normalized_shape: Final[List[int]]

  def __init__(self, w, b, num_feature):
    super(MyModule, self).__init__()
    if w:
      self.w = torch.nn.parameter.Parameter(torch.empty(num_feature))
    else:
      self.register_parameter("w", None)
    if b:
      self.b = torch.nn.parameter.Parameter(torch.empty(num_feature))
    else:
      self.register_parameter("b", None)

    self.normalized_shape = [num_feature]

  def forward(self, input):
    return torch.nn.functional.layer_norm(input, self.normalized_shape, self.w, self.b)

device = "cpu"
s = 2

my_module = MyModule(True, True, s)
func = torch.jit.script(my_module)

inputs = torch.randn(s, s, s, s, device=device, dtype=torch.float, requires_grad=True)
grad = torch.randn(s, s, s, s, device=device, dtype=torch.float)

# warmup/profile/optimization runs
for i in range(0, 4):
  l1 = func(inputs)
  grads1 = torch.autograd.grad(l1, inputs, grad_outputs=[grad], create_graph=True)[0]
  l2 = grads1 * l1
  grads2 = torch.autograd.grad(l2, inputs, grad_outputs=[grad])[0]

print(timeit.timeit(lambda : func(inputs).backward(grad), number=100))

l1 = func(inputs)
with profile(with_stack=True, record_shapes=True) as prof:
  grads1 = torch.autograd.grad(l1, inputs, grad_outputs=[grad], create_graph=True)[0]
print("== grad profiles:\n", prof.key_averages().table(sort_by="cpu_time_total", row_limit=100))

l2 = grads1 * l1
with profile(with_stack=True, record_shapes=True) as prof:
  grads2 = torch.autograd.grad(l2, inputs, grad_outputs=[grad])[0]
print("== gradgrad profiles:", prof.key_averages().table(sort_by="cpu_time_total", row_limit=100))


