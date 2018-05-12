import torch
import torch.nn as nn
# The squash function specified in Dynamic Routing Between Capsules
# x: input tensor 
def squash(x):
  norm_squared = (x ** 2).sum(-1, keepdim=True)
  part1 = norm_squared / (1 +  norm_squared)
  part2 = x / torch.sqrt(norm_squared)

  output = part1 * part2 
  return output

def weights_init_xavier(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.xavier_normal_(m.weight.data, gain=0.02)
    elif classname.find('Linear') != -1:
        nn.init.xavier_normal_(m.weight.data, gain=0.02)
    elif classname.find('BatchNorm2d') != -1:
        nn.init.normal(m.weight.data, 1.0, 0.02)
        nn.init.constant(m.bias.data, 0.0)