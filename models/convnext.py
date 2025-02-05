import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F


class LayerNorm2d(nn.Module):
    def __init__(self, in_channels):
      super().__init__()
      #TODO

    def forward(self, x: Tensor) -> Tensor:
      #TODO

class ConvNextStem(nn.Module):
   def __init__(self, in_channels, out_channels, kernel_size=3):
    super().__init__()

    # TODO
  
   def forward(self,x):
    # TODO

class ConvNextBlock(nn.Module):

  def __init__(self, d_in, layer_scale=1e-6, kernel_size=7, stochastic_depth_prob=1):
    super().__init__()

    #TODO

  def forward(self,x):
    #TODO

class ConvNextDownsample(nn.Module):
  def __init__(self, d_in, d_out, width=2):
    super().__init__()

    #TODO

  def forward(self,x):
    #TODO

class ConvNextClassifier(nn.Module):
  def __init__(self, d_in, d_out):
    super().__init__()

    #TODO

  def forward(self,x):
    #TODO


class ConvNext(nn.Module):

  def __init__(self, in_channels, out_channels, blocks=[96]):
    super().__init__()

    #TODO

  def forward(self,x):
    #TODO