import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
import torchvision.ops.stochastic_depth as stochastic_depth


class LayerNorm2d(nn.Module):
    def __init__(self, in_channels):
      super().__init__()
      # Input has shape B x C x H x W
      self.layer_norm = torch.nn.LayerNorm(in_channels)

    def forward(self, x: Tensor) -> Tensor:
      x = x.permute(0, 2, 3, 1)
      x = self.layer_norm(x)
      x = x.permute(0, 3, 1, 2)
      return x

class ConvNextStem(nn.Module):
   def __init__(self, in_channels, out_channels, kernel_size=3):
    super().__init__()
    self.patchy_stem_conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size) #padding kernel?
    # self.patchy_stem_conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=kernel_size) #padding kernel?
  
   def forward(self,x):
    x = self.patchy_stem_conv(x)
    return x



class ConvNextBlock(nn.Module):
  def __init__(self, d_in, layer_scale=1e-6, kernel_size=7, stochastic_depth_prob=1):
    super().__init__()
    # he initial layer scale (default 1e-6), and a stochastic depth probability of survival (default 1). 
    depthwise_multiplier = 1 #?
    # depth_survival = 1
    
    self.depth_conv = nn.Conv2d(d_in, d_in*depthwise_multiplier, kernel_size=kernel_size, groups=d_in, padding=kernel_size // 2)
    self.layer_norm = LayerNorm2d(d_in)
    self.pointwise_conv = nn.Conv2d(in_channels=d_in, out_channels=d_in, kernel_size=1 )#bias=False 

    self.layer_scale = nn.Parameter(layer_scale * torch.ones(d_in))

    # could do if self.training, dont think i care
    self.stochastic_depth_prob = stochastic_depth_prob

  def forward(self,x):
    identity = x
    x = self.depth_conv(x)

    # Permute for LayerNorm (B x C x H x W -> B x H x W x C)
    # x = x.permute(0, 2, 3, 1)
    x = self.layer_norm(x)
    # x = x.permute(0, 3, 1, 2)
    # print("after permute", x.shape)

    # Pointwise convolution
    x = self.pointwise_conv(x)

    # Layer scaling
    x = x * self.layer_scale[None, :, None, None]

    if self.training:
      # Stochastic depth
      mask = torch.rand(x.shape[0], 1, 1, 1, device=x.device) < self.stochastic_depth_prob
      x = x * mask / self.stochastic_depth_prob

      # x = stochastic_depth(x, self.stochastic_depth_prob, "row")
      # if torch.rand(1) < self.stochastic_depth_prob:
      #   x = x / self.stochastic_depth_prob
      # else:
      #   x = torch.zeros

    # Stochastic depth
    # if self.training:
    #   x = self.stochastic_depth(x)

    # Residual connection
    # print(x.shape)
    # print(identity.shape)
    x = x + identity
    return x
       


class ConvNextDownsample(nn.Module):
  def __init__(self, d_in, d_out, width=2):
    super().__init__()
    self.downsample = nn.Conv2d(d_in, d_out, kernel_size=width, stride=width)


  def forward(self,x):
    x = self.downsample(x)
    return x



class ConvNextClassifier(nn.Module):
  def __init__(self, d_in, d_out):
    # global average pooling, a standard layer norm, and a final linear layer to map to the number of classes.
    super().__init__()
    self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
    self.flatten = nn.Flatten()
    self.layer_norm = nn.LayerNorm(d_in)
    self.linear = nn.Linear(d_in, d_out)

  def forward(self,x):
    x = self.global_avg_pool(x)
    x = self.flatten(x)  #dont know if needed?
    x = self.layer_norm(x)
    x = self.linear(x)
    return x


class ConvNext(nn.Module):
  def __init__(self, in_channels, out_channels, blocks=[96]):
    super().__init__()

    for module in self.modules():
      if isinstance(module, (nn.Conv2d, nn.Linear)):
          # Initialize weights with truncated normal distribution
          nn.init.trunc_normal_(module.weight, mean=0.0, std=0.02, a=-2.0, b=2.0)
          # Initialize biases to 0
          if module.bias is not None:
            nn.init.zeros_(module.bias)
      elif isinstance(module, nn.LayerNorm):
          # Initialize LayerNorm weights (gamma) to 1
          nn.init.ones_(module.weight)
          # Initialize LayerNorm biases (beta) to 0
          nn.init.zeros_(module.bias)


    # how is out_channels used?
    self.stem = ConvNextStem(in_channels, 64)
    self.res64_1 = ConvNextBlock(64)
    self.res64_2 = ConvNextBlock(64)
    self.downsample128 = ConvNextDownsample(64, 128)
    self.res128_1 = ConvNextBlock(128)
    self.res128_2 = ConvNextBlock(128)
    self.downsample256 = ConvNextDownsample(128, 256)
    self.res256_1 = ConvNextBlock(256)
    self.res256_2 = ConvNextBlock(256)
    self.downsample512 = ConvNextDownsample(256, 512)
    self.res512_1 = ConvNextBlock(512)
    self.res512_2 = ConvNextBlock(512)
    self.classifier = ConvNextClassifier(512, out_channels)


  def forward(self,x):
    x = self.stem(x)
    x = self.res64_1(x)
    x = self.res64_2(x)
    x = self.downsample128(x)
    x = self.res128_1(x)
    x = self.res128_2(x)
    x = self.downsample256(x)
    x = self.res256_1(x)
    x = self.res256_2(x)
    x = self.downsample512(x)
    x = self.res512_1(x)
    x = self.res512_2(x)
    x = self.classifier(x)
    return x