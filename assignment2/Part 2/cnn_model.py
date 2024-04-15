from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch.nn as nn

class CNN(nn.Module):

  def __init__(self, n_channels, n_classes):
    """
    Initializes CNN object. 
    
    Args:
      n_channels: number of input channels
      n_classes: number of classes of the classification problem
    """
    super(CNN, self).__init__()
    self.layers = []
    self.layers.append(nn.Conv2d(n_channels, 64, kernel_size=3, padding=1))


  def forward(self, x):
    """
    Performs forward pass of the input.
    
    Args:
      x: input to the network
    Returns:
      out: outputs of the network
    """
    return out
