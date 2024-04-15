from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn

class MLP(nn.Module):
    
    def __init__(self, n_inputs, n_hidden, n_classes):
        """
        Initializes multi-layer perceptron object.    
        Args:
            n_inputs: number of inputs (i.e., dimension of an input vector).
            n_hidden: list of integers, where each integer is the number of units in each linear layer
            n_classes: number of classes of the classification problem (i.e., output dimension of the network)
        """
        super(MLP, self).__init__()
        
        self.layers = nn.ModuleList()
        # add hidden layers
        for num_nodes in n_hidden:
            self.layers.append(nn.Linear(n_inputs, num_nodes))
            self.layers.append(nn.ReLU())
            n_inputs = num_nodes
        # add output layer
        self.layers.append(nn.Linear(n_inputs, n_classes))
        self.layers.append(nn.Softmax(dim=1))


    def forward(self, x):
        """
        Predict network output from input by passing it through several layers.
        Args:
            x: input to the network
        Returns:
            out: output of the network
        """
        out = x
        for layer in self.layers:
            out = layer(out)
        return out
