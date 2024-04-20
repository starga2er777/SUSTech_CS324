from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn

class VanillaRNN(nn.Module):

    def __init__(self, input_length, input_dim, hidden_dim, output_dim, device):
        super(VanillaRNN, self).__init__()
        
        self.input_length = input_length
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.device = device
        
        # Define the layers:
        # w_hx: Weight for the input to hidden layer connections.
        self.w_hx = nn.Linear(input_dim, hidden_dim)
        # w_hh: Weight for the hidden to hidden layer connections (recurrent connections).
        self.w_hh = nn.Linear(hidden_dim, hidden_dim)
        # w_ph: Weight for the hidden to output layer connections.
        self.w_ph = nn.Linear(hidden_dim, output_dim)
        # activation: tanh
        self.activation = nn.Tanh()
        # softmax: normalizing the output to a probability distribution (0~9).
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        # x.size: 0 (batch_size), 1 (sequence_length), 2 (input_dim(=1 if not one-hot encoded))
        # Init hidden state to the vector of all zeros, shape = (batch_size, hidden_dim)
        h = torch.zeros(x.size(0), self.hidden_dim, device=x.device)

        # for each digit: update hidden state h
        for t in range(x.size(1)):
            # x_t shape: (batch_size, input_dim)
            x_t = x[:,t,:]
            # h_t = tanh(w_hx * x_t + w_hh * h_{t-1} + b_h)
            h = self.activation(self.w_hx(x_t) + self.w_hh(h))
        
        # o: shape: (batch_size, output_dim)
        o = self.w_ph(h)
        out = self.softmax(o)
        return out
        
