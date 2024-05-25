from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn

################################################################################

class LSTM(nn.Module):

    def __init__(self, seq_length, input_dim, hidden_dim, output_dim, batch_size, device):
        super(LSTM, self).__init__()
        # Initialization here ...
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.batch_size = batch_size

        # g_t
        self.Wgx = nn.Linear(input_dim, hidden_dim)
        self.Wgh = nn.Linear(hidden_dim, hidden_dim)
        self.bg = torch.zeros(batch_size, hidden_dim)
        # i_t
        self.Wix = nn.Linear(input_dim, hidden_dim)
        self.Wih = nn.Linear(hidden_dim, hidden_dim)
        self.bi = torch.zeros(batch_size, hidden_dim)
        # f_t
        self.Wfx = nn.Linear(input_dim, hidden_dim)
        self.Wfh = nn.Linear(hidden_dim, hidden_dim)
        self.bf = torch.zeros(batch_size, hidden_dim)
        # o_t
        self.Wox = nn.Linear(input_dim, hidden_dim)
        self.Woh = nn.Linear(hidden_dim, hidden_dim)
        self.bo = torch.zeros(batch_size, hidden_dim)
        # p_t
        self.Wph = nn.Linear(hidden_dim, output_dim)
        self.bp = torch.zeros(batch_size, output_dim)
        # activation
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()
        # softmax
        self.softmax = nn.Softmax(dim=1)


    def forward(self, x):
        # x.size: 0 (batch_size), 1 (sequence_length), 2 (input_dim(=1 if not one-hot encoded))
        # Init hidden states to the vector of all zeros, shape = (batch_size, hidden_dim)
        h = torch.zeros(self.hidden_dim, self.hidden_dim, device=x.device)
        c = torch.zeros(self.hidden_dim, self.hidden_dim, device=x.device)

        for t in range(x.size(1)):
            x_t = x[:,t,:]
            g = self.tanh(self.Wgx(x_t) + self.Wgh(h) + self.bg)
            i = self.sigmoid(self.Wix(x_t) + self.Wih(h) + self.bi)
            f = self.sigmoid(self.Wfx(x_t) + self.Wfh(h) + self.bf)
            o = self.sigmoid(self.Wox(x_t) + self.Woh(h) + self.bo)
            c = g * i + c * f
            h = self.tanh(c) * o

        p = self.Wph(h) + self.bp
        out = self.softmax(p)
        return out
        