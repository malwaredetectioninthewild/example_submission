
import torch
from torch import nn
from torch.nn import functional as F
import math


class ResnetBackbone(nn.Module):
    def __init__(self, cfg):
                         
        super(ResnetBackbone, self).__init__()

        self.__name__ = 'Resnet'

        self.cfg = cfg

        regularization_coeff = cfg['regularization_coeff']
        capacity_coeff = cfg['capacity_coeff']
        input_len = cfg['input_len']

        self.activation = F.relu
        self.rdo = regularization_coeff*0.1
        self.hdo = regularization_coeff*0.2

        hidden_size = int(capacity_coeff*256)
        hidden_factor = 1.5
        n_layers = 2

        self.first_layer = nn.Linear(input_len, hidden_size)

        inner_hidden = int(hidden_size * hidden_factor)

        self.layers = nn.ModuleList()

        for _ in range(n_layers):

            cur_layer = nn.ModuleDict({'norm': nn.BatchNorm1d(hidden_size), 
                                       'linear0': nn.Linear(hidden_size, inner_hidden), 
                                       'linear1': nn.Linear(inner_hidden, hidden_size)})
            
            self.layers.append(cur_layer)

        self.last_normalization = nn.BatchNorm1d(hidden_size)
        self.feat_dim = hidden_size
        self.init = None

    def forward_w_intermediate(self, x):

        if hasattr(self, 'keep_feats') and self.keep_feats is not None:
            x = x[:, self.keep_feats]

        all_reprs = []
        x = self.first_layer(x)
        all_reprs.append(x)

        for layer in self.layers:
            z = x
            z = layer['norm'](z)
            z = layer['linear0'](z)
            z = self.activation(z)
            if self.hdo > 0:
                z = F.dropout(z, self.hdo, self.training)
            z = layer['linear1'](z)
            if self.rdo:
                z = F.dropout(z, self.rdo, self.training)
            x = x + z
            all_reprs.append(x)

        x = self.last_normalization(x)
        x = self.activation(x)
        return x, all_reprs

    def forward(self, x):
        if hasattr(self, 'keep_feats') and self.keep_feats is not None:
            x = x[:, self.keep_feats]

        x = self.first_layer(x)
        for layer in self.layers:
            z = x
            z = layer['norm'](z)
            z = layer['linear0'](z)
            z = self.activation(z)
            if self.hdo > 0:
                z = F.dropout(z, self.hdo, self.training)
            z = layer['linear1'](z)
            if self.rdo:
                z = F.dropout(z, self.rdo, self.training)
            x = x + z

        x = self.last_normalization(x)
        x = self.activation(x)
        return x