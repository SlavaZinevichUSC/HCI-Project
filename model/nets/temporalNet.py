from torch import nn
from model.tools.Storage import Datapoint
import torch
from core.Config import config


class TemporalNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, n_layers, output_dim, dropout_prob=config.dropout_prob):
        super(TemporalNet, self).__init__()
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim
        self.temporal = nn.GRU(input_dim, hidden_dim, n_layers, dropout=dropout_prob)
        self.out = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, output_dim),
                                 nn.Softmax(dim=0))
        self.initial = self.init_hidden()

    def forward(self, x):
        gru, h = self.temporal(x, self.initial)
        return self.out(gru[-1, :])

    def init_hidden(self):
        weight = next(self.parameters()).data
        hidden = weight.new(self.n_layers, self.hidden_dim).zero_()
        return hidden
