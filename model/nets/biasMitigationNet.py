from torch import nn
from model.tools.Storage import Datapoint
import torch
from core.Config import config


class BiasMitigationNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, embed_dim):
        super(BiasMitigationNet, self).__init__()
        layers = [nn.Linear(input_dim, hidden_dim), nn.ReLU()]
        for _ in range(config.num_debias_layers):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(hidden_dim, embed_dim))
        layers.append(nn.Softmax(dim=0))
        self.debiasFC = nn.Sequential(*layers)

    def forward(self, x):
        return self.debiasFC(x)
