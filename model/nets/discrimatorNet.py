from torch import nn
from model.tools.Storage import Datapoint
import torch
from core.Config import config


class DiscriminatorNet(nn.Module):
    def __init__(self, embed_dim, hidden_dim, num_labels):
        super(DiscriminatorNet, self).__init__()
        self.discriminator = nn.Sequential(nn.Linear(embed_dim, hidden_dim),
                                           nn.ReLU(),
                                           nn.Linear(hidden_dim, hidden_dim),
                                           nn.ReLU(),
                                           nn.Linear(hidden_dim, num_labels),
                                           nn.Softmax(dim=1))

    def forward(self, x):
        return self.discriminator(x)