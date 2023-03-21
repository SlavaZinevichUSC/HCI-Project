from torch import nn, optim
from core.Config import config


def GetLoss() -> nn.modules.loss:
    lossFns = {'none': nn.CrossEntropyLoss, 'ce': nn.CrossEntropyLoss}
    return lossFns.get(config.lossFn, nn.CrossEntropyLoss)()


def GetOptimizer(parameters) -> optim.Optimizer:
    opts = {'none': optim.Adam, 'adam': optim.Adam}
    return opts.get(config.optimizer, optim.Adam)(parameters, lr = config.optimizer_lr)
