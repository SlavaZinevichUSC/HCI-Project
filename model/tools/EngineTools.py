from torch import nn, optim
from core.Config import config


def GetLoss() -> nn.modules.loss:
    lossFns = {'none': nn.CrossEntropyLoss, 'ce': nn.CrossEntropyLoss}
    return lossFns.get(config.lossFn, nn.CrossEntropyLoss)()


def GetOptimizer(parameters, lr = None) -> optim.Optimizer:
    if lr is None:
        lr = config.optimizer_lr
    opts = {'none': optim.Adam, 'adam': optim.Adam}
    opt = opts.get(config.optimizer, optim.Adam)(parameters, lr = lr)
    opt.zero_grad()
    return opt
