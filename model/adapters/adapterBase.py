import torch.optim

from model.tools.Datapoint import Datapoint
from model.tools.modelResults import ModelResults


class AdapterBase():
    def __init__(self):
        self.loss = {}
        self.optimizers : [torch.optim.Optimizer] = []
        pass

    def AddOptimizer(self, opt : torch.optim.Optimizer):
        self.optimizers.append(opt)

    def AddLoss(self, lossName):
        self.loss[lossName] = []

    def Run(self, datapoint: Datapoint) -> ModelResults:
        return ModelResults.Empty()

    def ApplyLoss(self, results : ModelResults, datapoint: Datapoint):
        pass

    def BaseApplyLoss(self):
        losses = [sum(loss) / len(loss) for loss in self.loss.values()]
        torch.autograd.backward(losses)
        for opt in self.optimizers:
            opt.step()
            opt.zero_grad()



