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

    def AddLossType(self, lossName : str):
        self.loss[lossName] = []

    def AddMultiLossType(self, lossNames: [str]):
        for name in lossNames:
            self.AddLossType(name)

    def AddLoss(self, lossName, loss):
        self.loss[lossName].append(loss)

    def Run(self, datapoint: Datapoint) -> ModelResults:
        return ModelResults.Empty()

    def ApplyLoss(self, results : ModelResults, datapoint: Datapoint):
        pass

    def BatchApplyLoss(self):
        losses = [sum(loss) / len(loss) for loss in self.loss.values()]
        for k,v in self.loss.items():
            self.loss[k] = []
        torch.autograd.backward(losses)
        self.StepOptimizer()
        for opt in self.optimizers:
            opt.step()
            opt.zero_grad()

    def StepOptimizer(self):
        pass


