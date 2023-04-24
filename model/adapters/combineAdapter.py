import torch

from model.adapters.adapterBase import AdapterBase
from model.adapters.biasAdapter import BiasAdapter
from model.tools.Datapoint import Datapoint
from model.tools.modelResults import ModelResults
from core.Config import config


class CombineAdapter(AdapterBase):
    def __init__(self):
        self.acousticAdapter = BiasAdapter.CreateModalAdapter('acoustic')
        self.visualAdapter = BiasAdapter.CreateModalAdapter('visual')
        weight_visual = config.late_weighted_visual
        weight_acoustic = config.late_weighted_acoustic
        combined = weight_acoustic + weight_visual
        self.weight_visual = weight_visual / combined
        self.weigh_acoustic = weight_acoustic / combined

    def Run(self, datapoint: Datapoint) -> ModelResults:
        acoustic = self.acousticAdapter.Run(datapoint)
        visual = self.visualAdapter.Run(datapoint)
        results = acoustic.result * self.weight_visual + visual.result * self.weigh_acoustic
        return ModelResults(results, [acoustic.FirstAdvResult(), visual.FirstAdvResult()])
        pass

    # relying on correct ordering between run and apply loss for adversarial data
    def ApplyLoss(self, results: ModelResults, datapoint: Datapoint):
        lossA = self.acousticAdapter.GetLoss(ModelResults(results.result, [results.advResult[0]]), datapoint)
        lossV = self.visualAdapter.GetLoss(ModelResults(results.result, [results.advResult[1]]), datapoint)
        torch.autograd.backward(lossA + lossV)
        self.acousticAdapter.StepOptimizer()
        self.visualAdapter.StepOptimizer()

