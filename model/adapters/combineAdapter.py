import torch

from model.adapters.adapterBase import AdapterBase
from model.adapters.biasAdapter import BiasAdapter
from model.adapters.modalAdapter import ModalAdapter
from model.adapters.modalEmbedAdapter import ModalEmbedAdapter
from model.tools.Datapoint import Datapoint
from model.tools.modelResults import ModelResults
from core.Config import config


class CombineAdapter(AdapterBase):
    def __init__(self):
        super(CombineAdapter, self).__init__()
        adapter = BiasAdapter
        if 'embed' in config.adapter:
            adapter = ModalEmbedAdapter
        if 'basic' in config.adapter:
            adapter = ModalAdapter
        self.acousticAdapter = adapter.CreateModalAdapter('acoustic')
        self.visualAdapter = adapter.CreateModalAdapter('visual')
        weight_visual = config.late_weighted_visual
        weight_acoustic = config.late_weighted_acoustic
        combined = weight_acoustic + weight_visual
        self.weight_visual = weight_visual / combined
        self.weigh_acoustic = weight_acoustic / combined

    def Run(self, datapoint: Datapoint) -> ModelResults:
        acoustic = self.acousticAdapter.Run(datapoint)
        visual = self.visualAdapter.Run(datapoint)
        self.acousticAdapter.ApplyLoss(acoustic, datapoint)
        self.visualAdapter.ApplyLoss(visual, datapoint)
        results = acoustic.result * self.weight_visual + visual.result * self.weigh_acoustic
        return ModelResults(results, [acoustic.FirstAdvResult(), visual.FirstAdvResult()])

    # relying on correct ordering between run and apply loss for adversarial data
    def ApplyLoss(self, results: ModelResults, datapoint: Datapoint):
        pass

    def StepOptimizer(self):
        self.acousticAdapter.StepOptimizer()
        self.visualAdapter.StepOptimizer()

