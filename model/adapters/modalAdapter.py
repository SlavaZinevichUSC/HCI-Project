import torch.autograd

from model.adapters.adapterBase import AdapterBase
from model.nets.biasMitigationNet import BiasMitigationNet
from model.nets.temporalNet import TemporalNet
from model.tools import EngineTools
from model.tools.Datapoint import Datapoint
from model.nets.EarlyFusionNet import EarlyFusionNet
from model.nets.LateFusionNet import LateFusionNet
from core.Config import config as c
from torch import nn
from model.nets.discrimatorNet import DiscriminatorNet
from itertools import chain

from model.tools.modelResults import ModelResults


class ModalAdapter(AdapterBase):  # Ended up unnecessary as all networks have the same API
    def __init__(self, size_in, modality):
        super(ModalAdapter, self).__init__()
        self.temporalNet = TemporalNet(size_in, c.size_hidden, c.gru_num_layers, c.num_labels)
        self.loss_fn = EngineTools.GetLoss()
        self.temporal_optimizer = EngineTools.GetOptimizer(self.temporalNet.parameters())
        self.GetInput = self.DefineGetInput(modality)
        self.t_loss = 'loss'
        self.AddLossType(self.t_loss)
        self.AddOptimizer(self.temporal_optimizer)

    def DefineGetInput(self, modality):
        def acoustic(dataPoint: Datapoint):
            return dataPoint.GetAcoustic()

        def visual(dataPoint: Datapoint):
            return dataPoint.GetVisual()

        if modality == 'visual':
            return visual
        return acoustic

    def Run(self, datapoint: Datapoint) -> ModelResults:
        temporal = self.temporalNet(self.GetInput(datapoint))
        return ModelResults(temporal)

    def ApplyLoss(self, results: ModelResults, datapoint: Datapoint):
        temporalLoss = self.loss_fn(results.result, datapoint.labels)
        self.AddLoss(self.t_loss, temporalLoss)

    @staticmethod
    def CreateModalAdapter(modality='acoustic'):
        if 'acoustic' in modality:
            return ModalAdapter(c.size_acoustic, 'acoustic')
        return ModalAdapter(c.size_visual, 'visual')
