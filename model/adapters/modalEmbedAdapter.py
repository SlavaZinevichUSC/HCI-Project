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


class ModalEmbedAdapter(AdapterBase):  # Ended up unnecessary as all networks have the same API
    def __init__(self, size_in, modality):
        super(ModalEmbedAdapter, self).__init__()
        size_embed_hidden = c.size_embed_hidden_visual if modality == 'visual' else c.size_embed_hidden_acoustic
        self.biasNet = BiasMitigationNet(size_in, size_embed_hidden, size_in)
        self.temporalNet = TemporalNet(size_in, c.size_hidden, c.gru_num_layers, c.num_labels)
        self.loss_fn = EngineTools.GetLoss()
        self.net = torch.nn.Sequential(self.biasNet, self.temporalNet)
        self.temporal_optimizer = EngineTools.GetOptimizer(self.net.parameters())
        self.t_loss = 'loss'
        self.GetInput = self.DefineGetInput(modality)
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
        # embed = self.biasNet(self.GetInput(datapoint))
        # temporal = self.temporalNet(embed)
        temporal = self.net(self.GetInput(datapoint))
        return ModelResults(temporal)

    def ApplyLoss(self, results: ModelResults, datapoint: Datapoint):
        temporalLoss = self.loss_fn(results.result, datapoint.labels)
        self.AddLoss(self.t_loss, temporalLoss)

    @staticmethod
    def CreateModalAdapter(modality='acoustic'):
        if 'acoustic' in modality:
            return ModalEmbedAdapter(c.size_acoustic, 'acoustic')
        return ModalEmbedAdapter(c.size_visual, 'visual')
