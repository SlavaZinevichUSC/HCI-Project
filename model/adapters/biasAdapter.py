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


class BiasAdapter(AdapterBase):  # Ended up unnecessary as all networks have the same API
    def __init__(self, size_in, modality: str | None = None):
        super(BiasAdapter, self).__init__()
        self.biasNet = BiasMitigationNet(size_in, c.size_embed_hidden, size_in)
        self.advNet = DiscriminatorNet(size_in, c.size_hidden, c.classes_disc)
        self.temporalNet = TemporalNet(size_in, c.size_hidden, c.gru_num_layers, c.num_labels)
        self.loss_fn = EngineTools.GetLoss()
        #label_parameters = [self.biasNet.parameters(), self.temporalNet.parameters()]
        self.temporal_optimizer = EngineTools.GetOptimizer(self.temporalNet.parameters())
        self.adv_optimizer = EngineTools.GetOptimizer(self.advNet.parameters())
        self.bias_optimizer = EngineTools.GetOptimizer(self.biasNet.parameters())
        self.GetInput = self.DefineGetInput(modality)
        self.t_loss = 'temporalLoss'
        self.a_loss = 'advLoss'
        self.b_loss = 'biasLoss'
        self.AddMultiLossType([self.t_loss, self.a_loss, self.b_loss])

    def DefineGetInput(self, modality):
        def acoustic(dataPoint: Datapoint):
            return dataPoint.GetAcoustic()

        def visual(dataPoint: Datapoint):
            return dataPoint.GetVisual()

        if modality == 'visual':
            return visual
        return acoustic

    def Run(self, datapoint: Datapoint) -> ModelResults:
        embed = self.biasNet(self.GetInput(datapoint))
        temporal = self.temporalNet(embed)
        adv = self.advNet(embed.detach())
        return ModelResults(temporal, [adv])

    def ApplyLoss(self, results: ModelResults, datapoint: Datapoint):
        loss = self.GetLoss(results, datapoint)
        self.AddLoss(self.t_loss, loss[0])
        self.AddLoss(self.b_loss, loss[1])
        self.AddLoss(self.a_loss, loss[2])

    def GetLoss(self, results: ModelResults, datapoint: Datapoint):
        temporalLoss = self.loss_fn(results.result, datapoint.labels)
        advResult = results.FirstAdvResult()
        biasVar = datapoint.GenderLike(advResult)
        advLoss = self.loss_fn(advResult, biasVar)
        biasLoss = temporalLoss - c.bias_weight * advLoss
        return [temporalLoss, biasLoss, advLoss]

    def StepOptimizer(self):
        self.adv_optimizer.step()
        self.adv_optimizer.zero_grad()
        self.temporal_optimizer.step()
        self.temporal_optimizer.zero_grad()
        self.bias_optimizer.step()
        self.bias_optimizer.zero_grad()

    @staticmethod
    def CreateModalAdapter(modality='acoustic'):
        if 'acoustic' in modality:
            return BiasAdapter(c.size_acoustic, 'acoustic')
        return BiasAdapter(c.size_visual, 'visual')
