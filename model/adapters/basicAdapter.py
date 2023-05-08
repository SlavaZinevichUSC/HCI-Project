from model.tools import EngineTools
from model.tools.Datapoint import Datapoint
from model.nets.EarlyFusionNet import EarlyFusionNet
from model.nets.LateFusionNet import LateFusionNet
from core.Config import config as c
from model.tools.modelResults import ModelResults


class BasicAdapter:  # Ended up unnecessary as all networks have the same API
    def __init__(self):
        self.model = self.GetNet()
        self.loss_fn = EngineTools.GetLoss()
        self.optimizer = EngineTools.GetOptimizer(self.model.parameters())
        self
        pass

    def Run(self, datapoint: Datapoint) -> ModelResults:
        return ModelResults(self.model(datapoint))

    def ApplyLoss(self, results: ModelResults, datapoint: Datapoint):
        loss = self.loss_fn(results.result, datapoint.labels)
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()

    def GetNet(self):
        net = EarlyFusionNet
        if c.net_type == 'early':
            net = EarlyFusionNet
        if c.net_type == 'late':
            net = LateFusionNet
        return net(c.size_hidden, c.num_labels, c.gru_num_layers, c.size_visual, c.size_acoustic,
                   c.size_lexical)
