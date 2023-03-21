from model.tools.Datapoint import Datapoint
from model.nets.EarlyFusionNet import EarlyFusionNet
from model.nets.LateFusionNet import LateFusionNet
from core.Config import config as c


class Adapter: #Ended up unnecessary as all networks have the same API
    def __init__(self):
        self.model = self.GetNet()
        pass

    def Run(self, datapoint: Datapoint) -> int:
        return self.model(datapoint)

    def GetParameters(self):
        return self.model.parameters()

    def GetNet(self):
        net = EarlyFusionNet
        if c.net_type == 'early':
            net = EarlyFusionNet
        if c.net_type == 'late':
            net = LateFusionNet
        return net(c.size_hidden, c.num_labels, c.gru_num_layers, c.size_visual, c.size_acoustic,
                             c.size_lexical)
