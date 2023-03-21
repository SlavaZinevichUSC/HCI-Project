from torch import nn
from model.tools.Storage import Datapoint
import torch
from core.Config import config

torch.set_default_dtype(torch.float64)


class EarlyFusionNet(nn.Module):
    def __init__(self, hidden_dim, output_dim, n_layers, input_dim_visual, input_dim_audio, input_dim_lex,
                 drop_prob=0.2):
        super(EarlyFusionNet, self).__init__()
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.h0 = None

        self.visualGRU = nn.GRU(input_dim_visual, hidden_dim, n_layers, batch_first=True, dropout=drop_prob)
        self.audioGRU = nn.GRU(input_dim_audio, hidden_dim, n_layers, batch_first=True, dropout=drop_prob)
        self.lexical = nn.Sequential(nn.Linear(input_dim_lex, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim))

        self.fc = nn.Sequential(nn.Linear(3 * hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, output_dim))

    def forward(self, datapoint: Datapoint):
        if self.h0 is None:
            self.h0 = self.init_hidden()
        gruVisual, h = self.visualGRU(datapoint.GetVisual(), self.h0)
        outVisual = h[-1, :]

        gruAudio, h = self.audioGRU(datapoint.GetAcoustic(), self.h0)
        outAudio = h[-1, :]

        outLexical = self.lexical(datapoint.GetLexical())

        def softmax(n):
            return nn.functional.softmax(n, dim=0)

        cat = torch.cat((outVisual, outAudio, outLexical))
        out = self.fc(cat)

        return softmax(out).float()

    def init_hidden(self):
        weight = next(self.parameters()).data
        hidden = weight.new(self.n_layers, self.hidden_dim).zero_()
        return hidden

    def GetLabelPredictor(self):
        def w_mean(v, a, l):
            return 0.2 * v + 0.3 * a + 0.5 * l

        def mean(v, a, l):
            return v + a + l / 3.0

        def max(v, a, l):
            return torch.maximum(torch.maximum(v, a), l)

        funcs = {'w_mean': w_mean, 'mean': mean, 'max': max}
        return funcs.get(config.early_predictor, mean)
