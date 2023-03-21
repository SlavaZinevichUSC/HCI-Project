from torch import nn
from model.tools.Storage import Datapoint
import torch
from core.Config import config

torch.set_default_dtype(torch.float64)


class LateFusionNet(nn.Module):
    def __init__(self, hidden_dim, output_dim, n_layers, input_dim_visual, input_dim_audio, input_dim_lex,
                 drop_prob=0.2):
        super(LateFusionNet, self).__init__()
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.predictor = self.GetLabelPredictor()

        self.visualGRU = nn.GRU(input_dim_visual, hidden_dim, n_layers, dropout=drop_prob)
        self.audioGRU = nn.GRU(input_dim_audio, hidden_dim, n_layers, dropout=drop_prob)
        self.lexical = nn.Sequential(nn.Linear(input_dim_lex, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, output_dim))

        self.out_visual = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, output_dim))
        self.out_audio = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, output_dim))
        self.conv = nn.Linear(output_dim * 3, output_dim)

    def forward(self, datapoint: Datapoint):
        hidden = self.init_hidden()
        gruVisual, h = self.visualGRU(datapoint.GetVisual(), hidden)
        outVisual = self.out_visual(gruVisual[-1, :])
        gruAudio, h = self.audioGRU(datapoint.GetAcoustic(), hidden)
        outAudio = self.out_audio(gruAudio[-1, :])
        outLexical = self.lexical(datapoint.GetLexical())

        def softmax(n):
            return nn.functional.softmax(n, dim=0)

        outVisual = softmax(outVisual)
        outAudio = softmax(outAudio)
        outLexical = softmax(outLexical)

        out = self.predictor(outVisual, outAudio, outLexical)
        return out.float()

    def init_hidden(self):
        weight = next(self.parameters()).data
        hidden = weight.new(self.n_layers, self.hidden_dim).zero_()
        return hidden

    def GetLabelPredictor(self):
        def w_mean(v, a, l):
            return config.late_weighted_visual * v + config.late_weighted_acoustic * a + config.late_weighted_lexical * l


        def mean(v, a, l):
            return v + a + l / 3.0

        def max(v, a, l):
            return torch.maximum(torch.maximum(v, a), l)

        def visual(v, a, l):
            return v

        def audio(v, a, l):
            return a

        def lexical(v, a, l):
            return l

        funcs = {'w_mean': w_mean, 'mean': mean, 'max': max, 'visual': visual, 'acoustic': audio, 'lexical': lexical}
        return funcs.get(config.late_predictor, mean)
