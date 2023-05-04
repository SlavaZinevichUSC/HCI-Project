from core.Config import config
from torch import argmax
from model.tools.Writer import WriteToFile
from model.tools.Datapoint import Datapoint


class BiasData:
    def __init__(self):
        self.counter = [0 for _ in range(config.num_labels)]

    def AddTensorResult(self, result):
        self.counter[argmax(result)] += 1


class BiasCollector:
    def __init__(self):
        self.data = []

    def AddResult(self, result, datapoint: Datapoint):
        self.data.append([int(argmax(result)), int(argmax(datapoint.labels)), datapoint.GetBiasLabelString()])

    def PrintResults(self):
        WriteToFile(self.data, config.adapter + '.json')
