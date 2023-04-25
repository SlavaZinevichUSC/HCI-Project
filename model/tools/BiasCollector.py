from core.Config import config
from torch import argmax


class BiasData:
    def __init__(self):
        self.counter = [0 for _ in range(config.num_labels)]

    def AddTensorResult(self, result):
        self.counter[argmax(result)] += 1


class BiasCollector:
    def __init__(self):
        self.biases = {}
        self.InitializeBiases()

    def InitializeBiases(self):
        if config.testing_bias == 'gender':
            self.biases['M'] = BiasData()
            self.biases['F'] = BiasData()

    def AddResult(self, result, bias):
        self.biases[bias].AddTensorResult(result)

    def PrintResults(self):
        if not config.display_bias:
            return
        print(f'TESTED: {config.test_set} on mode: {config.adapter}')
        for k,v in self.biases.items():
            print(f'bias name {k}: {v.counter}')

