from core.Config import config
from torch import argmax

labelNames = ['anger', 'sadness', 'happiness', 'neutral']  # suuuper lazy


class SingleClassConfusion:
    def __init__(self, label):
        self.label = label
        self.truePos = 0
        self.trueNeg = 0
        self.falsePos = 0
        self.falseNeg = 0

    def Add(self, label, prediction):
        if prediction == self.label:
            if label == self.label:
                self.truePos += 1
            else:
                self.falsePos += 1
        else:
            if label == self.label:
                self.falseNeg += 1
            else:
                self.trueNeg += 1

    def DisplayConfusion(self):
        print(f'Confusion matrix for label {self.label} aka {labelNames[self.label]}:')
        print(f'---------------------------------------------------')
        print('             actual positive | actual negative')
        print(f'---------------------------------------------------')
        print(f'pred positive |    {self.truePos}      |   {self.falsePos} ')
        print(f'---------------------------------------------------')
        print(f'pred negative |    {self.falseNeg}      |   {self.trueNeg} ')
        print('\n')


class ConfusionCollector():
    def __init__(self):
        self.labels = config.num_labels
        self.singles = [SingleClassConfusion(i) for i in range(self.labels)]

    def Add(self, datapoint, prediction):
        label = int(argmax(datapoint.labels))
        pred = int(argmax(prediction))
        for i in range(self.labels):
            self.singles[i].Add(label, pred)

    def DisplayConfusion(self):
        for single in self.singles:
            single.DisplayConfusion()
