import torch
from core.Config import config
from sklearn.metrics import f1_score


class ErrorCollector:
    def __init__(self):
        self.currentError = 0.0
        self.count = 0.0
        self.errors = []
        self.f1scores = []
        self.rollingNum = config.rolling_num
        self.singleRunPred = []
        self.singleRunLabel = []

    def AddError(self, pred, label):
        predLabel, actualLabel = torch.argmax(pred), torch.argmax(label)
        self.currentError += 1 if predLabel != actualLabel else 0
        self.count += 1
        self.singleRunPred.append(predLabel)
        self.singleRunLabel.append(actualLabel)

    def GetError(self):
        return self.currentError / self.count

    def Archive(self):
        err = self.GetError()
        self.errors.append(err)
        self.currentError = 0.0
        self.count = 0.0
        self.f1scores.append(f1_score(self.singleRunLabel, self.singleRunPred, average=None))
        self.singleRunPred = []
        self.singleRunLabel = []
        return err

    def GetRollingError(self):
        if (len(self.errors)) < self.rollingNum:
            return 0
        return sum(self.errors[-self.rollingNum-1:-1]) / self.rollingNum

    def DisplayCurrentError(self, epoch=-1):
        if len(self.errors) < 1:
            print(f'There Are No Errors Recorded!')
            return
        print(f'Errors for epoch {epoch}: ')
        print(f'Single run error: {self.GetLastError()}')
        print(f'Rolling Error for {self.rollingNum} epochs: {self.GetRollingError()}')
        print(f'f1 score: {self.f1scores[-1]}')
        print()

    def GetLastError(self):
        if (len(self.errors) > 0):
            return self.errors[-1]
        return 0.0
