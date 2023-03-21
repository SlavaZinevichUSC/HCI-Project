from model.tools.Storage import Storage
from model.tools.Datapoint import Datapoint
from core.Config import config
from model.tools import EngineTools, ErrorCollector, ConfusionCollector
from model.Adapter import Adapter


class Engine:
    def __init__(self, storage: Storage):
        self.storage = storage
        self.loss_fn = EngineTools.GetLoss()
        self.model: Adapter = Adapter()
        self.optimizer = EngineTools.GetOptimizer(self.model.GetParameters())
        self.epochs = config.epochs

    def Run(self):
        errorCollector = ErrorCollector.ErrorCollector()
        for i in range(self.epochs):
            batch: list[Datapoint] = self.storage.GetRandomBatch(config.batch_size)

            for datapoint in batch:
                self.optimizer.zero_grad()
                out = self.model.Run(datapoint)
                loss = self.loss_fn(out, datapoint.labels)
                loss.backward()
                self.optimizer.step()
                errorCollector.AddError(out, datapoint.labels)
            errorCollector.Archive()
            if config.display_error and i % 5 == 0:
                errorCollector.DisplayCurrentError(i)

    def EvaluateModel(self):
        datapoints = self.storage.GetAll()
        errorCollector = ErrorCollector.ErrorCollector()
        confusion = ConfusionCollector.ConfusionCollector()
        for datapoint in datapoints:
            out = self.model.Run(datapoint)
            errorCollector.AddError(out, datapoint.labels)
            confusion.Add(datapoint, out)
        errorCollector.Archive()
        if config.display_error:
            errorCollector.DisplayCurrentError()
        if config.display_confusion:
            confusion.DisplayConfusion()
        return errorCollector.GetLastError()

    def RunSearch(self, options):
        bestError = 1.42
        bestOption = {}
        for option in options:
            config.Replace(option)
            self.model = Adapter()
            self.optimizer = EngineTools.GetOptimizer(self.model.GetParameters())
            self.Run()
            error = self.EvaluateModel()
            print(f'error for options: {option} is: {error}')
            if error < bestError:
                bestError = error
                bestOption = option
                print(f'new best option found!')
        print(f'best option found at option: {bestOption} with error: {bestError}')
        return bestOption



