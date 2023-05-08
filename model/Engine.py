from model.adapters.adapterBase import AdapterBase
from model.tools.BiasCollector import BiasCollector
from model.tools.Storage import Storage
from model.tools.Datapoint import Datapoint
from core.Config import config
from model.tools import EngineTools, ErrorCollector, ConfusionCollector
from model.adapters.basicAdapter import BasicAdapter
from model.tools.Factories import GetAdapter

class Engine:
    def __init__(self, storage: Storage):
        self.storage = storage
        self.modelAdapter: AdapterBase = GetAdapter()
        self.epochs = config.epochs


    def Run(self):
        errorCollector = ErrorCollector.ErrorCollector()
        for j in range(10):
            for i in range(self.epochs):
                batch: list[Datapoint] = self.storage.GetRandomBatch(config.batch_size)
                for datapoint in batch:
                    out = self.modelAdapter.Run(datapoint)
                    self.modelAdapter.ApplyLoss(out, datapoint)
                    errorCollector.AddError(out, datapoint.labels)
                self.modelAdapter.BatchApplyLoss()
                errorCollector.Archive()
                if config.display_error and i + j*50 % config.train_display_interval == 0:
                    errorCollector.DisplayCurrentError(i+ j*50)
        errorCollector.DisplayCurrentError(self.epochs)
        #errorCollector.DisplayErrorGraph()

    def EvaluateModel(self):
        datapoints = self.storage.GetAll()
        errorCollector = ErrorCollector.ErrorCollector()
        confusion = ConfusionCollector.ConfusionCollector()
        biasCollector = BiasCollector()
        for datapoint in datapoints:
            out = self.modelAdapter.Run(datapoint)
            errorCollector.AddError(out, datapoint.labels)
            confusion.Add(datapoint, out.result)
            biasCollector.AddResult(out.result, datapoint)
        errorCollector.Archive()
        biasCollector.PrintResults()
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
            self.modelAdapter = GetAdapter()
            self.Run()
            error = self.EvaluateModel()
            print(f'error for options: {option} is: {error}')
            if error < bestError:
                bestError = error
                bestOption = option
                print(f'new best option found!')
        print(f'best option found at option: {bestOption} with error: {bestError}')
        return bestOption
