from model.adapters.biasAdapter import BiasAdapter
from model.adapters.combineAdapter import CombineAdapter
from model.tools.Storage import Storage
from model.tools.Datapoint import Datapoint
from core.Config import config
from model.tools import EngineTools, ErrorCollector, ConfusionCollector
from model.adapters.basicAdapter import BasicAdapter


class Engine:
    def __init__(self, storage: Storage):
        self.storage = storage
        self.modelAdapter: BasicAdapter = self.GetAdapter()
        self.epochs = config.epochs

    def GetAdapter(self):
        adapterName = config.adapter

        def acousticBias():
            return BiasAdapter.CreateModalAdapter('acoustic')

        def visualBias():
            return BiasAdapter.CreateModalAdapter('visual')

        adapterSource = {'basic_multimodal': BasicAdapter,
                         'acoustic_bias': acousticBias,
                         'visual_bias': visualBias,
                         'multi_bias': CombineAdapter}
        if adapterName not in adapterSource.keys():
            print('WARNING: adapter name not found in adapters, returning basic')
            return BasicAdapter()
        adapter = adapterSource.get(adapterName, BasicAdapter)
        return adapter()

    def Run(self):
        errorCollector = ErrorCollector.ErrorCollector()
        for i in range(self.epochs):
            batch: list[Datapoint] = self.storage.GetRandomBatch(config.batch_size)

            for datapoint in batch:
                out = self.modelAdapter.Run(datapoint)
                self.modelAdapter.ApplyLoss(out, datapoint)
                errorCollector.AddError(out, datapoint.labels)
            errorCollector.Archive()
            if config.display_error and i % 5 == 0:
                errorCollector.DisplayCurrentError(i)

    def EvaluateModel(self):
        datapoints = self.storage.GetAll()
        errorCollector = ErrorCollector.ErrorCollector()
        confusion = ConfusionCollector.ConfusionCollector()
        for datapoint in datapoints:
            out = self.modelAdapter.Run(datapoint)
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
            self.modelAdapter = BasicAdapter()
            self.optimizer = EngineTools.GetOptimizer(self.modelAdapter.GetParameters())
            self.Run()
            error = self.EvaluateModel()
            print(f'error for options: {option} is: {error}')
            if error < bestError:
                bestError = error
                bestOption = option
                print(f'new best option found!')
        print(f'best option found at option: {bestOption} with error: {bestError}')
        return bestOption
