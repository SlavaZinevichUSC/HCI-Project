from model.tools.Datapoint import Datapoint
from model.tools.modelResults import ModelResults


class AdapterBase():
    def __init__(self):
        pass

    def Run(self, datapoint: Datapoint) -> ModelResults:
        return ModelResults.Empty()

    def ApplyLoss(self, results : ModelResults, datapoint: Datapoint):
        pass

