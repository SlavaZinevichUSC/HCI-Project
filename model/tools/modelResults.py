from torch import Tensor, zeros
from dataclasses import dataclass

@dataclass
class ModelResults:
    def __init__(self, result: Tensor, advResult: [Tensor] = None):
        self.result: Tensor = result
        self.advResult: [Tensor] | None = advResult

    def FirstAdvResult(self):
        return self.advResult[0]

    @staticmethod
    def Empty():
        return ModelResults(zeros(0))
