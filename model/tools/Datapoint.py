from dataclasses import dataclass
import torch
from core.Config import config

@dataclass()
class Datapoint:
    name: str
    speakers: str
    visual: torch.Tensor
    acoustic: torch.Tensor
    lexical: torch.Tensor
    labels: int

    #Extremely ugly solution to avoid reloading and instead dynamically adujsting to modality
    def GetVisual(self):
        if config.modality != 'visual' and config.modality != 'multi':
            return torch.zeros_like(self.visual)
        return self.visual

    def GetAcoustic(self):
        if config.modality != 'acoustic' and config.modality != 'multi':
            return torch.zeros_like(self.acoustic)
        return self.acoustic

    def GetLexical(self):
        if config.modality != 'lexical' and config.modality != 'multi':
            return torch.zeros_like(self.lexical)
        return self.lexical
