from dataclasses import dataclass
import torch
from core.Config import config

femaleTensor = torch.FloatTensor([1, 0])
maleTensor = torch.FloatTensor([0,1])


@dataclass()
class Datapoint:
    name: str
    speakers: str
    visual: torch.Tensor
    acoustic: torch.Tensor
    lexical: torch.Tensor
    labels: int
    gender: torch.Tensor | None = None

    # Hideous solution to avoid reloading and instead dynamically adujsting to modality
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

    def Gender(self) -> torch.Tensor:
        if self.gender is None:
            self.gender = femaleTensor if 'F' in self.speakers else maleTensor
        return self.gender

    def GenderAsString(self) -> str:
        return 'F' if 'F' in self.speakers else 'M'

    def GenderLike(self, t: torch.Tensor) -> torch.Tensor:
        gender = self.Gender()
        return gender.unsqueeze(0).repeat(t.size()[0],1)
