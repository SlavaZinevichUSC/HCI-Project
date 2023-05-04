from dataclasses import dataclass
import torch
from core.Config import config

femaleTensor = torch.FloatTensor([1, 0])
maleTensor = torch.FloatTensor([0, 1])
races = {'british': torch.FloatTensor([1, 0, 0, 0, 0, 0]),
         'chinese': torch.FloatTensor([0, 1, 0, 0, 0, 0]),
         'german': torch.FloatTensor([0, 0, 1, 0, 0, 0]),
         'greek': torch.FloatTensor([0, 0, 0, 1, 0, 0]),
         'hungarian': torch.FloatTensor([0, 0, 0, 0, 1, 0]),
         'serbian': torch.FloatTensor([0, 0, 0, 0, 0, 1])}


@dataclass()
class Datapoint:
    name: str
    speakers: str
    visual: torch.Tensor
    acoustic: torch.Tensor
    lexical: torch.Tensor
    labels: torch.Tensor
    gender: torch.Tensor | None = None
    race: torch.Tensor | None = None

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

    def GetBiasLabel(self):
        if config.test_set == 'iemocap':
            return self.Gender()
        return self.GetRace()

    def GetBiasLabelString(self) -> str:
        if config.test_set == 'iemocap':
            return self.GenderAsString()
        return self.GetRaceAsString()

    def BiasLike(self, t: torch.Tensor):
        if config.test_set == 'iemocap':
            return self.GenderLike(t)
        return

    def GetRace(self):
        if self.race is not None:
            return self.race
        self.race = races[self.speakers]  # based on number of sewa races

    def GetRaceAsString(self):
        return self.speakers

    def GetRaceLike(self, t: torch.Tensor):
        return self.GetRace().unsqueeze(0).repeat(t.size()[0], 1)

    def Gender(self) -> torch.Tensor:
        if self.gender is None:
            self.gender = femaleTensor if 'F' in self.speakers else maleTensor
        return self.gender

    def GenderAsString(self) -> str:
        return 'F' if 'F' in self.speakers else 'M'

    def GenderLike(self, t: torch.Tensor) -> torch.Tensor:
        gender = self.Gender()
        return gender.unsqueeze(0).repeat(t.size()[0], 1)
