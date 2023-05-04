import pandas as pd
from dataclasses import dataclass
from core.Config import config

@dataclass()
class IemoEntryMeta:
    filename: str
    speakers: str
    visual: str
    acoustic: str
    lexical: str
    labels: int

@dataclass()
class SewaEntryMeta:
    video: str
    audio: str
    good: int
    like: int
    excite: int
    positive: int
    negative: int
    ethnicity: str




class Metadata():
    def __init__(self, path='./data', filename='/dataset.csv'):
        self.path = path
        self.df = pd.read_csv(path + filename)
        self.numItems = len(self.df)
        # self.entries = self.df.progress_apply(lambda row: Entry(*row.to_list()), axis=1)
        self.CreateMetadata = self.RouteMetadata()

    def RouteMetadata(self):
        def iemo(args):
            return IemoEntryMeta(*args)
        def sewa(args):
            return SewaEntryMeta(*args[1:])
        if config.test_set == 'iemocap':
            return iemo
        return sewa

    def AsEntries(self, indexes) -> list[IemoEntryMeta]:
        return self.df.take(indexes).apply(lambda row: self.CreateMetadata(row.to_list()), axis=1).tolist()

    def GetAllEntries(self) -> list[IemoEntryMeta]:
        return self.df.apply(lambda row: self.CreateMetadata(row.to_list()), axis=1).tolist()
