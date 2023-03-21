import pandas as pd
from dataclasses import dataclass


@dataclass()
class EntryMeta:
    filename : str
    speakers: str
    visual: str
    acoustic : str
    lexical: str
    labels: int




class Metadata():
    def __init__(self, path='./data', filename='/dataset.csv'):
        self.path = path
        self.df = pd.read_csv(path + filename)
        self.numItems = len(self.df)
        # self.entries = self.df.progress_apply(lambda row: Entry(*row.to_list()), axis=1)

    def AsEntries(self, indexes) -> list[EntryMeta]:
        return self.df.take(indexes).apply(lambda row: EntryMeta(*row.to_list()), axis=1).tolist()

    def GetAllEntries(self) -> list[EntryMeta]:
        return self.df.apply(lambda row: EntryMeta(*row.to_list()), axis=1).tolist()

