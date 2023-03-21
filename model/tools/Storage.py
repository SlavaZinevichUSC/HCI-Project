import numpy as np
from core.Metadata import Metadata
from model.tools.Pooling import Standardize
from model.tools.Datapoint import Datapoint


class Storage:
    def __init__(self, metadata: Metadata):
        self.metadata: Metadata = metadata
        pass

    def Get(self, indexes):
        return []

    def GetRandomBatch(self, size) -> list[Datapoint]:
        indexes = np.random.randint(0, self.metadata.numItems, size=size)
        return self.Get(indexes)

    def GetAll(self):
        return self.Get(range(self.metadata.numItems))

    def Reset(self):
        pass


# Didn't know how big the data will be when loaded
# A more sophisticated system would involve cache based on RAM size but who cares
class TinyStorage(Storage):
    def __init__(self, metadata: Metadata):
        Storage.__init__(metadata)

    def Get(self, indexes):
        entries = self.metadata.AsEntries(indexes)
        datapoints = [Standardize(e, self.metadata.path) for e in entries]

        return datapoints


# This just helps quick retesting so all the data isn't loaded at once
class LazyStorage(Storage):
    def __init__(self, metadata: Metadata):
        Storage.__init__(self, metadata)
        self.store = {}

    def Get(self, indexes):  # UGLY
        datapoints = [self.Create(i) if i not in self.store.keys() else self.store[i] for i in indexes]
        return datapoints

    def Create(self, idx):
        entry = self.metadata.AsEntries([idx])
        p = Standardize(entry[0], self.metadata.path)
        self.store[idx] = p
        return p

    def Reset(self):
        self.store = {}


# Same as lazy but all at once
class ActiveStorage(Storage):
    def __init__(self, metadata: Metadata):
        Storage.__init__(self, metadata)
        self.store = {i: Standardize(e, metadata.path) for (i, e) in enumerate(metadata.GetAllEntries())}

    def Get(self, indexes):  # UGLY
        data = [self.store[i] for i in indexes]
        return data

    def Reset(self):
        self.store = {i: Standardize(e, self.metadata.path) for (i, e) in enumerate(self.metadata.GetAllEntries())}


class SubjectIndependentStorage(Storage):
    def __init__(self, metadata: Metadata, subjectName):
        Storage.__init__(self, metadata)
        self.store = {i: Standardize(e, metadata.path) for (i, e) in enumerate(metadata.GetAllEntries()) if
                      e.speakers != subjectName}
        self.storeSubject = {i: Standardize(e, metadata.path) for (i, e) in enumerate(metadata.GetAllEntries()) if
                             e.speakers == subjectName}
        self.exclude = True

    def Get(self, indexes):  # UGLY
        store = self.store if self.exclude else self.storeSubject
        data = [store[i] for i in indexes if i in store.keys()]
        if len(data) == 0:
            return self.GetRandomBatch(len(indexes))  # hack
        while len(data) < len(indexes):  # hack
            data += data
        return data

    def SetExclude(self, toExclude=True):
        self.exclude = toExclude

    def Reset(self):
        self.store = {i: Standardize(e, self.metadata.path) for (i, e) in enumerate(self.metadata.GetAllEntries())}
