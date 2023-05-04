from itertools import cycle


class LossScheduler:
    def __init__(self, choices):
        self.variations = cycle(choices)
        self.last = None
        pass

    def GetNext(self):
        self.last = next(self.variations)
        return self.last

    def SeeLast(self):
        return self.last

