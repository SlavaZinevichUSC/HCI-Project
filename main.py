from core.Metadata import Metadata
from model.tools import Storage
from model.Engine import Engine
from core.Config import config

meta = Metadata()
storage = Storage.SubjectIndependentStorage(meta, 'M01')
engine = Engine(storage)
"""engine.Run()
engine.EvaluateModel()
print('evaluate visual')
config.modality = 'visual'
engine.EvaluateModel()"""
config.Replace({'visual_pooling': 'max'})
engine.RunSearch([{'visual_pooling': 'max'}])
