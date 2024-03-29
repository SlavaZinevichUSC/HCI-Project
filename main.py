from core.Metadata import Metadata
from model.tools import Storage
from model.Engine import Engine
from core.Config import config


def RunBias():
    meta = Metadata('./data/iemocap_embed')
    storage = Storage.LazyStorage(meta)
    engine = Engine(storage)
    engine.Run()


def RunBasic():
    meta = Metadata('./data/iemocap_embed')
    config.adapter = 'basic_multimodal'
    storage = Storage.LazyStorage(meta)
    engine = Engine(storage)
    engine.Run()


def RunTrial(delta, path='./data/iemocap_embed', filename='/dataset.csv'):
    meta = Metadata(path, filename)
    storage = Storage.ActiveStorage(meta)
    for d in delta:
        print('----------------- delta ----------------')
        print(d)
        config.Replace(d)
        engine = Engine(storage)
        engine.Run()
        engine.EvaluateModel()


# RunTrial(delta)
delta = [
    {'adapter': 'multi_basic'},
         {'adapter': 'multi_embed'},
         {'adapter': 'multi_bias'},
         ]
#RunTrial(delta, './data/sewa_t', '/key.csv')
RunTrial(delta)
