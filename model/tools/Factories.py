from core.Config import config
from model.adapters.basicAdapter import BasicAdapter
from model.adapters.biasAdapter import BiasAdapter
from model.adapters.combineAdapter import CombineAdapter
from model.adapters.modalAdapter import ModalAdapter
from model.adapters.modalEmbedAdapter import ModalEmbedAdapter


def GetAdapter():
    adapterName = config.adapter

    def acousticBias():
        return BiasAdapter.CreateModalAdapter('acoustic')

    def visualBias():
        return BiasAdapter.CreateModalAdapter('visual')

    def acousticBasic():
        return ModalAdapter.CreateModalAdapter('acoustic')

    def visualBasic():
        return ModalAdapter.CreateModalAdapter('visual')

    def acousticEmbed():
        return ModalEmbedAdapter.CreateModalAdapter('acoustic')

    def visualEmbed():
        return ModalEmbedAdapter.CreateModalAdapter('visual')

    adapterSource = {'basic_multimodal': BasicAdapter,
                     'acoustic_bias': acousticBias,
                     'visual_bias': visualBias,
                     'basic_acoustic': acousticBasic,
                     'basic_visual': visualBasic,
                     'multi_basic': CombineAdapter,
                     'multi_bias': CombineAdapter,
                     'embed_acoustic': acousticEmbed,
                     'embed_visual': visualEmbed}
    if adapterName not in adapterSource.keys():
        print('WARNING: adapter name not found in adapters, returning basic')
        return BasicAdapter()
    adapter = adapterSource.get(adapterName, BasicAdapter)
    return adapter()
