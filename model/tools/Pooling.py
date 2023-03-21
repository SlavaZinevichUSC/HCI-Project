from core.Config import config
from core.Metadata import EntryMeta
import numpy as np
from skimage.measure import block_reduce
from model.tools.Datapoint import Datapoint
import torch


def Pool(poolType, poolNum):  # Fake Factory
    def No(vData):
        return vData

    def MeanPool(vData):
        data = block_reduce(vData, (poolNum, 1), np.mean)
        return data

    def MaxPool(vData):
        data = block_reduce(vData, (poolNum, 1), np.max)
        return data

    methods = {'none': No, 'mean': MeanPool, 'max': MaxPool}

    return methods.get(poolType, No)


def Standardize(data: EntryMeta, metaPath):
    def load(path):
        url = metaPath + path
        return np.load(url)

    def toTorch(item, modality): #Very ugly
        res = torch.from_numpy(item).double()
        #if config.modality != modality and config.modality != 'multi':
        #    res = torch.zeros_like(res)
        return res

    def LabelAsTensor(label):
        return torch.nn.functional.one_hot(torch.tensor([label]), num_classes=config.num_labels)[0, :].float()

    visual = toTorch(Pool(config.visual_pooling, config.pooling_num_vis)(load(data.visual)), 'visual')
    audio = toTorch(Pool(config.acoustic_pooling, config.pooling_num_acoustic)(load(data.acoustic)), 'acoustic')
    lexical = toTorch(load(data.lexical), 'lexical')
    label = LabelAsTensor(data.labels)
    return Datapoint(data.filename, data.speakers, visual, audio, lexical, label)
