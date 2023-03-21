from dataclasses import dataclass
import pprint

@dataclass()  # set defaults!
class Config:
    visual_pooling: str = 'max'
    acoustic_pooling: str = 'mean'
    pooling_num_vis: int = 5
    pooling_num_acoustic: int = 2
    lossFn: str = 'ce' #only cross entropy is available
    optimizer: str = 'adam' #only one option because lets be honest adam is all thats used
    optimizer_lr: float = 0.001
    epochs: int = 20
    batch_size: int = 100
    size_visual: int = 2048 #can probably make these dynamic but who gives a damn
    size_lexical: int = 768
    size_acoustic: int = 128
    size_hidden: int = 128
    num_labels: int = 4
    gru_num_layers: int = 2
    net_type: str = 'late'
    rolling_num: int = 3
    late_predictor: str = 'w_mean' #options available at model.nets.LateFusionNet
    late_weighted_visual: float = 0.2
    late_weighted_acoustic: float = 0.3
    late_weighted_lexical: float = 0.5
    modality: str = 'multi' #options are multi, visual, acoustic, lexical
    display_confusion: bool = True
    display_error: bool = True

    def Display(self):
        pprint.pprint(self)

    @staticmethod
    def FromDict(d: dict):
        cnf = Config()
        cnf.Replace(d)
        return cnf

    def Replace(self, d: dict):
        for k, v in d.items():
            if hasattr(self, k):
                setattr(self, k, v)



config = Config.FromDict({})