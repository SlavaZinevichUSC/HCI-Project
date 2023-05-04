from dataclasses import dataclass
import pprint


@dataclass()  # set defaults!
class Config:
    visual_pooling: str = 'max'
    acoustic_pooling: str = 'mean'
    pooling_num_vis: int = 5
    pooling_num_acoustic: int = 5
    lossFn: str = 'ce'  # only cross entropy is available
    optimizer: str = 'adam'  # only one option because let's be honest adam is all that's used
    optimizer_lr: float = 0.005
    epochs: int = 800
    batch_size: int = 100
    size_visual: int = 2048  # can probably make these dynamic but who gives a damn
    size_lexical: int = 768
    size_acoustic: int = 128
    size_hidden: int = 128
    size_embed_hidden: int = 128
    num_debias_layers: int = 2
    gru_num_layers: int = 2
    dropout_prob: float = 0.15
    net_type: str = 'late'
    rolling_num: int = 3
    late_predictor: str = 'w_mean'  # options available at model.nets.LateFusionNet
    late_weighted_visual: float = 0.2
    late_weighted_acoustic: float = 0.3
    late_weighted_lexical: float = 0.5
    modality: str = 'multi'  # options are multi, visual, acoustic, lexical
    display_confusion: bool = False
    display_error: bool = True
    display_bias: bool = True
    train_display_interval: int = 10
    classes_disc: int = 2
    adapter: str = 'embed_acoustic'  # if multi don't forget to change 'modality'. its bad programming but oh well
    # modes: 'acoustic_bias','visual_bias','multi_bias'
    # 'basic_acoustic', 'basic_visual', 'multi_basic'
    # 'embed_acoustic', 'embed_visual', 'multi_embed'
    bias_weight: float = 0.025  # technically should normalize
    testing_bias: str = 'race'
    test_set: str = 'sewa'
    num_labels = 4 if test_set == 'iemocap' else 3 #Might be edited by SEWA label generator
    bias_print_method: str = 'json'


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
