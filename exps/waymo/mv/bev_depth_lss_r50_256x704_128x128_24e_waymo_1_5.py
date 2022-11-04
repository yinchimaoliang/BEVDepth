# Copyright (c) Megvii Inc. All rights reserved.

from exps.base_cli import run_cli
from exps.waymo.base_exp import \
    BEVDepthLightningModel as BaseBEVDepthLightningModel


class BEVDepthLightningModel(BaseBEVDepthLightningModel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.train_info_paths = 'data/waymo/v1.4/waymo_infos_training_1_5.pkl'


if __name__ == '__main__':
    run_cli(BEVDepthLightningModel,
            'bev_depth_lss_r50_256x704_128x128_24e_waymo_1_5')
