# Copyright (c) Megvii Inc. All rights reserved.
from exps.base_cli import run_cli
from exps.fusion.bev_depth_fusion_lss_r50_256x704_128x128_24e_2key import \
    BEVDepthLightningModel as BaseBEVDepthLightningModel
from models.fusion_bev_depth import FusionBEVDepth


class BEVDepthLightningModel(BaseBEVDepthLightningModel):
    def __init__(self, *args, **kwargs) -> None:
        super(BaseBEVDepthLightningModel, self).__init__(*args, **kwargs)
        scale = 2
        self.backbone_conf.update({
            'img_backbone_conf':
            dict(
                type='ResNet',
                depth=101,
                frozen_stages=0,
                out_indices=[0, 1, 2, 3],
                norm_eval=False,
                init_cfg=dict(type='Pretrained',
                              checkpoint='torchvision://resnet101'),
            ),
            'img_neck_conf':
            dict(
                type='SECONDFPN',
                in_channels=[256, 512, 1024, 2048],
                upsample_strides=[0.25, 0.5, 1, 2],
                out_channels=[128, 128, 128, 128],
            ),
        })
        self.backbone_conf.update({'final_dim': (256 * scale, 704 * scale)})
        self.ida_aug_conf['resize_lim'] = [0.386 * scale, 0.55 * scale]
        self.ida_aug_conf['final_dim'] = [512, 1408]
        self.backbone_conf.update({'x_bound': [-51.2, 51.2, 0.4]})
        self.backbone_conf.update({'y_bound': [-51.2, 51.2, 0.4]})
        self.head_conf['bbox_coder'].update({'out_size_factor': 2})
        self.head_conf['train_cfg'].update({'out_size_factor': 2})
        self.head_conf['test_cfg'].update({'out_size_factor': 2})
        self.model = FusionBEVDepth(self.backbone_conf, self.head_conf)


if __name__ == '__main__':
    run_cli(BEVDepthLightningModel,
            'bev_depth_fusion_lss_r101_512x1408_256x256_24e')
