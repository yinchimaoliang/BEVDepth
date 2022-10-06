# Copyright (c) Megvii Inc. All rights reserved.
from exps.base_cli import run_cli
from exps.fusion.bev_depth_fusion_lss_r50_256x704_128x128_24e_2key import \
    BEVDepthLightningModel as BaseBEVDepthLightningModel
from exps.mv.bev_depth_lss_r50_256x704_128x128_20e_cbgs_2key_da import \
    DepthAggregation
from layers.backbones.fusion_lss_fpn import FusionLSSFPN as BaseFusionLSSFPN
from models.fusion_bev_depth import FusionBEVDepth as BaseFusionBEVDepth


class LSSFPN(BaseFusionLSSFPN):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.depth_aggregation_net = self._configure_depth_aggregation_net()

    def _configure_depth_aggregation_net(self):
        """build pixel cloud feature extractor"""
        return DepthAggregation(self.output_channels, self.output_channels,
                                self.output_channels)

    def _forward_voxel_net(self, img_feat_with_depth):
        # BEVConv2D [n, c, d, h, w] -> [n, h, c, w, d]
        img_feat_with_depth = img_feat_with_depth.permute(
            0, 3, 1, 4, 2).contiguous()  # [n, c, d, h, w] -> [n, h, c, w, d]
        n, h, c, w, d = img_feat_with_depth.shape
        img_feat_with_depth = img_feat_with_depth.view(-1, c, w, d)
        img_feat_with_depth = (
            self.depth_aggregation_net(img_feat_with_depth).view(
                n, h, c, w, d).permute(0, 2, 4, 1, 3).contiguous().float())
        return img_feat_with_depth


class FusionBEVDepth(BaseFusionBEVDepth):
    def __init__(self, backbone_conf, head_conf, is_train_depth=True):
        super(BaseFusionBEVDepth, self).__init__(backbone_conf, head_conf,
                                                 is_train_depth)
        self.backbone = LSSFPN(**backbone_conf)


class BEVDepthLightningModel(BaseBEVDepthLightningModel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.model = FusionBEVDepth(self.backbone_conf, self.head_conf)


if __name__ == '__main__':
    run_cli(BEVDepthLightningModel,
            'bev_depth_fusion_lss_r50_256x704_128x128_24e_2key_da')
