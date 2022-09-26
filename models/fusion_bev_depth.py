from layers.backbones.fusion_lss_fpn import FusionLSSFPN
from layers.heads.bev_depth_head import BEVDepthHead

from .base_bev_depth import BaseBEVDepth

__all__ = ['FusionBEVDepth']


class FusionBEVDepth(BaseBEVDepth):
    """Source code of `BEVDepth`, `https://arxiv.org/abs/2112.11790`.

    Args:
        backbone_conf (dict): Config of backbone.
        head_conf (dict): Config of head.
        is_train_depth (bool): Whether to return depth.
            Default: False.
    """

    # TODO: Reduce grid_conf and data_aug_conf
    def __init__(self, backbone_conf, head_conf, is_train_depth=False):
        super(BaseBEVDepth, self).__init__()
        self.backbone = FusionLSSFPN(**backbone_conf)
        self.head = BEVDepthHead(**head_conf)
        self.is_train_depth = is_train_depth

    def forward(
        self,
        x,
        mats_dict,
        lidar_depth,
        timestamps=None,
    ):
        """Forward function for BEVDepth

        Args:
            x (Tensor): Input feature map.
            mats_dict(dict):
                sensor2ego_mats(Tensor): Transformation matrix from
                    camera to ego with shape of (B, num_sweeps,
                    num_cameras, 4, 4).
                intrin_mats(Tensor): Intrinsic matrix with shape
                    of (B, num_sweeps, num_cameras, 4, 4).
                ida_mats(Tensor): Transformation matrix for ida with
                    shape of (B, num_sweeps, num_cameras, 4, 4).
                sensor2sensor_mats(Tensor): Transformation matrix
                    from key frame camera to sweep frame camera with
                    shape of (B, num_sweeps, num_cameras, 4, 4).
                bda_mat(Tensor): Rotation matrix for bda with shape
                    of (B, 4, 4).
            timestamps (long): Timestamp.
                Default: None.

        Returns:
            tuple(list[dict]): Output results for tasks.
        """
        if self.is_train_depth and self.training:
            x = self.backbone(x, mats_dict, lidar_depth, timestamps)
            preds = self.head(x)
            return preds
        else:
            x = self.backbone(x, mats_dict, lidar_depth, timestamps)
            preds = self.head(x)
            return preds
