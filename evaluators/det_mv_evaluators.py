import os
import os.path as osp
import tempfile
from typing import Sequence

import mmcv
import numpy as np
import pyquaternion
from nuscenes.utils.data_classes import Box
from perceptron.engine.callbacks import Callback
from perceptron.engine.executors import BaseExecutor
# TODO: Change to exp in BEVDepth
from perceptron.exps.base_exp import BaseExp
from pyquaternion import Quaternion
from tqdm import tqdm

from utils import torch_dist

from .evaluation_utils import get_evaluation_results

__all__ = ['DetMVNuscEvaluator']


class DetMVNuscEvaluator(BaseExecutor):
    ErrNameMapping = {
        'trans_err': 'mATE',
        'scale_err': 'mASE',
        'orient_err': 'mAOE',
        'vel_err': 'mAVE',
        'attr_err': 'mAAE',
    }

    DefaultAttribute = {
        'car': 'vehicle.parked',
        'pedestrian': 'pedestrian.moving',
        'trailer': 'vehicle.parked',
        'truck': 'vehicle.parked',
        'bus': 'vehicle.moving',
        'motorcycle': 'cycle.without_rider',
        'construction_vehicle': 'vehicle.parked',
        'bicycle': 'cycle.without_rider',
        'barrier': '',
        'traffic_cone': '',
    }

    def __init__(
        self,
        exp=BaseExp,
        callbacks=Sequence['Callback'],
        logger=None,
        eval_version='detection_cvpr_2019',
        data_root='./data/nuscenes',
        version='v1.0-trainval',
        modality=dict(use_lidar=False,
                      use_camera=True,
                      use_radar=False,
                      use_map=False,
                      use_external=False),
        output_dir=None,
    ) -> None:
        super(DetMVNuscEvaluator, self).__init__(exp, callbacks, logger)
        self.eval_version = eval_version
        self.data_root = data_root
        if self.eval_version is not None:
            from nuscenes.eval.detection.config import config_factory

            self.eval_detection_configs = config_factory(self.eval_version)
        self.version = version
        self.class_names = exp.class_names
        self.modality = modality
        self.mode = exp.mode
        assert self.mode in ['valid', 'test']
        output_dir = os.path.split(logger._core.handlers[1]._name)[0][1:]
        self.output_dir = output_dir

    @property
    def test_dataloader(self):
        return self.exp.test_dataloader

    def _evaluate_single(self,
                         result_path,
                         logger=None,
                         metric='bbox',
                         result_name='img_bbox'):
        """Evaluation for a single model in nuScenes protocol.

        Args:
            result_path (str): Path of the result file.
            logger (logging.Logger | str | None): Logger used for printing
                related information during evaluation. Default: None.
            metric (str): Metric name used for evaluation. Default: 'bbox'.
            result_name (str): Result name in the metric prefix.
                Default: 'img_bbox'.

        Returns:
            dict: Dictionary of evaluation details.
        """

        output_dir = osp.join(*osp.split(result_path)[:-1])
        nuscenes_meta_path = os.path.join(self.data_root,
                                          'nuscenes_v1.0-trainval_meta.pkl')
        self.meta_info = mmcv.load(nuscenes_meta_path)

        ap_dict = get_evaluation_results(
            nusc_meta_info=self.meta_info,
            result_path=result_path,
            output_dir=output_dir,
            eval_set='val',
            verbose=False,
            plot_examples=0,
            render_curves=False,
        )
        print(ap_dict)

    def format_results(self,
                       results,
                       img_metas,
                       result_names=['img_bbox'],
                       jsonfile_prefix=None,
                       **kwargs):
        """Format the results to json (standard format for COCO evaluation).

        Args:
            results (list[tuple | numpy.ndarray]): Testing results of the
                dataset.
            jsonfile_prefix (str | None): The prefix of json files. It includes
                the file path and the prefix of filename, e.g., "a/b/prefix".
                If not specified, a temp file will be created. Default: None.

        Returns:
            tuple: (result_files, tmp_dir), result_files is a dict containing \
                the json filepaths, tmp_dir is the temporal directory created \
                for saving json files when jsonfile_prefix is not specified.
        """
        assert isinstance(results, list), 'results must be a list'

        if jsonfile_prefix is None:
            tmp_dir = tempfile.TemporaryDirectory()
            jsonfile_prefix = osp.join(tmp_dir.name, 'results')
        else:
            tmp_dir = None

        # currently the output prediction results could be in two formats
        # 1. list of dict('boxes_3d': ..., 'scores_3d': ..., 'labels_3d': ...)
        # 2. list of dict('pts_bbox' or 'img_bbox':
        #     dict('boxes_3d': ..., 'scores_3d': ..., 'labels_3d': ...))
        # this is a workaround to enable evaluation of both formats on nuScenes
        # refer to https://github.com/open-mmlab/mmdetection3d/issues/449
        # should take the inner dict out of 'pts_bbox' or 'img_bbox' dict
        result_files = dict()
        # refactor this.
        for rasult_name in result_names:
            # not evaluate 2D predictions on nuScenes
            if '2d' in rasult_name:
                continue
            print(f'\nFormating bboxes of {rasult_name}')
            tmp_file_ = osp.join(jsonfile_prefix, rasult_name)
            if self.output_dir:
                result_files.update({
                    rasult_name:
                    self._format_bbox(results, img_metas, self.output_dir)
                })
            else:
                result_files.update({
                    rasult_name:
                    self._format_bbox(results, img_metas, tmp_file_)
                })
        return result_files, tmp_dir

    def evaluate(
        self,
        results,
        img_metas,
        metric='bbox',
        logger=None,
        jsonfile_prefix=None,
        result_names=['img_bbox'],
        show=False,
        out_dir=None,
        pipeline=None,
    ):
        """Evaluation in nuScenes protocol.

        Args:
            results (list[dict]): Testing results of the dataset.
            metric (str | list[str]): Metrics to be evaluated.
            logger (logging.Logger | str | None): Logger used for printing
                related information during evaluation. Default: None.
            jsonfile_prefix (str | None): The prefix of json files. It includes
                the file path and the prefix of filename, e.g., "a/b/prefix".
                If not specified, a temp file will be created. Default: None.
            show (bool): Whether to visualize.
                Default: False.
            out_dir (str): Path to save the visualization results.
                Default: None.
            pipeline (list[dict], optional): raw data loading for showing.
                Default: None.

        Returns:
            dict[str, float]: Results of each evaluation metric.
        """
        result_files, tmp_dir = self.format_results(results, img_metas,
                                                    result_names,
                                                    jsonfile_prefix)
        if self.mode == 'valid':
            if isinstance(result_files, dict):
                for name in result_names:
                    print('Evaluating bboxes of {}'.format(name))
                    self._evaluate_single(result_files[name])
            elif isinstance(result_files, str):
                self._evaluate_single(result_files)

        if tmp_dir is not None:
            tmp_dir.cleanup()

    def eval(self, sigma=1e-6):
        exp = self.exp
        if self.mode == 'valid':
            self.val_iter = iter(self.val_dataloader)
            dataloader = self.val_dataloader
        else:
            self.val_iter = iter(self.test_dataloader)
            dataloader = self.test_dataloader
        self._invoke_callback('before_eval')
        self.model.cuda()
        self.model.eval()
        all_results = list()
        all_img_metas = list()
        for i in tqdm(range(len(dataloader))):
            batch = next(self.val_iter)
            # results, gt_boxes, gt_labels = exp.test_step(data)
            img_metas, _, _ = batch[-3:]
            results = exp.test_step(batch)
            all_results.extend(results)
            all_img_metas.extend(img_metas)
        torch_dist.synchronize()
        all_results = sum(
            map(list, zip(*torch_dist.all_gather_object(all_results))),
            [])[:len(dataloader.dataset)]
        all_img_metas = sum(
            map(list, zip(*torch_dist.all_gather_object(all_img_metas))),
            [])[:len(dataloader.dataset)]
        if torch_dist.get_rank() == 0:
            self.evaluate(all_results, all_img_metas)

    def _format_bbox(self, results, img_metas, jsonfile_prefix=None):
        """Convert the results to the standard format.

        Args:
            results (list[dict]): Testing results of the dataset.
            jsonfile_prefix (str): The prefix of the output jsonfile.
                You can specify the output directory/filename by
                modifying the jsonfile_prefix. Default: None.

        Returns:
            str: Path of the output json file.
        """
        nusc_annos = {}
        mapped_class_names = self.class_names

        print('Start to convert detection format...')

        for sample_id, det in enumerate(mmcv.track_iter_progress(results)):
            boxes, scores, labels = det
            boxes = boxes
            sample_token = img_metas[sample_id]['token']
            trans = np.array(img_metas[sample_id]['ego2global_translation'])
            rot = Quaternion(img_metas[sample_id]['ego2global_rotation'])
            annos = list()
            for i, box in enumerate(boxes):
                name = mapped_class_names[labels[i]]
                center = box[:3]
                wlh = box[[4, 3, 5]]
                box_yaw = box[6]
                box_vel = box[7:].tolist()
                box_vel.append(0)
                quat = pyquaternion.Quaternion(axis=[0, 0, 1], radians=box_yaw)
                nusc_box = Box(center, wlh, quat, velocity=box_vel)
                nusc_box.rotate(rot)
                nusc_box.translate(trans)
                if np.sqrt(nusc_box.velocity[0]**2 +
                           nusc_box.velocity[1]**2) > 0.2:
                    if name in [
                            'car',
                            'construction_vehicle',
                            'bus',
                            'truck',
                            'trailer',
                    ]:
                        attr = 'vehicle.moving'
                    elif name in ['bicycle', 'motorcycle']:
                        attr = 'cycle.with_rider'
                    else:
                        attr = self.DefaultAttribute[name]
                else:
                    if name in ['pedestrian']:
                        attr = 'pedestrian.standing'
                    elif name in ['bus']:
                        attr = 'vehicle.stopped'
                    else:
                        attr = self.DefaultAttribute[name]
                nusc_anno = dict(
                    sample_token=sample_token,
                    translation=nusc_box.center.tolist(),
                    size=nusc_box.wlh.tolist(),
                    rotation=nusc_box.orientation.elements.tolist(),
                    velocity=nusc_box.velocity[:2],
                    detection_name=name,
                    detection_score=float(scores[i]),
                    attribute_name=attr,
                )
                annos.append(nusc_anno)
            # other views results of the same frame should be concatenated
            if sample_token in nusc_annos:
                nusc_annos[sample_token].extend(annos)
            else:
                nusc_annos[sample_token] = annos
        nusc_submissions = {
            'meta': self.modality,
            'results': nusc_annos,
        }
        mmcv.mkdir_or_exist(jsonfile_prefix)
        res_path = osp.join(jsonfile_prefix, 'results_nusc.json')
        print('Results writes to', res_path)
        mmcv.dump(nusc_submissions, res_path)
        return res_path
