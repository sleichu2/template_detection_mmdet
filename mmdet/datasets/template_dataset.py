import mmcv
import numpy as np

from .custom import CustomDataset
from .registry import DATASETS


@DATASETS.register_module
class TemplateDataset(CustomDataset):
    """Custom dataset for detection.

    Annotation format:
    [
        {
            'template_img': 'a.jpg',
            'fact_img': 'b.jpg',
            'width': 1280,
            'height': 720,
            'bboxes': <list> [[x1, y1, w, h], ...]
            'labels': <list> [1, ...]
        },
        ...
    ]
    """

    def load_annotations(self, ann_file):
        img_infos = mmcv.load(ann_file)
        self.img_ids = np.arange(0, len(img_infos), dtype=np.uint8)
        return img_infos

    def _filter_imgs(self):
        """Assume all your images have ground truths.
        You can modify the code here to check all imgs before training
        """
        valid_inds = []
        for i, img_info in enumerate(self.img_infos):
            valid_inds.append(i)
        return valid_inds

    def _set_group_flag(self):
        """You can delete this method if you annotate the
        height and width of images, then use the superclass method
        """
        self.flag = np.zeros(len(self), dtype=np.uint8)

    def prepare_train_img(self, idx):
        img_info = self.img_infos[idx]
        ann_info = {}
        ann_info['bboxes1'] = img_info.get('bboxes1')#fact
        ann_info['bboxes2'] = img_info.get('bboxes2')#template
        ann_info['labels'] = img_info.get('labels')
        results = dict(img_info=img_info, ann_info=ann_info)
        if self.proposals is not None:
            results['proposals'] = self.proposals[idx]
        self.pre_pipeline(results)
        return self.pipeline(results)

    def prepare_test_img(self, idx):
        img_info = self.img_infos[idx]
        results = dict(img_info=img_info)
        if self.proposals is not None:
            results['proposals'] = self.proposals[idx]
        self.pre_pipeline(results)
        return self.pipeline(results)