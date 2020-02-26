from ..registry import DETECTORS
from .single_stage import SingleStageDetector
from mmdet.core import bbox2result
import torch
@DETECTORS.register_module
class TemplateDetector(SingleStageDetector):

    def __init__(self,
                 backbone,
                 neck,
                 bbox_head,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None):
        super(TemplateDetector, self).__init__(backbone, neck, bbox_head,
                                               train_cfg, test_cfg, pretrained)

    def extract_feat(self, template_img, fact_img):
        """Directly extract features from the the backbone+neck
        """
        #a = torch.sum(torch.isnan(template_img))
        template_feat = self.backbone(template_img)
        #b = torch.sum(torch.isnan(template_feat[1]))
        fact_feat = self.backbone(fact_img)
        assert self.with_neck
        x = self.neck(template_feat, fact_feat)
        return x
    def simple_test(self, template_img, fact_img, img_meta, rescale=False):
        x = self.extract_feat(template_img,fact_img)
        outs = self.bbox_head(x)
        bbox_inputs = outs + (img_meta, self.test_cfg, rescale)
        bbox_list = self.bbox_head.get_bboxes(*bbox_inputs)
        bbox_results = [
            bbox2result(det_bboxes, det_labels, self.bbox_head.num_classes)
            for det_bboxes, det_labels in bbox_list
        ]
        return bbox_results[0]

    def forward_test(self,template_img, fact_img, img_metas, **kwargs):
        """
        Args:
            imgs (List[Tensor]): the outer list indicates test-time
                augmentations and inner Tensor should have a shape NxCxHxW,
                which contains all images in the batch.
            img_meta (List[List[dict]]): the outer list indicates test-time
                augs (multiscale, flip, etc.) and the inner list indicates
                images in a batch
        """
        for var, name in [(template_img, 'template_img'), (fact_img, 'fact_img'), (img_metas, 'img_metas')]:
            if not isinstance(var, list):
                raise TypeError('{} must be a list, but got {}'.format(
                    name, type(var)))

        num_augs = len(fact_img)
        if num_augs != len(img_metas):
            raise ValueError(
                'num of augmentations ({}) != num of image meta ({})'.format(
                    len(fact_img), len(img_metas)))
        # TODO: remove the restriction of imgs_per_gpu == 1 when prepared
        imgs_per_gpu = fact_img[0].size(0)
        assert imgs_per_gpu == 1

        if num_augs == 1:
            return self.simple_test(template_img[0], fact_img[0], img_metas[0], **kwargs)
        else:
            return self.aug_test(template_img, fact_img, img_metas, **kwargs)
    def forward_train(self,
                      template_img,
                      fact_img,
                      img_metas,
                      gt_bboxes,
                      gt_labels,
                      gt_bboxes_ignore=None):
        x = self.extract_feat(template_img, fact_img)
        outs = self.bbox_head(x)
        loss_inputs = outs + (gt_bboxes, gt_labels, img_metas, self.train_cfg)
        losses = self.bbox_head.loss(
            *loss_inputs, gt_bboxes_ignore=gt_bboxes_ignore)
        return losses

    def forward(self,
                template_img,
                fact_img,
                img_meta,
                return_loss=True,
                **kwargs):
        """
        The only difference between this detector class and other
        mmdet detectors is that it takes two imgs, i.e., `template_img`
        and `fact_img` as the input.
        """
        if return_loss:
            return self.forward_train(template_img, fact_img, img_meta,
                                      **kwargs)
        else:
            return self.forward_test(template_img, fact_img, img_metas=img_meta,
                                     **kwargs)
