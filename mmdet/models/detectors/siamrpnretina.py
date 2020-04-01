import torch
import torch.nn as nn
import torch.nn.functional as F
from mmdet.core import bbox2result, bbox2roi, build_assigner, build_sampler
from .. import builder
from ..registry import DETECTORS
from .base import BaseDetector
from .test_mixins import BBoxTestMixin, MaskTestMixin, RPNTestMixin


@DETECTORS.register_module
class SiamRpnRetina(BaseDetector, RPNTestMixin, BBoxTestMixin,
                       MaskTestMixin):
    """Base class for two-stage detectors.

    Two-stage detectors typically consisting of a region proposal network and a
    task-specific regression head.
    """

    def __init__(self,
                 backbone,
                 neck=None,
                 shared_head=None,
                 rpn_head=None,
                 bbox_roi_extractor=None,
                 bbox_head=None,
                 mask_roi_extractor=None,
                 mask_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None):
        super(SiamRpnRetina, self).__init__()
        self.backbone = builder.build_backbone(backbone)

        if neck is not None:
            self.neck = builder.build_neck(neck)

        if rpn_head is not None:
            self.rpn_head = builder.build_head(rpn_head)

        if bbox_head is not None:
            self.bbox_roi_extractor = builder.build_roi_extractor(
                bbox_roi_extractor)
            self.bbox_head = builder.build_head(bbox_head)

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

        self.init_weights(pretrained=pretrained)

    @property
    def with_rpn(self):
        return hasattr(self, 'rpn_head') and self.rpn_head is not None

    def init_weights(self, pretrained=None):
        super(SiamRpnRetina, self).init_weights(pretrained)
        self.backbone.init_weights(pretrained=pretrained)
        if self.with_neck:
            if isinstance(self.neck, nn.Sequential):
                for m in self.neck:
                    m.init_weights()
            else:
                self.neck.init_weights()
        if self.with_shared_head:
            self.shared_head.init_weights(pretrained=pretrained)
        if self.with_rpn:
            self.rpn_head.init_weights()
        if self.with_bbox:
            self.bbox_roi_extractor.init_weights()
            self.bbox_head.init_weights()

    def extract_feat(self, template_img, fact_img):
        x = self.backbone(template_img)
        if self.with_neck:
            x = self.neck(x)

        y = self.backbone(fact_img)
        if self.with_neck:
            y = self.neck(y)
        return x,y

    def forward_train(self,
                      template_img,
                      fact_img,
                      img_meta,
                      gt_bboxes1,
                      gt_bboxes2,
                      gt_labels,
                      gt_bboxes_ignore=None,
                      proposals=None):
        losses = {}
        template_feat, fact_feat = self.extract_feat(template_img, fact_img)
        if self.with_rpn:
            img_meta[0]['scale_factor'] = img_meta[0]['scale_factor2']
            rpn_outs = self.rpn_head(template_feat)
            proposal_cfg = self.train_cfg.get('rpn_proposal',
                                              self.test_cfg.rpn)
            proposal_inputs = rpn_outs + (img_meta, proposal_cfg)
            proposal_list = self.rpn_head.get_bboxes(*proposal_inputs)
        else:
            proposal_list = proposals

        if self.with_bbox:
            bbox_assigner = build_assigner(self.train_cfg.rcnn.assigner)
            bbox_sampler = build_sampler(
                self.train_cfg.rcnn.sampler, context=self)
            num_imgs = fact_img.size(0)
            if gt_bboxes_ignore is None:
                gt_bboxes_ignore = [None for _ in range(num_imgs)]
            sampling_results = []
            for i in range(num_imgs):
                assign_result = bbox_assigner.assign(proposal_list[i],
                                                     gt_bboxes2[i],
                                                     gt_bboxes_ignore[i],
                                                     gt_labels[i])
                sampling_result = bbox_sampler.sample(
                    assign_result,
                    proposal_list[i],
                    gt_bboxes2[i],
                    gt_labels[i],
                    feats=[lvl_feat[i][None] for lvl_feat in template_feat])
                sampling_results.append(sampling_result)
            rois = bbox2roi([res.bboxes for res in sampling_results])
            # TODO: a more flexible way to decide which feature maps to use
            bbox_feats = self.bbox_roi_extractor(
                template_feat[:self.bbox_roi_extractor.num_inputs], rois.float())
            out = []
            for feat in fact_feat:
                featmap = F.conv2d(feat,bbox_feats,padding=3)
                out.append(featmap)
            outs = self.bbox_head(out)
            loss_inputs = outs + (gt_bboxes1, gt_labels, img_meta, self.train_cfg)
            losses = self.bbox_head.loss(
                *loss_inputs, gt_bboxes_ignore=gt_bboxes_ignore)
        return losses

    async def async_simple_test(self,img,img_meta,proposals=None,rescale=False):
        pass
    def simple_test(self, template_img, fact_img, img_meta, rescale=False,proposals=None):
        """Test without augmentation."""
        assert self.with_bbox, "Bbox head must be implemented."

        template_feat, fact_feat = self.extract_feat(template_img, fact_img)
        img_meta[0]['scale_factor'] = img_meta[0]['scale_factor2']
        if proposals is None:
            proposal_list = self.simple_test_rpn(template_feat, img_meta,
                                                 self.test_cfg.rpn)
        else:
            proposal_list = proposals

        rois = bbox2roi(proposal_list)
        roi_feats = self.bbox_roi_extractor(
            template_feat[:len(self.bbox_roi_extractor.featmap_strides)], rois)
        out = []
        for feat in fact_feat:
            featmap = F.conv2d(feat, roi_feats, padding=3)
            out.append(featmap)
        img_meta[0]['scale_factor'] = img_meta[0]['scale_factor1']
        outs = self.bbox_head(out)
        bbox_inputs = outs + (img_meta, self.test_cfg, rescale)
        bbox_list = self.bbox_head.get_bboxes(*bbox_inputs)
        bbox_results = [
            bbox2result(det_bboxes, det_labels, self.bbox_head.num_classes)
            for det_bboxes, det_labels in bbox_list
        ]
        return bbox_results[0]

    def aug_test(self, imgs, img_metas, rescale=False):
        pass
    def forward_test(self,template_img, fact_img, img_metas, **kwargs):
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
    def forward(self,
                template_img,
                fact_img,
                img_meta,
                return_loss=True,
                **kwargs):
        if return_loss:
            return self.forward_train(template_img, fact_img, img_meta,
                                      **kwargs)
        else:
            return self.forward_test(template_img, fact_img, img_meta,
                                     **kwargs)
