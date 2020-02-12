from ..registry import DETECTORS
from .single_stage import SingleStageDetector


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
        template_feat = self.backbone(template_img)
        fact_feat = self.backbone(fact_img)
        assert self.with_neck
        x = self.neck(template_feat, fact_feat)
        return x

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
            return self.forward_test(template_img, fact_img, img_meta,
                                     **kwargs)
