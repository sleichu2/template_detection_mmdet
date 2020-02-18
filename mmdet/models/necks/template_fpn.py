import torch

from ..registry import NECKS
from .fpn import FPN


@NECKS.register_module
class TemplateFPN(FPN):

    def __init__(self,
                 in_channels,
                 out_channels,
                 num_outs,
                 merge_method='concat',
                 **kwargs):
        self.merge_method = merge_method
        if self.merge_method == 'concat':
            in_channels = [2 * t for t in in_channels]
        super(TemplateFPN, self).__init__(in_channels, out_channels, num_outs,
                                          **kwargs)

    def merge_feats(self, template_feat, fact_feat):
        if self.merge_method == 'concat':
            # here, we use the simple concat to merge the features
            # you can explore more methods for more possibility
            # and better performance
            x = [
                torch.cat((x1, x2), dim=1)
                for x1, x2 in zip(template_feat, fact_feat)
            ]
            return x
        else:
            raise NotImplementedError

    def forward(self, template_feat, fact_feat):
        assert len(template_feat) == len(fact_feat) == len(self.in_channels)

        # merge the features from two images
        inputs = self.merge_feats(template_feat, fact_feat)

        # then forward as FPN
        return super(TemplateFPN, self).forward(inputs)
