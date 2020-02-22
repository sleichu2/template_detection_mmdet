import mmcv
import collections
from mmdet.utils import build_from_cfg
from mmdet.datasets.registry import PIPELINES

class Compose(object):
    def __init__(self, transforms):
        assert isinstance(transforms, collections.abc.Sequence)
        self.transforms = []
        for transform in transforms:
            if isinstance(transform, dict):
                transform = build_from_cfg(transform, PIPELINES)
                self.transforms.append(transform)
            elif callable(transform):
                self.transforms.append(transform)
            else:
                raise TypeError('transform must be callable or a dict')

    def __call__(self, data):
        for t in self.transforms:
            data = t(data)
            if data is None:
                return None
        return data

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        for t in self.transforms:
            format_string += '\n'
            format_string += '    {0}'.format(t)
        format_string += '\n)'
        return format_string
class mypipline():
    def __init__(self,config):
        print(config['data']['test'].pipeline)
        self.pipeline = Compose(config['data']['test'].pipeline)


if __name__ == "__main__":

    configfile = "../template_detector_r50.py"
    config = mmcv.Config.fromfile(configfile)
    pipeline = mypipline(config)
    pass
