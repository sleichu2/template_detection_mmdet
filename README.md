
# README Document

The codes are based on mmdetection. If you have never used the mmdetection framework before, please read the orignal [README](ORI_README.md) document.

## Template Detector

The detector takes two images as inputs: template image and fact image. You can see the [detector code](mmdet/models/detectors/template_detector.py) for details.
Load config file: [template_detector_r50.py](template_detector_r50.py)

Other implementation of the detector is similar to other mmdetection detectors.

- **Overview**
  The detector is based on RetinaNet, which is a multi-class RPN, i.e., single-stage detector. If you are not familiar with the RetinaNet, please read the paper for details.

- **Backbone**
  The backbone is ResNet-50. You can change the backbone to ResNet-101 in the config file. The input images of the backbone are template and fact images. It outputs the features of both images.

- **Neck**
  The neck part of RetinaNet in mmdetection is FPN. Original FPN take feature list of one image as the input. Here modified [TemplateFPN](mmdet/models/necks/template_fpn.py) takes two feature lists. You can write your codes with the merge_feats method to hanlde the feature merge process.

- **Heads**
  The two heads, which are namely subnets in the RetinaNet paper, are still used in the detector for classification and regresssion.


## Data

Write your dataset file like [COCO](mmdet/datasets/coco.py) and [CustomDataset](mmdet/datasets/custom.py) classes in [TemplateDataset](mmdet/datasets/template_dataset.py) file.

### Annotations

```
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
```
