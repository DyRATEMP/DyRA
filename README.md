# DyRA: Dynamic Resolution Adjustment Network
* [arxiv paper](https://arxiv.org/abs/2311.17098)
* Companion NN for existing detectors
* This project was built based on detectron2

## Models (on COCO)
* Download link - Google Drive

| Networks | Baseline AP | AP with DyRA | Weights | 
|---|---|---|---|
| RetinaNet (R50)  | 38.7  | 40.1 (+1.4%)  |  [link](https://drive.google.com/file/d/18Z-kc65QtU6EG4v4Tmc7c0JzTb1U8axl/view?usp=drive_link) |
| RetinaNet (R101) |  40.4 | 41.6 (+1.2%)  |  [link](https://drive.google.com/file/d/1uWjLIONmQhNuBb6ppBWcxDzVVxh4kx3H/view?usp=drive_link) |
| Faster-RCNN (R50)  | 40.2  | 41.2 (+1.0%)  | [link](https://drive.google.com/file/d/1ghuLWhXciebLNS8fu_2PfNyrQSU_ggss/view?usp=sharing)  |
| Faster-RCNN (R101)  | 42.0  | 42.9 (+0.9%)  |  [link](https://drive.google.com/file/d/1HmeIXElpKPRKRJukWC1TbzRCl4DJPAWc/view?usp=sharing) |
| FCOS (R50)  | 41.0  | 42.5 (+1.5%) | [link](https://drive.google.com/file/d/1aGA0eJwNZ0ceVt0UBi-sxkf4oEuwVsEu/view?usp=drive_link)  |
| FCOS (R101)  | 43.1  | 44.1 (+1.0%)  | [link](https://drive.google.com/file/d/1Jy3QZSqmv68brYHWCrx9trsJHtqry2i6/view?usp=drive_link) |
| DINO (R50)*  | 49.0 | 50.5 (+1.5%)  | [link](https://drive.google.com/file/d/1hRBgHbdf3FkZ5lHfbGYGLUky03HqwlrT/view?usp=sharing)  |
| H-deformable-detr (R50)*  | 48.6 | 49.9 (+1.3%)  |  [link](https://drive.google.com/file/d/1jq0mmJDHM295ADssQF8651ZXt5QljEQw/view?usp=sharing) |

* Lower accuracy for H-deformable-detr: due to the miss-implemented code
* We'll publish a modified version of code

(*) : trained with AMP trainer

## Config (detectron2/configs/DyRA)
* MODEL.RESIZER.WEIGHTS: A path for pre-trained weights of the resizer must contain a "resizer" string
  * ex) "../outputs/resizer-R-18.pkl"
  * Pretrained weights of R-18: [weight link](https://drive.google.com/file/d/1-mxrNicuyxWJcx3sc1j9PNv5i2l27BpM/view?usp=drive_link)
  * When without setting RESIZER.WEIGHTS, the model will be trained without DyRA
* PARETO_SCALE_ST/END: define a range of Pareto Scale (base anchor sizes: [32, 64, 128, 256, 512] ** 2)
  * If ST != END, Pareto OPT -> enabled, else MLE for all boxes
  * Pareto OPT: effective when multiple-sized objects are in the same scene
* ENCODER: default setting - ResNet-18
* ConstCosineLR: BASE_LR_END is set to LastLR/2 of baselines

## How to plug into the network
* Add DynamicResizer to the network attr.
* Before image processing,
    * sf_dict: contains scale factors and ps_loss
<pre>
  <code>
if self.resizer is not None:
  sf_dict = self.resizer(batched_inputs, debug=self.training)
  batched_inputs = self.resizer.get_resized_inputs(sf_dict.pop("sf"), batched_inputs)
  </code>
</pre>
* Compute balanceloss, such as,
   * "gt_boxes" must be one-to-one matched with "loss_box_reg"
<pre>
  <code>
resizer_dict = self.resizer.balance_loss(gt_boxes, loss_box_reg)
  </code>
</pre>
* Put sf_dict and resizer_dict in your final loss dictionary
* Start training :)

## Commends
* Same as the detectron2

## Acknowledgement
* [detectron2](https://github.com/facebookresearch/detectron2.git) / [detrex](https://github.com/IDEA-Research/detrex.git)
