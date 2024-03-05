# DyRA (Dynamic Resolution Adjustment Network)
* Companion NN for existing detectors
* This project is built based on detectron2


## Config (detectron2/configs/DyRA)
* WEIGHTS: the path of weights of the resizer must contain the "resizer"
  * ex) "../outputs/resizer-R-18.pkl"
  * Pretrained weights of R-18:
* PARETO_SCALE_ST/END: define a range of the Pareto Scale (base anchor sizes: [32, 64, 128, 256, 512] ** 2)
  * Defaults: COCO - [32, 64 ]** 2, other datasets: all anchors
  * Pareto Opt.: effective for when multiple-sized objects are in the same scene / for some dataset, removing this optimality can help to achieve more acc.
* ENCODER: default setting - ResNet-18
* ConstCosineLR: BASE_LR_END is setted BASE_LR_END/2 of baselines

## How to plug into the network
* Add DynamicResizer into the network attr.
* Before the image processing,
    * sf_dict: contains scale factors and ps_loss
<pre>
  <code>
if self.resizer is not None:
  sf_dict = self.resizer(batched_inputs, debug=self.training)
  batched_inputs = self.resizer.get_resized_inputs(sf_dict.pop("sf"), batched_inputs)
  </code>
</pre>
* Compute balanceloss, such as,
   * The "gt_boxes" must be one-to-one matched with the "loss_box_reg"
<pre>
  <code>
resizer_dict = self.resizer.balance_loss(gt_boxes, loss_box_reg)
  </code>
</pre>
* Put sf_dict and resizer_dict in your final loss dictionary
* Start Training!

## Commends
* The same as the detectron2
