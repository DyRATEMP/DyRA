import logging
import argparse, math, os, copy
from typing import List, Dict, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from detectron2.config import configurable
from detectron2.utils.events import get_event_storage
from detectron2.structures import ImageList, Boxes
from detectron2.layers import cat

from .match_predictor import SimplePredictor, bilinear_interpolation
from ..anchor_generator import build_anchor_generator
from ..backbone import build_backbone
from ..meta_arch import META_ARCH_REGISTRY

__all__ = ["DynamicResizer"]


def normalize(logits, tau=1):
    weight = tau / torch.clamp(tau - logits.detach(), min=1e-1)
    weight /= weight.sum(dim=-1).unsqueeze(dim=-1)
    return weight

def sub_one(x):
    x *= x.numel()
    return x - x.min() + 1


logger = logging.getLogger(__name__)

@META_ARCH_REGISTRY.register()
class DynamicResizer(nn.Module): 
    @configurable
    def __init__(self, *, 
                 net: SimplePredictor, 
                 image_encoder, 
                 base_anchors, 
                 pixel_mean, 
                 pixel_std, 
                 pareto_scale_st,
                 pareto_scale_end,
                 pareto_opt=True,
                 out_layer=["res4"]):
        super().__init__()
        
        self.predictor = net
        self.image_encoder = image_encoder
        self.out_feat = out_layer
        self.use_pareto_opt = pareto_opt
        self.eps = 1e-12
        
        self.register_buffer("pixel_mean", torch.tensor(pixel_mean).view(-1, 1, 1), False)
        self.register_buffer("pixel_std", torch.tensor(pixel_std).view(-1, 1, 1), False)
        
        self.base_anchors = torch.tensor(base_anchors).to(self.device) ** 2
        self.coco_bnd = torch.tensor([32., 96.]).to(self.device) ** 2
        self.mapping_ratio = nn.Parameter((self.base_anchors[1::2]).mean() / self.coco_bnd.mean())
        self.anchor_range = [pareto_scale_st, pareto_scale_end]
        
        logger.info("Pareto Scale: {}".format(self.base_anchors[self.anchor_range[0]:self.anchor_range[1]].sqrt().tolist()))

        ## If you want faster training speed, use the following code
        # print("No parameter updating after {}th block".format(int(out_layer[0][-1])))
        # for idx, stage in enumerate(self.image_encoder.stages, start=int(out_layer[0][-1])+1):
        #     for block in stage.children():    block.freeze()
        
    @property
    def device(self):
        return self.predictor.device
    
    @classmethod
    def from_config(cls, cfg):
        image_encoder = build_backbone(cfg, is_resizer=True)
        
        return {
            "net": SimplePredictor(in_chan=15, device=cfg.MODEL.DEVICE,
                                   num_mlayer=cfg.MODEL.RESIZER.ENCODER.RES2_OUT_CHANNELS*4),
            "image_encoder": image_encoder,
            "pareto_opt": True,
            "out_layer": cfg.MODEL.RESIZER.ENCODER.OUT_FEATURES,
            "base_anchors": [32., 64., 128., 256., 512.], 
            "pareto_scale_st": cfg.MODEL.RESIZER.PARETO_SCALE_ST,
            "pareto_scale_end": cfg.MODEL.RESIZER.PARETO_SCALE_END, 
            "pixel_mean": cfg.MODEL.PIXEL_MEAN, 
            "pixel_std": cfg.MODEL.PIXEL_STD,
        }
    
    def forward(self, batched_inputs: List[Dict[str, torch.Tensor]], debug=False):
        sc_facs, size_deb = [], []
        
        for i, x in enumerate(batched_inputs):
            img = self.preprocess_image(x)
            sc_sf = self.predictor(self.image_encoder(img.tensor)[self.out_feat[0]])
            sc_facs.append(sc_sf.squeeze())
            
            if debug and "instances" in x:
                gt_boxes = x["instances"].get_fields()["gt_boxes"].tensor.to(self.device)
                box_sizes = (gt_boxes[:, 2] - gt_boxes[:, 0]) * (gt_boxes[:, 3] - gt_boxes[:, 1])
                size_deb.append((-self.map_range(box_sizes, self.coco_bnd)).sigmoid().mean())
                
        if debug:
            get_event_storage().put_scalar("sf", sc_facs[0].item())
            get_event_storage().put_scalar("gamma", self.mapping_ratio.item())
            if size_deb:    get_event_storage().put_scalar("size_ratio", size_deb[0].item())
            
        if self.training:
            return dict(self.ps_loss(batched_inputs, sc_facs), **{"sf": sc_facs})
        else:
            return {"sf": sc_facs}
    
    def get_resized_inputs(self, factors, batched_inputs, norm=8):
        resized_inputs = []
        
        for i, x in enumerate(batched_inputs):
            hw = torch.tensor(x["image"].shape[-2:]).to(self.device)
            sf = (factors[i] * hw.max() / norm).round() * norm / hw.max()
            resized_img = bilinear_interpolation(x["image"], (hw*sf).round()).squeeze(0)

            resized_inst = copy.deepcopy(x)
            assert not torch.any(torch.isnan(resized_img))            
            if "instances" in x:
                resized_inst["instances"]._image_size = (resized_img.shape[-2], resized_img.shape[-1])
                
                if "gt_boxes" in resized_inst["instances"]._fields:
                    resized_inst["instances"]._fields["gt_boxes"].tensor *= sf.item()    

                if "gt_masks" in resized_inst["instances"]._fields:
                    poly_annos = resized_inst["instances"]._fields["gt_masks"].polygons
                    poly_annos = [np.array([poly_lst[0] * sf.item()]) for poly_lst in poly_annos]
                    resized_inst["instances"]._fields["gt_masks"].polygons = poly_annos
              
            resized_inst["image"] = resized_img
            resized_inputs.append(resized_inst)
        
        return resized_inputs
        
    def ps_loss(self, batched_inputs, sc_sf):
        assert not torch.any(torch.isnan(torch.stack(sc_sf, dim=0)))
        sc_losses = []
        
        for i, x in enumerate(batched_inputs):
            scaled_boxes = x["instances"].get_fields()["gt_boxes"].tensor.detach().to(self.device)
            scaled_boxes *= sc_sf[i].detach()
            scaled_boxes = (scaled_boxes[:, 2] - scaled_boxes[:, 0]) * (scaled_boxes[:, 3] - scaled_boxes[:, 1])
            
            size_ratio = (-self.map_range(scaled_boxes, self.coco_bnd * self.mapping_ratio)).sigmoid()
            sc_bnd = self.predictor.gain
            sc_loss = - (size_ratio * (sc_sf[i] / sc_bnd).clamp(min=self.eps).log() + 
                        (1 - size_ratio) * (1 - sc_sf[i] / sc_bnd).clamp(min=self.eps).log())
            
            if self.use_pareto_opt:
                base_sizes = self.base_anchors[self.anchor_range[0]:self.anchor_range[1]]
                losses = [sc_loss[torch.where((scaled_boxes>=bnd[0]) & (scaled_boxes<bnd[1]))].mean()
                          for bnd in zip(base_sizes, base_sizes[1:])]
                losses.append(sc_loss[torch.where(scaled_boxes<base_sizes[0])].mean())
                losses.append(sc_loss[torch.where(scaled_boxes>=base_sizes[-1])].mean())
                pareto_sc = torch.stack(losses, dim=0)
                pareto_sc = -(((-pareto_sc[torch.where(pareto_sc.isnan()==False)]).exp()).prod()).log()
                sc_losses.append(pareto_sc)
            else:
                sc_losses.append(sc_loss)
        
        if self.use_pareto_opt:
            return {"loss_ps": torch.stack(sc_losses, dim=0).mean()}
        else:
            return {"loss_ps": torch.cat(sc_losses, dim=0).mean()}
    
    def balance_loss(self, box_sizes: List[torch.Tensor], loc_loss: torch.Tensor, reduction="mean"):
        if box_sizes.size()[-1] == 4:
            box_sizes = (box_sizes[:, 3] - box_sizes[:, 1]) * (box_sizes[:, 2] - box_sizes[:, 0])
        
        assert box_sizes.size()[0] == loc_loss.size()[0]
        ac_size_mean = (self.coco_bnd * self.mapping_ratio).mean()
        
        if reduction == "sum":
            losses_group = torch.stack([loc_loss[torch.where(box_sizes<ac_size_mean)].sum(), 
                                        loc_loss[torch.where(box_sizes>=ac_size_mean)].sum()])
        elif reduction == "mean":
            losses_group = torch.stack([loc_loss[torch.where(box_sizes<ac_size_mean)].mean(), 
                                        loc_loss[torch.where(box_sizes>=ac_size_mean)].mean()])
        else:
            raise Exception("Undefined method, it must be mean or sum")
        
        loc_weight = normalize(losses_group.nan_to_num(nan=(losses_group[losses_group.isnan()==False][0].item())))
        loc_weight = sub_one(loc_weight)
        tar_ratio = (loc_weight.flip(-1) * self.base_anchors[1::2]).mean() / self.coco_bnd.mean()
    
        return {"loss_bal": F.l1_loss(self.mapping_ratio, tar_ratio)}
    
    def preprocess_image(self, x: Dict[str, torch.Tensor]):
        """
        Normalize, pad and batch the input images.
        """
        image = (x["image"]).to(self.device)
        image = (image - self.pixel_mean) / self.pixel_std
        image = ImageList.from_tensors([image,])
        return image
        
    def map_range(self, x, xrange, target_range=[-6, 6]):
        div_num = (xrange[1] - xrange[0]) / (target_range[1] - target_range[0])
        rng_min = target_range[0] - xrange[0] / div_num
        return x / div_num + rng_min
