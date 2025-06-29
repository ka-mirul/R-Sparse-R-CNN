# Modified by Kamirul Kamirul
# Contact: kamirul.apr@gmail.com

# Original implementation by Peize Sun, Rufeng Zhang
# Contact: {sunpeize, cxrfzhang}@foxmail.com

# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
RSparseRCNN Transformer class.

Copy-paste from torch.nn.Transformer with modifications:
    * positional encodings are passed in MHattention
    * extra LN at the end of encoder is removed
    * decoder returns a stack of activations from all decoding layers
"""
import copy
import math
import torch
from torch import nn
import torch.nn.functional as F
from detectron2.modeling.poolers import ROIPooler
from detectron2.structures import RotatedBoxes
import numpy as np


_DEFAULT_SCALE_CLAMP = math.log(100000.0 / 16)


class DynamicHead(nn.Module):

    def __init__(self, cfg, roi_input_shape):
        super().__init__()

        # Build RoI.
        box_pooler = self._init_box_pooler(cfg, roi_input_shape)
        self.box_pooler = box_pooler
        
        # Build heads.
        num_classes = cfg.MODEL.RSparseRCNN.NUM_CLASSES
        d_model = cfg.MODEL.RSparseRCNN.HIDDEN_DIM
        dim_feedforward = cfg.MODEL.RSparseRCNN.DIM_FEEDFORWARD
        nhead = cfg.MODEL.RSparseRCNN.NHEADS
        dropout = cfg.MODEL.RSparseRCNN.DROPOUT
        activation = cfg.MODEL.RSparseRCNN.ACTIVATION
        num_heads = cfg.MODEL.RSparseRCNN.NUM_HEADS
        rcnn_head = RCNNHead(cfg, d_model, num_classes, dim_feedforward, nhead, dropout, activation)        
        self.head_series = _get_clones(rcnn_head, num_heads)
        self.return_intermediate = cfg.MODEL.RSparseRCNN.DEEP_SUPERVISION
        
        # Init parameters.
        self.use_focal = cfg.MODEL.RSparseRCNN.USE_FOCAL
        self.num_classes = num_classes
        if self.use_focal:
            prior_prob = cfg.MODEL.RSparseRCNN.PRIOR_PROB
            self.bias_value = -math.log((1 - prior_prob) / prior_prob)
        self._reset_parameters()
        

    def _reset_parameters(self):
        # init all parameters.
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

            # initialize the bias for focal loss.
            if self.use_focal:
                if p.shape[-1] == self.num_classes:
                    nn.init.constant_(p, self.bias_value)

    @staticmethod
    def _init_box_pooler(cfg, input_shape):

        in_features = cfg.MODEL.ROI_HEADS.IN_FEATURES
        pooler_resolution = cfg.MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION
        pooler_scales = tuple(1.0 / input_shape[k].stride for k in in_features)
        sampling_ratio = cfg.MODEL.ROI_BOX_HEAD.POOLER_SAMPLING_RATIO
        pooler_type = cfg.MODEL.ROI_BOX_HEAD.POOLER_TYPE

        # If StandardROIHeads is applied on multiple feature maps (as in FPN),
        # then we share the same predictors and therefore the channel counts must be the same
        in_channels = [input_shape[f].channels for f in in_features]
        # Check all channel counts are equal
        assert len(set(in_channels)) == 1, in_channels

        box_pooler = ROIPooler(
            output_size=13, #pooler_resolution
            scales=pooler_scales,
            sampling_ratio=sampling_ratio,
            pooler_type=pooler_type,
        )
        return box_pooler

    def forward(self, features, init_bboxes, init_features, init_bg_features):

        inter_class_logits = []
        inter_pred_bboxes = []

        bs = len(features[0]) #bs=2
        bboxes = init_bboxes
        
        init_features = init_features[None].repeat(1, bs, 1) #sihape = [1,200,256]
        proposal_features = init_features.clone()

        init_bg_features = init_bg_features[None].repeat(1, bs, 1) #sihape = [1,200,256]
        proposal_bg_features = init_bg_features.clone()


        
        for rcnn_head in self.head_series: 
            class_logits, pred_bboxes, proposal_features, proposal_bg_features = rcnn_head(features, bboxes, 
                                                                     proposal_features, 
                                                                     proposal_bg_features, 
                                                                     self.box_pooler)

            if self.return_intermediate:
                inter_class_logits.append(class_logits)
                inter_pred_bboxes.append(pred_bboxes)
            bboxes = pred_bboxes.detach()
            
        if self.return_intermediate:
            return torch.stack(inter_class_logits), torch.stack(inter_pred_bboxes)
        
        return class_logits[None], pred_bboxes[None]


class RCNNHead(nn.Module):

    def __init__(self, cfg, d_model, num_classes, dim_feedforward=2048, nhead=8, dropout=0.1, activation="relu",
                 scale_clamp: float = _DEFAULT_SCALE_CLAMP, bbox_weights=(2.0, 2.0, 1.0, 1.0, 1.0)):
        super().__init__()

        self.d_model = d_model

        # Foreground
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.inst_interact = DynamicConv(cfg)

        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)


        # Background
        self.self_attn_bg = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.inst_interact_bg = DynamicConv(cfg)

        self.linear1_bg = nn.Linear(d_model, dim_feedforward)
        self.dropout_bg = nn.Dropout(dropout)
        self.linear2_bg = nn.Linear(dim_feedforward, d_model)

        self.norm1_bg = nn.LayerNorm(d_model)
        self.norm2_bg = nn.LayerNorm(d_model)
        self.norm3_bg = nn.LayerNorm(d_model)
        self.dropout1_bg = nn.Dropout(dropout)
        self.dropout2_bg = nn.Dropout(dropout)
        self.dropout3_bg = nn.Dropout(dropout)


        self.dropout1_fuse = nn.Dropout(dropout)
        self.norm1_fuse = nn.LayerNorm(d_model)

        self.activation_bg = _get_activation_fn(activation)

        self.multihead_attention_fuse = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.device = torch.device(cfg.MODEL.DEVICE)

        # cls.
        num_cls = cfg.MODEL.RSparseRCNN.NUM_CLS
        cls_module = list()
        for _ in range(num_cls):
            cls_module.append(nn.Linear(d_model, d_model, False))
            cls_module.append(nn.LayerNorm(d_model))
            cls_module.append(nn.ReLU(inplace=True))
        self.cls_module = nn.ModuleList(cls_module)

        # reg.
        num_reg = cfg.MODEL.RSparseRCNN.NUM_REG
        reg_module = list()
        for _ in range(num_reg):
            reg_module.append(nn.Linear(d_model, d_model, False))
            reg_module.append(nn.LayerNorm(d_model))
            reg_module.append(nn.ReLU(inplace=True))
        self.reg_module = nn.ModuleList(reg_module)
        
        # pred.
        self.use_focal = cfg.MODEL.RSparseRCNN.USE_FOCAL
        if self.use_focal:
            self.class_logits = nn.Linear(d_model, num_classes)
        else:
            self.class_logits = nn.Linear(d_model, num_classes + 1)
        self.bboxes_delta = nn.Linear(d_model, 5)
        self.scale_clamp = scale_clamp
        self.bbox_weights = bbox_weights #bbox_weights=(2.0, 2.0, 1.0, 1.0)


    def forward(self, features, bboxes, pro_features, pro_bg_features, pooler):
        """
        :param bboxes: (N, nr_boxes, 5)
        :param pro_features: (N, nr_boxes, d_model)
        """

        bg_bboxes = bboxes.clone()
        N, nr_boxes = bg_bboxes.shape[:2] # N: batch_size, nr_boxes: num_proposal(100)
        proposal_boxes = list()

        for b in range(N):
            bg_bboxes[b,:,2] = bg_bboxes[b,:,2]*1.857
            bg_bboxes[b,:,3] = bg_bboxes[b,:,3]*1.857
            proposal_boxes.append(RotatedBoxes(bg_bboxes[b,:,:5]))


        roi_features_combined = pooler(features, proposal_boxes) 

        roi_features = roi_features_combined[:,:,3:10,3:10]
        roi_features = roi_features.reshape(N * nr_boxes, self.d_model, -1).permute(2, 0, 1)        

        # self_att. on (proposal object features)
        pro_features = pro_features.view(N, nr_boxes, self.d_model).permute(1, 0, 2)
        pro_features2 = self.self_attn(pro_features, pro_features, value=pro_features)[0]
        pro_features = pro_features + self.dropout1(pro_features2)
        pro_features = self.norm1(pro_features)

        # inst_interact. (Dynamic Convolution)
        pro_features = pro_features.view(nr_boxes, N, self.d_model).permute(1, 0, 2).reshape(1, N * nr_boxes, self.d_model)
        pro_features2 = self.inst_interact(pro_features, roi_features) #DynamicConv
        pro_features = pro_features + self.dropout2(pro_features2)
        obj_features = self.norm2(pro_features)

        # obj_feature.
        obj_features2 = self.linear2(self.dropout(self.activation(self.linear1(obj_features))))
        obj_features = obj_features + self.dropout3(obj_features2)
        obj_features_out = self.norm3(obj_features)
        # ========================================================================================

        # extract background RoI features
        roi_bg_features = roi_features_combined.clone()
        roi_bg_features = self.average_center(roi_bg_features)
        roi_bg_features = F.interpolate(roi_bg_features, size=(7, 7), mode='bilinear', align_corners=False)
        roi_bg_features = roi_bg_features.reshape(N * nr_boxes, self.d_model, -1).permute(2, 0, 1)   

        # self_att. on (proposal background features)
        pro_bg_features = pro_bg_features.view(N, nr_boxes, self.d_model).permute(1, 0, 2)
        pro_bg_features2 = self.self_attn_bg(pro_bg_features, pro_bg_features, value=pro_bg_features)[0]
        pro_bg_features = pro_bg_features + self.dropout1_bg(pro_bg_features2)
        pro_bg_features = self.norm1_bg(pro_bg_features)

        # inst_interact of background
        pro_bg_features = pro_bg_features.view(nr_boxes, N, self.d_model).permute(1, 0, 2).reshape(1, N * nr_boxes, self.d_model)
        pro_bg_features2 = self.inst_interact_bg(pro_bg_features, roi_bg_features) #DynamicConv
        pro_bg_features = pro_bg_features + self.dropout2_bg(pro_bg_features2)
        bg_features = self.norm2_bg(pro_bg_features)

        # bg_feature.
        bg_features2 = self.linear2_bg(self.dropout_bg(self.activation_bg(self.linear1_bg(bg_features))))
        bg_features = bg_features + self.dropout3_bg(bg_features2)
        bg_features_out = self.norm3_bg(bg_features)

        # fusion
        obj_features = self.multihead_attention_fuse(obj_features_out, bg_features_out, bg_features_out)[0]
        obj_features = self.dropout1_fuse(obj_features)
        obj_features = self.norm1_fuse(obj_features)
        fc_feature = obj_features.transpose(0, 1).reshape(N * nr_boxes, -1)
    

        # BBOX Classification
        cls_feature = fc_feature.clone()
        for cls_layer in self.cls_module: # Check reference of 'self.cls_module' below
            cls_feature = cls_layer(cls_feature)
        class_logits = self.class_logits(cls_feature)

        # BBOX Regression
        reg_feature = fc_feature.clone()
        
        for reg_layer in self.reg_module: # Check reference of 'self.reg_module' below
            reg_feature = reg_layer(reg_feature)
        bboxes_deltas = self.bboxes_delta(reg_feature) 
        pred_bboxes = self.apply_deltas(bboxes_deltas, bg_bboxes.view(-1, 5)) #bbox size will be (batch*num_prop, 6)
        
        return class_logits.view(N, nr_boxes, -1), pred_bboxes.view(N, nr_boxes, -1), obj_features_out, bg_features_out
    


    def apply_deltas(self, deltas, rois, wh_ratio_clip=16 / 1000):

        wcx, wcy, ww, wh, wa = self.bbox_weights
        dx = deltas[:, 0::5] / wcx
        dy = deltas[:, 1::5] / wcy
        dw = deltas[:, 2::5] / ww
        dh = deltas[:, 3::5] / wh
        dangle = deltas[:, 4::5] / wa

        max_ratio = np.abs(np.log(wh_ratio_clip))
        dw = dw.clamp(min=-max_ratio, max=max_ratio)
        dh = dh.clamp(min=-max_ratio, max=max_ratio)
        roi_x = (rois[:, 0]).unsqueeze(1).expand_as(dx)
        roi_y = (rois[:, 1]).unsqueeze(1).expand_as(dy)
        roi_w = (rois[:, 2]).unsqueeze(1).expand_as(dw)
        roi_h = (rois[:, 3]).unsqueeze(1).expand_as(dh)
        roi_angle = torch.deg2rad((rois[:, 4]).unsqueeze(1).expand_as(dangle))
        gx = dx * roi_w * torch.cos(roi_angle) \
             - dy * roi_h * torch.sin(roi_angle) + roi_x
        gy = dx * roi_w * torch.sin(roi_angle) \
             + dy * roi_h * torch.cos(roi_angle) + roi_y
        gw = roi_w * dw.exp()
        gh = roi_h * dh.exp()

        ga = (dangle*torch.pi) + torch.rad2deg(roi_angle)
        ga = torch.remainder(ga + 180.0, 360.0) - 180.0

        bboxes = torch.stack([gx, gy, gw, gh, ga], dim=-1).view_as(deltas)

        return bboxes



    def average_center(self, features):

        # Coordinates for the central 7x7 region
        center_start, center_end = 3, 10  # Indices for center region

        # Step 1: Calculate the total sum and count of all elements in the 13x13 planes
        total_sum = features.sum(dim=(-1, -2), keepdim=True)  # Shape: [400, 256, 1, 1]
        total_count = features.size(-1) * features.size(-2)  # 13 * 13 = 169

        # Step 2: Subtract the sum of the central 7x7 region
        central_region = features[:, :, center_start:center_end, center_start:center_end]
        central_sum = central_region.sum(dim=(-1, -2), keepdim=True)  # Shape: [400, 256, 1, 1]

        # Total sum outside the 7x7 region
        outside_sum = total_sum - central_sum

        # Count of elements outside the 7x7 region
        central_count = (center_end - center_start) ** 2  # 7x7 = 49
        outside_count = total_count - central_count  # 169 - 49 = 120

        # Compute the average outside the 7x7 region
        outside_avg = outside_sum / outside_count  # Shape: [400, 256, 1, 1]

        # Step 3: Replace the central 7x7 region with the average
        features[:, :, center_start:center_end, center_start:center_end] = outside_avg

        return features

            


class DynamicConv(nn.Module):

    def __init__(self, cfg):
        super().__init__()

        self.hidden_dim = cfg.MODEL.RSparseRCNN.HIDDEN_DIM
        self.dim_dynamic = cfg.MODEL.RSparseRCNN.DIM_DYNAMIC
        self.num_dynamic = cfg.MODEL.RSparseRCNN.NUM_DYNAMIC
        self.num_params = self.hidden_dim * self.dim_dynamic
        self.dynamic_layer = nn.Linear(self.hidden_dim, self.num_dynamic * self.num_params)

        self.norm1 = nn.LayerNorm(self.dim_dynamic)
        self.norm2 = nn.LayerNorm(self.hidden_dim)

        self.activation = nn.ReLU(inplace=True)

        pooler_resolution = cfg.MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION
        num_output = self.hidden_dim * pooler_resolution ** 2
        self.out_layer = nn.Linear(num_output, self.hidden_dim)
        self.norm3 = nn.LayerNorm(self.hidden_dim)

    def forward(self, pro_features, roi_features):
        #DynamicConv
        '''
        pro_features: (1,  N * nr_boxes, self.d_model)
        roi_features: (49, N * nr_boxes, self.d_model)
        '''
        features = roi_features.permute(1, 0, 2)
        parameters = self.dynamic_layer(pro_features).permute(1, 0, 2)

        param1 = parameters[:, :, :self.num_params].view(-1, self.hidden_dim, self.dim_dynamic)
        param2 = parameters[:, :, self.num_params:].view(-1, self.dim_dynamic, self.hidden_dim)

        features = torch.bmm(features, param1)
        features = self.norm1(features)
        features = self.activation(features)

        features = torch.bmm(features, param2)
        features = self.norm2(features)
        features = self.activation(features)

        features = features.flatten(1)
        features = self.out_layer(features)
        features = self.norm3(features)
        features = self.activation(features)

        return features


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")
