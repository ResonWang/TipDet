#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.
import math
from copy import deepcopy
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
from yoloV.utils.box_op import box_cxcywh_to_xyxy, generalized_box_iou
import numpy as np

class IOUloss(nn.Module):
    def __init__(self, reduction="none", loss_type="iou"):
        super(IOUloss, self).__init__()
        self.reduction = reduction
        self.loss_type = loss_type

    def forward(self, pred, target):
        assert pred.shape[0] == target.shape[0]

        pred = pred.view(-1, 4)
        target = target.view(-1, 4)
        tl = torch.max(
            (pred[:, :2] - pred[:, 2:] / 2), (target[:, :2] - target[:, 2:] / 2)
        )
        br = torch.min(
            (pred[:, :2] + pred[:, 2:] / 2), (target[:, :2] + target[:, 2:] / 2)
        )

        area_p = torch.prod(pred[:, 2:], 1)
        area_g = torch.prod(target[:, 2:], 1)

        en = (tl < br).type(tl.type()).prod(dim=1)
        area_i = torch.prod(br - tl, 1) * en
        area_u = area_p + area_g - area_i
        iou = (area_i) / (area_u + 1e-16)

        if self.loss_type == "iou":
            loss = 1 - iou ** 2
        elif self.loss_type == "giou":
            c_tl = torch.min(
                (pred[:, :2] - pred[:, 2:] / 2), (target[:, :2] - target[:, 2:] / 2)
            )
            c_br = torch.max(
                (pred[:, :2] + pred[:, 2:] / 2), (target[:, :2] + target[:, 2:] / 2)
            )
            area_c = torch.prod(c_br - c_tl, 1)
            giou = iou - (area_c - area_u) / area_c.clamp(1e-16)
            loss = 1 - giou.clamp(min=-1.0, max=1.0)

        if self.reduction == "mean":
            loss = loss.mean()
        elif self.reduction == "sum":
            loss = loss.sum()

        return loss

class YOLOLoss(nn.Module):    
    def __init__(self, num_classes, fp16, MultiInputs, features = "234", focal_loss=False, alpha=0.99, gamma=5, Dali=False, group_num=4):
        super().__init__()
        self.num_classes        = num_classes
        if features == "123":
            self.strides = [2, 4, 8]
        if features == "234":
            self.strides = [4, 8, 16]
        if features == "345":
            self.strides = [8, 16, 32]
        if features == "23":
            self.strides = [4, 8]
        if features == "2":
            self.strides = [4]
        if features == "3":
            self.strides = [8]
        if features == "34":
            self.strides = [8, 16]
        if features == "2345":
            self.strides = [4, 8, 16, 32]

        self.bcewithlog_loss    = nn.BCEWithLogitsLoss(reduction="none")   # 包含了sigmoid操作
        self.iou_loss           = IOUloss(reduction="none")
        self.grids              = [torch.zeros(1)] * len(self.strides)
        self.fp16               = fp16

        self.l1_loss = nn.L1Loss(reduction="none")
        self.use_l1 = False
        self.MultiInputs = MultiInputs
        self.focal_L = focal_loss
        self.focal_L_a = alpha
        self.focal_L_gamma = gamma
        self.Dali = Dali
        self.group_num = group_num

    def forward(self, inputs, labels=None, iteration=None):
        if self.Dali:
            if isinstance(labels,list):   # val阶段，不用dali
                pass

            else:
                labels[..., 0:2] = labels[..., 0:2] + labels[..., 2:4] / 2  # 0-1 (x1,y1)-->(xc,yc)
                # labels[..., 0:4] = labels[..., 0:4] * 640             # (x1y1wh)
                labels *= 640             # (x1y1wh)

        labels = labels[self.group_num-1::self.group_num]                # 从3开始, bs维度隔4个取一个

        if self.MultiInputs:
            inputs_yolox = [inputs[0][0][self.group_num-1::self.group_num],inputs[0][1][self.group_num-1::self.group_num]]   # inputs[0]：list2:(bs,6,80,80),(bs,6,40,40)  bs维度隔4个取一个
            VID_FC = inputs[1][self.group_num-1::self.group_num]                                 # inputs[1]：(256,10,1)
            pred_result = inputs[2][self.group_num-1::self.group_num]                            # inputs[2]：list:bs (10,6)
            pred_idx = inputs[3][self.group_num-1::self.group_num]                               # inputs[3]：list:bs (10,)

        else:
            inputs_yolox = inputs

        outputs             = []
        x_shifts            = []
        y_shifts            = []
        expanded_strides    = []

        #-----------------------------------------------#
        # inputs    [[batch_size, num_classes + 5, 20, 20]
        #            [batch_size, num_classes + 5, 40, 40]
        #            [batch_size, num_classes + 5, 80, 80]]
        # outputs   [[batch_size, 400, num_classes + 5]
        #            [batch_size, 1600, num_classes + 5]
        #            [batch_size, 6400, num_classes + 5]]
        # x_shifts  [[batch_size, 400]
        #            [batch_size, 1600]
        #            [batch_size, 6400]]
        #-----------------------------------------------#
        for k, (stride, output) in enumerate(zip(self.strides, inputs_yolox)):
            output, grid = self.get_output_and_grid(output, k, stride)     # 解码预测的bbox到640尺寸
            x_shifts.append(grid[:, :, 0])
            y_shifts.append(grid[:, :, 1])
            expanded_strides.append(torch.ones_like(grid[:, :, 0]) * stride)
            outputs.append(output)

        if self.MultiInputs:
            return self.get_losses(x_shifts, y_shifts, expanded_strides, labels, torch.cat(outputs, 1), MultiInputs=self.MultiInputs, VID_FC=VID_FC, idx=pred_idx, pred_res=pred_result)
        else:
            return self.get_losses(x_shifts, y_shifts, expanded_strides, labels, torch.cat(outputs, 1), iteration)

    def focal_loss(self, pred, gt, alpha, gamma):
        a = alpha
        y = gamma
        pos_inds = gt.eq(1).float()
        neg_inds = gt.eq(0).float()
        pos_loss = torch.log(pred + 1e-5) * torch.pow(1 - pred, y) * pos_inds * a     # a=0.75控制类别权重 y=2控制难易程度划分,越大越重视困难样本
        neg_loss = torch.log(1 - pred + 1e-5) * torch.pow(pred, y) * neg_inds * (1-a)
        loss = -(pos_loss + neg_loss)
        return loss

    def get_output_and_grid(self, output, k, stride):
        grid            = self.grids[k]
        hsize, wsize    = output.shape[-2:]
        if grid.shape[2:4] != output.shape[2:4]:
            yv, xv          = torch.meshgrid([torch.arange(hsize), torch.arange(wsize)])
            grid            = torch.stack((xv, yv), 2).view(1, hsize, wsize, 2).type(output.type())
            self.grids[k]   = grid
        grid                = grid.view(1, -1, 2)

        output              = output.flatten(start_dim=2).permute(0, 2, 1)
        output[..., :2]     = (output[..., :2] + grid.type_as(output)) * stride
        output[..., 2:4]    = torch.exp(output[..., 2:4]) * stride
        return output, grid

    def get_losses(self, x_shifts, y_shifts, expanded_strides, labels, outputs, MultiInputs=False, VID_FC=None, idx=None, pred_res=None):
        #-----------------------------------------------#
        #   [batch, n_anchors_all, 4]
        #-----------------------------------------------#
        bbox_preds  = outputs[:, :, :4]  
        #-----------------------------------------------#
        #   [batch, n_anchors_all, 1]
        #-----------------------------------------------#
        obj_preds   = outputs[:, :, 4:5]
        #-----------------------------------------------#
        #   [batch, n_anchors_all, n_cls]
        #-----------------------------------------------#
        cls_preds   = outputs[:, :, 5:]  

        total_num_anchors   = outputs.shape[1]
        #-----------------------------------------------#
        #   x_shifts            [1, n_anchors_all]
        #   y_shifts            [1, n_anchors_all]
        #   expanded_strides    [1, n_anchors_all]
        #-----------------------------------------------#
        x_shifts            = torch.cat(x_shifts, 1).type_as(outputs)
        y_shifts            = torch.cat(y_shifts, 1).type_as(outputs)
        expanded_strides    = torch.cat(expanded_strides, 1).type_as(outputs)


        cls_targets = []
        reg_targets = []
        obj_targets = []
        fg_masks    = []
        ref_masks = []
        ref_targets = []

        num_fg  = 0.0
        for batch_idx in range(outputs.shape[0]):
            num_gt          = len(labels[batch_idx])
            if num_gt == 0:
                cls_target  = outputs.new_zeros((0, self.num_classes))
                reg_target  = outputs.new_zeros((0, 4))
                obj_target  = outputs.new_zeros((total_num_anchors, 1))
                fg_mask     = outputs.new_zeros(total_num_anchors).bool()
                if MultiInputs:
                    ref_target = outputs.new_zeros((idx[batch_idx].shape[0], 1))  # idx: list 256: (10,) 每个值为在对应帧8000个特征中索引    # ref_target (10,1)

            else:
                #-----------------------------------------------#
                #   gt_bboxes_per_image     [num_gt, 4]
                #   gt_classes              [num_gt]
                #   bboxes_preds_per_image  [n_anchors_all, 4]
                #   cls_preds_per_image     [n_anchors_all, num_classes]
                #   obj_preds_per_image     [n_anchors_all, 1]
                #-----------------------------------------------#
                gt_bboxes_per_image     = labels[batch_idx][..., :4].type_as(outputs)
                gt_classes              = labels[batch_idx][..., 4].type_as(outputs)
                bboxes_preds_per_image  = bbox_preds[batch_idx]
                cls_preds_per_image     = cls_preds[batch_idx]
                obj_preds_per_image     = obj_preds[batch_idx]


                gt_matched_classes, fg_mask, pred_ious_this_matching, matched_gt_inds, num_fg_img = self.get_assignments(
                    num_gt, total_num_anchors, gt_bboxes_per_image, gt_classes, bboxes_preds_per_image, cls_preds_per_image, obj_preds_per_image,
                    expanded_strides, x_shifts, y_shifts,
                )

                torch.cuda.empty_cache()
                num_fg      += num_fg_img                                                                                                        #  num_fg_img 为单张图片dynamic_k匹配到的positive数，相对于5376而言其中的3个
                cls_target  = F.one_hot(gt_matched_classes.to(torch.int64), self.num_classes).float() * pred_ious_this_matching.unsqueeze(-1)    # 分类标签，(3,2) 3个positive的one-hot类别 * 预测的IOU
                obj_target  = fg_mask.unsqueeze(-1)                                                                                              # 目标置信度标签，(5376,1) obj_mask, 3个positives位置为True
                reg_target  = gt_bboxes_per_image[matched_gt_inds]                                                                               # 边界框回归标签 (3,4) 3个positives的标注框真值坐标
                # 以下为VID分类部分，包括分类标签分配
                if MultiInputs:
                    fg_idx = torch.where(fg_mask)[0]
                    ref_target = outputs.new_zeros((idx[batch_idx].shape[0],1))
                    fg = 0
                    for ele_idx, ele in enumerate(idx[batch_idx]):
                        loc = torch.where(fg_idx == ele)[0]
                        if len(loc):
                            ref_target[ele_idx, :] = obj_target[ele, :]
                            fg += 1
                            continue

            cls_targets.append(cls_target)
            reg_targets.append(reg_target)
            obj_targets.append(obj_target.type(cls_target.type()))
            fg_masks.append(fg_mask)
            if MultiInputs:
                ref_targets.append(ref_target[:, :])

        cls_targets = torch.cat(cls_targets, 0)
        reg_targets = torch.cat(reg_targets, 0)
        obj_targets = torch.cat(obj_targets, 0)
        fg_masks    = torch.cat(fg_masks, 0)
        if MultiInputs:
            ref_targets = torch.cat(ref_targets, 0)  # (16x10,2) VID类别分类软标签，标签值由和每帧positives的IOU决定


        num_fg      = max(num_fg, 1)
        loss_iou    = (self.iou_loss(bbox_preds.view(-1, 4)[fg_masks], reg_targets)).sum()         # 只统计dynamic-k个positives位置的预测和gt的iou loss，注意同一张图片的全部k个positives位置的gt-box都一样
        if self.focal_L:
            loss_obj = (self.focal_loss(obj_preds.sigmoid().view(-1, 1), obj_targets,self.focal_L_a, self.focal_L_gamma)).sum()
        else:
            loss_obj    = (self.bcewithlog_loss(obj_preds.view(-1, 1), obj_targets)).sum()                       # dynamic-k个positives位置作为gt，统计全局所有特征点的obj loss
        loss_cls    = (self.bcewithlog_loss(cls_preds.view(-1, self.num_classes)[fg_masks], cls_targets)).sum()  # 只统计dynamic-k个positives位置的预测和gt的cls loss
        reg_weight  = 5.0

        if MultiInputs:
            ref_weight = 2.0
            obj_weight = 1.0

            if self.focal_L:
                loss_ref = (self.focal_loss(VID_FC.sigmoid().view(-1, 1), ref_targets, self.focal_L_a, self.focal_L_gamma)).sum()
            else:
                loss_ref = (self.bcewithlog_loss(VID_FC.contiguous().view(-1, 1), ref_targets)).sum()

            loss = reg_weight * loss_iou + obj_weight * loss_obj + loss_cls + ref_weight * loss_ref
            iou_loss = reg_weight * loss_iou
            obj_loss = obj_weight * loss_obj
            cls_loss = loss_cls
            cls_ref_loss = ref_weight * loss_ref

            return {"total_loss": loss / num_fg, "iou_loss": iou_loss / num_fg, "obj_loss": obj_loss / num_fg,
                    "cls_loss": cls_loss / num_fg, "cls_ref_loss": cls_ref_loss / num_fg}


        else:
            loss = reg_weight * loss_iou + loss_obj + loss_cls
            iou_loss = reg_weight * loss_iou
            obj_loss = loss_obj
            cls_loss = loss_cls

            return {"total_loss":loss / num_fg, "iou_loss":iou_loss / num_fg, "obj_loss":obj_loss / num_fg, "cls_loss":cls_loss / num_fg}

    @torch.no_grad()
    def get_assignments(self, num_gt, total_num_anchors, gt_bboxes_per_image, gt_classes, bboxes_preds_per_image, cls_preds_per_image, obj_preds_per_image, expanded_strides, x_shifts, y_shifts):
        #-------------------------------------------------------#
        #   fg_mask: 根据距离初选positives   [n_anchors_all] = (33600=160x160+80x80+40x40,)  在 gt框内 或者 gt中心center_radius*stride定义的邻域范围内 的特征图位置对应为True
        #   is_in_boxes_and_center  [num_gt, len(fg_mask=True)]  (1,len(fg_mask=True)) 1个gt框  fg_mask为True的特征点中同时在 gt框内 或者 gt中心center_radius*stride定义的邻域范围内 的特征点, True/False
        #-------------------------------------------------------#
        fg_mask, is_in_boxes_and_center = self.get_in_boxes_info(gt_bboxes_per_image, expanded_strides, x_shifts, y_shifts, total_num_anchors, num_gt)

        #-------------------------------------------------------#
        #   fg_mask                 [n_anchors_all]
        #   bboxes_preds_per_image  [fg_mask, 4]
        #   cls_preds_              [fg_mask, num_classes]
        #   obj_preds_              [fg_mask, 1]
        #-------------------------------------------------------#
        bboxes_preds_per_image  = bboxes_preds_per_image[fg_mask]    # bboxes_preds_per_image：(33600,4) ---> (len(fg_mask=True), 4) 找出gt框内的那些特征点位置上的预测框
        cls_preds_              = cls_preds_per_image[fg_mask]       # cls_preds_per_image：(33600,2) ---> (len(fg_mask=True), 2) 找出gt框内的那些特征点位置上的两个类别的logits
        obj_preds_              = obj_preds_per_image[fg_mask]       # obj_preds_per_image：(33600,1) ---> (len(fg_mask=True), 1) 找出gt框内的那些特征点位置上的object的logits
        num_in_boxes_anchor     = bboxes_preds_per_image.shape[0]    # len(fg_mask=True)

        #-------------------------------------------------------#
        #   pair_wise_ious      [num_gt, fg_mask]
        #-------------------------------------------------------#
        pair_wise_ious      = self.bboxes_iou(gt_bboxes_per_image, bboxes_preds_per_image, False)  # gt 和 len(fg_mask=True) 个特征点位置 预测框的 IOU       (1,len(fg_mask=True))
        pair_wise_ious_loss = -torch.log(pair_wise_ious + 1e-8)                                    # gt 和 len(fg_mask=True) 个特征点位置 预测框的 IOU LOSS  (1,len(fg_mask=True))
        
        #-------------------------------------------------------#
        #   cls_preds_          [num_gt, fg_mask, num_classes]
        #   gt_cls_per_image    [num_gt, fg_mask, num_classes]
        #-------------------------------------------------------#
        if self.fp16:
            with torch.cuda.amp.autocast(enabled=False):
                cls_preds_          = cls_preds_.float().unsqueeze(0).repeat(num_gt, 1, 1).sigmoid_() * obj_preds_.unsqueeze(0).repeat(num_gt, 1, 1).sigmoid_()   # (1,len(fg_mask=True),2) 在gt框范围内的len(fg_mask=True)个预测框的两个类别的confidence
                gt_cls_per_image    = F.one_hot(gt_classes.to(torch.int64), self.num_classes).float().unsqueeze(1).repeat(1, num_in_boxes_anchor, 1)              # (1,len(fg_mask=True),2) one-hot格式的gt类别ID
                pair_wise_cls_loss  = F.binary_cross_entropy(cls_preds_.sqrt_(), gt_cls_per_image, reduction="none").sum(-1)                                      # (1,len(fg_mask=True)) len(fg_mask=True)个预测框和gt框的类别分类loss
        else:
            cls_preds_          = cls_preds_.float().unsqueeze(0).repeat(num_gt, 1, 1).sigmoid_() * obj_preds_.unsqueeze(0).repeat(num_gt, 1, 1).sigmoid_()
            gt_cls_per_image    = F.one_hot(gt_classes.to(torch.int64), self.num_classes).float().unsqueeze(1).repeat(1, num_in_boxes_anchor, 1)
            pair_wise_cls_loss  = F.binary_cross_entropy(cls_preds_.sqrt_(), gt_cls_per_image, reduction="none").sum(-1)
            del cls_preds_

        cost = pair_wise_cls_loss + 3.0 * pair_wise_ious_loss + 100000.0 * (~is_in_boxes_and_center).float()
        num_fg, gt_matched_classes, pred_ious_this_matching, matched_gt_inds = self.dynamic_k_matching(cost, pair_wise_ious, gt_classes, num_gt, fg_mask)

        del pair_wise_cls_loss, cost, pair_wise_ious, pair_wise_ious_loss
        return gt_matched_classes, fg_mask, pred_ious_this_matching, matched_gt_inds, num_fg
    
    def bboxes_iou(self, bboxes_a, bboxes_b, xyxy=True):
        if bboxes_a.shape[1] != 4 or bboxes_b.shape[1] != 4:
            raise IndexError

        if xyxy:
            tl = torch.max(bboxes_a[:, None, :2], bboxes_b[:, :2])
            br = torch.min(bboxes_a[:, None, 2:], bboxes_b[:, 2:])
            area_a = torch.prod(bboxes_a[:, 2:] - bboxes_a[:, :2], 1)
            area_b = torch.prod(bboxes_b[:, 2:] - bboxes_b[:, :2], 1)
        else:
            tl = torch.max(
                (bboxes_a[:, None, :2] - bboxes_a[:, None, 2:] / 2),
                (bboxes_b[:, :2] - bboxes_b[:, 2:] / 2),
            )
            br = torch.min(
                (bboxes_a[:, None, :2] + bboxes_a[:, None, 2:] / 2),
                (bboxes_b[:, :2] + bboxes_b[:, 2:] / 2),
            )

            area_a = torch.prod(bboxes_a[:, 2:], 1)
            area_b = torch.prod(bboxes_b[:, 2:], 1)
        en = (tl < br).type(tl.type()).prod(dim=2)
        area_i = torch.prod(br - tl, 2) * en
        return area_i / (area_a[:, None] + area_b - area_i)

    def get_in_boxes_info(self, gt_bboxes_per_image, expanded_strides, x_shifts, y_shifts, total_num_anchors, num_gt, center_radius = 2.5):
        #-------------------------------------------------------#
        #   expanded_strides_per_image  [n_anchors_all]                x_shifts (33600,)  三个特征图每个位置的列id: [0,1,..,159,0,1,...39]
        #   x_centers_per_image         [num_gt, n_anchors_all]        y_shifts (33600,)  三个特征图每个位置的行id: [0,0,..,0,1,1,...39]
        #   x_centers_per_image         [num_gt, n_anchors_all]        gt_bboxes_per_image (n,4) 640下的框坐标(x,y,w,h)
        #-------------------------------------------------------#
        expanded_strides_per_image  = expanded_strides[0]                                                                 # (33600,)  特征图每个位置的stride, 共有三个值，前160x160为第一个stride，后80x80为第二个，...
        x_centers_per_image         = ((x_shifts[0] + 0.5) * expanded_strides_per_image).unsqueeze(0).repeat(num_gt, 1)   # (1,33600) 特征图每个位置对应于原图的列位置
        y_centers_per_image         = ((y_shifts[0] + 0.5) * expanded_strides_per_image).unsqueeze(0).repeat(num_gt, 1)   # (1,33600) 特征图每个位置对应于原图的行位置

        #-------------------------------------------------------#
        #   gt_bboxes_per_image_x       [num_gt, n_anchors_all]
        #-------------------------------------------------------#
        gt_bboxes_per_image_l = (gt_bboxes_per_image[:, 0] - 0.5 * gt_bboxes_per_image[:, 2]).unsqueeze(1).repeat(1, total_num_anchors) # (1,33600)  x1
        gt_bboxes_per_image_r = (gt_bboxes_per_image[:, 0] + 0.5 * gt_bboxes_per_image[:, 2]).unsqueeze(1).repeat(1, total_num_anchors) # (1,33600)  x2
        gt_bboxes_per_image_t = (gt_bboxes_per_image[:, 1] - 0.5 * gt_bboxes_per_image[:, 3]).unsqueeze(1).repeat(1, total_num_anchors) # (1,33600)  y1
        gt_bboxes_per_image_b = (gt_bboxes_per_image[:, 1] + 0.5 * gt_bboxes_per_image[:, 3]).unsqueeze(1).repeat(1, total_num_anchors) # (1,33600)  y2

        #-------------------------------------------------------#
        #   bbox_deltas     [num_gt, n_anchors_all, 4]  特征图每个点对应640尺寸下的位置与gt在640下的距离
        #-------------------------------------------------------#
        b_l = x_centers_per_image - gt_bboxes_per_image_l      # 列方向的距离 (1,33600)
        b_r = gt_bboxes_per_image_r - x_centers_per_image      # (1,33600)
        b_t = y_centers_per_image - gt_bboxes_per_image_t      # 行方向的距离 (1,33600)
        b_b = gt_bboxes_per_image_b - y_centers_per_image      # (1,33600)
        bbox_deltas = torch.stack([b_l, b_t, b_r, b_b], 2)     # (1,33600,4)

        #-------------------------------------------------------#
        #   is_in_boxes     [num_gt, n_anchors_all]
        #   is_in_boxes_all [n_anchors_all]
        #-------------------------------------------------------#
        is_in_boxes     = bbox_deltas.min(dim=-1).values > 0.0  # (1,33600) 特征图每个点对应的在640下的位置是否在gt_box范围内 4个差值都>0说明在范围内
        is_in_boxes_all = is_in_boxes.sum(dim=0) > 0            # (33600,) 整合所有gt框范围内的特征图点

        gt_bboxes_per_image_l = (gt_bboxes_per_image[:, 0]).unsqueeze(1).repeat(1, total_num_anchors) - center_radius * expanded_strides_per_image.unsqueeze(0)  # (1,33600) gt框中心位置center_radius*stride定义的邻域范围
        gt_bboxes_per_image_r = (gt_bboxes_per_image[:, 0]).unsqueeze(1).repeat(1, total_num_anchors) + center_radius * expanded_strides_per_image.unsqueeze(0)  # (1,33600)
        gt_bboxes_per_image_t = (gt_bboxes_per_image[:, 1]).unsqueeze(1).repeat(1, total_num_anchors) - center_radius * expanded_strides_per_image.unsqueeze(0)  # (1,33600)
        gt_bboxes_per_image_b = (gt_bboxes_per_image[:, 1]).unsqueeze(1).repeat(1, total_num_anchors) + center_radius * expanded_strides_per_image.unsqueeze(0)  # (1,33600)

        #-------------------------------------------------------#
        #   center_deltas   [num_gt, n_anchors_all, 4]   判断特征图每个点对应的在640下的位置是否在gt中心center_radius*stride定义的邻域范围内 4个差值都>0说明在范围内
        #-------------------------------------------------------#
        c_l = x_centers_per_image - gt_bboxes_per_image_l   # (1,33600)
        c_r = gt_bboxes_per_image_r - x_centers_per_image   # (1,33600)
        c_t = y_centers_per_image - gt_bboxes_per_image_t   # (1,33600)
        c_b = gt_bboxes_per_image_b - y_centers_per_image   # (1,33600)
        center_deltas       = torch.stack([c_l, c_t, c_r, c_b], 2)

        #-------------------------------------------------------#
        #   is_in_centers       [num_gt, n_anchors_all]
        #   is_in_centers_all   [n_anchors_all]
        #-------------------------------------------------------#
        is_in_centers       = center_deltas.min(dim=-1).values > 0.0   # (1,33600)
        is_in_centers_all   = is_in_centers.sum(dim=0) > 0             # (33600,)  统计所有gt的，对于一张图片一个gt，和is_in_centers相等

        #-------------------------------------------------------#
        #   is_in_boxes_anchor      [n_anchors_all]
        #   is_in_boxes_and_center  [num_gt, is_in_boxes_anchor]
        #-------------------------------------------------------#
        is_in_boxes_anchor      = is_in_boxes_all | is_in_centers_all
        is_in_boxes_and_center  = is_in_boxes[:, is_in_boxes_anchor] & is_in_centers[:, is_in_boxes_anchor]
        return is_in_boxes_anchor, is_in_boxes_and_center

    def dynamic_k_matching(self, cost, pair_wise_ious, gt_classes, num_gt, fg_mask):
        #-------------------------------------------------------#
        #   cost                [num_gt, fg_mask]
        #   pair_wise_ious      [num_gt, fg_mask]
        #   gt_classes          [num_gt]        
        #   fg_mask             [n_anchors_all]
        #   matching_matrix     [num_gt, fg_mask]
        #-------------------------------------------------------#
        matching_matrix         = torch.zeros_like(cost)

        #------------------------------------------------------------#
        #   选取iou最大的n_candidate_k个点
        #   然后求和，判断应该有多少点用于该框预测
        #   topk_ious           [num_gt, n_candidate_k]
        #   dynamic_ks          [num_gt]
        #   matching_matrix     [num_gt, fg_mask]
        #------------------------------------------------------------#
        n_candidate_k           = min(10, pair_wise_ious.size(1))
        topk_ious, _            = torch.topk(pair_wise_ious, n_candidate_k, dim=1)
        dynamic_ks              = torch.clamp(topk_ious.sum(1).int(), min=1)
        
        for gt_idx in range(num_gt):
            #------------------------------------------------------------#
            #   给每个真实框选取最小的动态k个点
            #------------------------------------------------------------#
            _, pos_idx = torch.topk(cost[gt_idx], k=dynamic_ks[gt_idx].item(), largest=False)
            matching_matrix[gt_idx][pos_idx] = 1.0
        del topk_ious, dynamic_ks, pos_idx

        #------------------------------------------------------------#
        #   anchor_matching_gt  [fg_mask]
        #------------------------------------------------------------#
        anchor_matching_gt = matching_matrix.sum(0)
        if (anchor_matching_gt > 1).sum() > 0:
            #------------------------------------------------------------#
            #   当某一个特征点指向多个真实框的时候
            #   选取cost最小的真实框。
            #------------------------------------------------------------#
            _, cost_argmin = torch.min(cost[:, anchor_matching_gt > 1], dim=0)
            matching_matrix[:, anchor_matching_gt > 1] *= 0.0
            matching_matrix[cost_argmin, anchor_matching_gt > 1] = 1.0
        #------------------------------------------------------------#
        #   fg_mask_inboxes  [fg_mask]
        #   num_fg为正样本的特征点个数
        #------------------------------------------------------------#
        fg_mask_inboxes = matching_matrix.sum(0) > 0.0
        num_fg          = fg_mask_inboxes.sum().item()

        #------------------------------------------------------------#
        #   对fg_mask进行更新
        #------------------------------------------------------------#
        fg_mask[fg_mask.clone()] = fg_mask_inboxes

        #------------------------------------------------------------#
        #   获得特征点对应的物品种类
        #------------------------------------------------------------#
        matched_gt_inds     = matching_matrix[:, fg_mask_inboxes].argmax(0)
        gt_matched_classes  = gt_classes[matched_gt_inds]

        pred_ious_this_matching = (matching_matrix * pair_wise_ious).sum(0)[fg_mask_inboxes]
        return num_fg, gt_matched_classes, pred_ious_this_matching, matched_gt_inds


def is_parallel(model):
    # Returns True if model is of type DP or DDP
    return type(model) in (nn.parallel.DataParallel, nn.parallel.DistributedDataParallel)

def de_parallel(model):
    # De-parallelize a model: returns single-GPU model if model is of type DP or DDP
    return model.module if is_parallel(model) else model
    
def copy_attr(a, b, include=(), exclude=()):
    # Copy attributes from b to a, options to only include [...] and to exclude [...]
    for k, v in b.__dict__.items():
        if (len(include) and k not in include) or k.startswith('_') or k in exclude:
            continue
        else:
            setattr(a, k, v)

class ModelEMA:
    """ Updated Exponential Moving Average (EMA) from https://github.com/rwightman/pytorch-image-models
    Keeps a moving average of everything in the model state_dict (parameters and buffers)
    For EMA details see https://www.tensorflow.org/api_docs/python/tf/train/ExponentialMovingAverage
    """

    def __init__(self, model, decay=0.9999, tau=2000, updates=0):
        # Create EMA
        self.ema = deepcopy(de_parallel(model)).eval()  # FP32 EMA
        # if next(model.parameters()).device.type != 'cpu':
        #     self.ema.half()  # FP16 EMA
        self.updates = updates  # number of EMA updates
        self.decay = lambda x: decay * (1 - math.exp(-x / tau))  # decay exponential ramp (to help early epochs)
        for p in self.ema.parameters():
            p.requires_grad_(False)

    def update(self, model):
        # Update EMA parameters
        with torch.no_grad():
            self.updates += 1
            d = self.decay(self.updates)

            msd = de_parallel(model).state_dict()  # model state_dict
            for k, v in self.ema.state_dict().items():
                if v.dtype.is_floating_point:
                    v *= d
                    v += (1 - d) * msd[k].detach()

    def update_attr(self, model, include=(), exclude=('process_group', 'reducer')):
        # Update EMA attributes
        copy_attr(self.ema, model, include, exclude)

def weights_init(net, init_type='normal', init_gain = 0.02):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and classname.find('Conv') != -1:
            if init_type == 'normal':
                torch.nn.init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                torch.nn.init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                torch.nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                torch.nn.init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
        elif classname.find('BatchNorm2d') != -1:
            torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
            torch.nn.init.constant_(m.bias.data, 0.0)
    print('initialize network with %s type' % init_type)
    net.apply(init_func)

def get_lr_scheduler(lr_decay_type, lr, min_lr, total_iters, warmup_iters_ratio = 0.05, warmup_lr_ratio = 0.1, no_aug_iter_ratio = 0.05, step_num = 10):
    def yolox_warm_cos_lr(lr, min_lr, total_iters, warmup_total_iters, warmup_lr_start, no_aug_iter, iters):
        if iters <= warmup_total_iters:   # iters即为epoch,如果当前epoch<warmup的位置，则热身训练
            # lr = (lr - warmup_lr_start) * iters / float(warmup_total_iters) + warmup_lr_start
            lr = (lr - warmup_lr_start) * pow(iters / float(warmup_total_iters), 2) + warmup_lr_start
        elif iters >= total_iters - no_aug_iter:
            lr = min_lr
        else:
            lr = min_lr + 0.5 * (lr - min_lr) * (
                1.0 + math.cos(math.pi* (iters - warmup_total_iters) / (total_iters - warmup_total_iters - no_aug_iter))
            )
        return lr

    def step_lr(lr, decay_rate, step_size, iters):
        if step_size < 1:
            raise ValueError("step_size must above 1.")
        n       = iters // step_size
        out_lr  = lr * decay_rate ** n
        return out_lr

    if lr_decay_type == "cos":
        warmup_total_iters  = min(max(warmup_iters_ratio * total_iters, 1), 3)
        warmup_lr_start     = max(warmup_lr_ratio * lr, 1e-6)
        no_aug_iter         = min(max(no_aug_iter_ratio * total_iters, 1), 15)
        func = partial(yolox_warm_cos_lr ,lr, min_lr, total_iters, warmup_total_iters, warmup_lr_start, no_aug_iter)
    else:
        decay_rate  = (min_lr / lr) ** (1 / (step_num - 1))
        step_size   = total_iters / step_num
        func = partial(step_lr, lr, decay_rate, step_size)

    return func

def set_optimizer_lr(optimizer, lr_scheduler_func, epoch):
    lr = lr_scheduler_func(epoch)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
