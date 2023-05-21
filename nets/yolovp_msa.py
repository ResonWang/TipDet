import copy
import math
from loguru import logger
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .darknet import BaseConv, CSPDarknet, CSPLayer, DWConv
from yoloV.utils.box_op import box_cxcywh_to_xyxy, generalized_box_iou
from yoloV.utils import bboxes_iou
from yoloV.models.post_trans import MSA_yolov, Native_MSA_block, \
    MSA_yolov_coord, get_Attn, MSA_yolov_Mutual_ViT, \
    MSA_yolov_Native_ViT, MSA_yolov_Mutual_ViT_parallel

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


class YOLOXHead_PriorBox_RegionalMSA1_OF_KeyFrame_Gray_or_OF_fineFeat(nn.Module):
    def __init__(
            self,
            num_classes,
            width=1.0,
            features="234",
            act="silu",
            depthwise=False,
            heads=1,
            drop=0.0,
            defualt_p=10,  # 每帧用于Attention的预测框数量，即每帧从预测结果从只选择Top defualt_p个，然后对这10个进行VID再分类
            sim_thresh=0.75,
            pre_nms=0.75,
            ave=False,
            defulat_pre=750,
            group_num=4,
    ):
        """
        Args:
            act (str): activation type of conv. Defalut value: "silu".
            depthwise (bool): whether apply depthwise conv in conv branch. Defalut value: False.
        """
        super().__init__()

        if features == "123":
            channel_factor = 0.5 * 0.5 * 2  # 因为融合了光流通道
            strides = [2, 4, 8]
            in_channels = [int(256 * channel_factor), int(512 * channel_factor), int(1024 * channel_factor)]
        if features == "234":
            channel_factor = 0.5 * 2
            strides = [4, 8, 16]
            in_channels = [int(256 * channel_factor), int(512 * channel_factor), int(1024 * channel_factor)]
        if features == "345":
            channel_factor = 1 * 2
            strides = [8, 16, 32]
            in_channels = [int(256 * channel_factor), int(512 * channel_factor), int(1024 * channel_factor)]
        if features == "23":
            self.in_features = ("dark2", "dark3")
            channel_factor = 0.5 * 2
            strides = [4, 8]
            in_channels = [int(256 * channel_factor), int(512 * channel_factor)]
        if features == "2":
            self.in_features = ("dark2")
            channel_factor = 0.5 * 2
            strides = [4]
            in_channels = [int(256 * channel_factor)]
        if features == "3":
            self.in_features = ("dark3")
            channel_factor = 1 * 2
            strides = [8]
            in_channels = [int(256 * channel_factor)]
        if features == "34":
            self.in_features = ("dark3", "dark4")
            channel_factor = 1 * 2
            strides = [8, 16]
            in_channels = [int(256 * channel_factor), int(512 * channel_factor)]

        self.Afternum = defualt_p
        self.Prenum = defulat_pre
        self.simN = defualt_p  # 用于计算MSA的单帧预测框数量
        self.nms_thresh = pre_nms
        self.n_anchors = 1
        self.num_classes = num_classes
        self.decode_in_inference = True  # for deploy, set to False
        self.group_num = group_num

        self.reg_convs = nn.ModuleList()
        self.cls_convs = nn.ModuleList()
        self.cls_convs2 = nn.ModuleList()

        # head
        self.reg_preds = nn.ModuleList()
        self.obj_preds = nn.ModuleList()
        self.cls_preds = nn.ModuleList()

        self.width = int(256 * width)
        self.trans = MSA_yolov(dim=self.width, out_dim=4 * self.width, num_heads=heads,
                               attn_drop=drop)  # 这里heads一定要能被self.width整除
        self.trans_coord = MSA_yolov_coord(dim=self.width, out_dim=4 * self.width, num_heads=heads,
                                           attn_drop=drop)  # 这里heads一定要能被self.width整除
        self.mutualMSA = MSA_yolov_Mutual_ViT(dim=self.width, depth=1, heads=4, dim_head=self.width, mlp_dim=self.width,
                                              dropout=0.5)
        self.mutualMSA_parallel = MSA_yolov_Mutual_ViT_parallel(dim=self.width, depth=1, heads=4, dim_head=self.width, mlp_dim=self.width, dropout=0.5)    # 无位置编码
        self.NativeMSA = MSA_yolov_Native_ViT(dim=self.width, depth=2, heads=1, dim_head=self.width, mlp_dim=self.width,
                                              dropout=0.5)  # heads=1 or 4
        self.native_trans = Native_MSA_block(depth=4, dim=self.width, num_heads=heads, attn_drop=drop)
        self.get_coord_attn = get_Attn(dim=3, num_heads=heads, attn_drop=drop)
        self.stems = nn.ModuleList()
        self.coordConv64 = BaseConv(
            in_channels=3,
            out_channels=64,
            ksize=1,
            stride=1,
            act=act)
        self.coordConv128 = BaseConv(
            in_channels=3,
            out_channels=128,
            ksize=1,
            stride=1,
            act=act)
        self.gray_128to64 = BaseConv(
            in_channels=128,
            out_channels=64,
            ksize=1,
            stride=1,
            act=act)
        self.coord_fc = nn.Linear(3, self.width)  # Mlp(in_features=3,hidden_features=64)
        self.OF_fc = nn.Linear(self.width, 1)  # Mlp(in_features=3,hidden_features=64)
        self.linear_pred = nn.Linear(64+64+64, 1)  # Mlp(in_features=512,hidden_features=self.num_classes+1) 560
        self.linear1568to64 = nn.Linear(1568, 64)
        self.sim_thresh = sim_thresh
        self.ave = ave
        Conv = DWConv if depthwise else BaseConv

        for i in range(len(in_channels)):  # 遍历每一个FPN的特征层，
            self.stems.append(             # 1x1卷积, 将不同size的feat 的channel进行统一
                BaseConv(
                    in_channels=int(in_channels[i] * width),
                    out_channels=int(256 * width),
                    ksize=1,
                    stride=1,
                    act=act,
                )
            )
            if 1:
                self.cls_convs2.append(  # 2个3x3的卷积
                    nn.Sequential(
                        *[
                            Conv(
                                in_channels=int(256 * width),
                                out_channels=int(256 * width),
                                ksize=3,
                                stride=1,
                                act=act,
                            ),
                            Conv(
                                in_channels=int(256 * width),
                                out_channels=int(256 * width),
                                ksize=3,
                                stride=1,
                                act=act,
                            ),
                        ]
                    )
                )
                self.reg_convs.append(  # 2个3x3的卷积
                    nn.Sequential(
                        *[
                            Conv(
                                in_channels=int(256 * width),
                                out_channels=int(256 * width),
                                ksize=3,
                                stride=1,
                                act=act,
                            ),
                            Conv(
                                in_channels=int(256 * width),
                                out_channels=int(256 * width),
                                ksize=3,
                                stride=1,
                                act=act,
                            ),
                        ]
                    )
                )
                self.cls_convs.append(  # 2个3x3的卷积
                    nn.Sequential(
                        *[
                            Conv(
                                in_channels=int(256 * width),
                                out_channels=int(256 * width),
                                ksize=3,
                                stride=1,
                                act=act,
                            ),
                            Conv(
                                in_channels=int(256 * width),
                                out_channels=int(256 * width),
                                ksize=3,
                                stride=1,
                                act=act,
                            ),
                        ]
                    )
                )
                self.cls_preds.append(  # 分类卷积，输出30通道，为类别数
                    nn.Conv2d(
                        in_channels=int(256 * width),
                        out_channels=self.n_anchors * self.num_classes,
                        kernel_size=1,
                        stride=1,
                        padding=0,
                    )
                )
                self.reg_preds.append(  # 回归卷积，输出4通道，为边界框
                    nn.Conv2d(
                        in_channels=int(256 * width),
                        out_channels=4,
                        kernel_size=1,
                        stride=1,
                        padding=0,
                    )
                )
                self.obj_preds.append(  # 前进背景分类，输出1通道
                    nn.Conv2d(
                        in_channels=int(256 * width),
                        out_channels=self.n_anchors * 1,
                        kernel_size=1,
                        stride=1,
                        padding=0,
                    )
                )

        self.use_l1 = False
        self.l1_loss = nn.L1Loss(reduction="none")
        self.bcewithlog_loss = nn.BCEWithLogitsLoss(reduction="none")
        self.iou_loss = IOUloss(reduction="none")
        self.strides = strides  # [8,16,32]
        self.grids = [torch.zeros(1)] * len(in_channels)

        # self.initialize_biases(1e-2)

    def initialize_biases(self, prior_prob):
        for conv in self.cls_preds:
            b = conv.bias.view(self.n_anchors, -1)
            b.data.fill_(-math.log((1 - prior_prob) / prior_prob))
            conv.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)

        for conv in self.obj_preds:
            b = conv.bias.view(self.n_anchors, -1)
            b.data.fill_(-math.log((1 - prior_prob) / prior_prob))
            conv.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)

    def forward(self, xin_batch):
        print("The key codes will be disclosed once the paper is accepted!")


    def get_output_and_grid(self, output, k, stride, dtype):
        grid = self.grids[k]

        batch_size = output.shape[0]
        n_ch = 5  # + self.num_classes
        hsize, wsize = output.shape[-2:]
        if grid.shape[2:4] != output.shape[2:4]:
            yv, xv = torch.meshgrid([torch.arange(hsize), torch.arange(wsize)])
            grid = torch.stack((xv, yv), 2).view(1, 1, hsize, wsize, 2).type(dtype)
            self.grids[k] = grid

        output = output.view(batch_size, self.n_anchors, n_ch, hsize, wsize)
        output = output.permute(0, 1, 3, 4, 2).reshape(
            batch_size, self.n_anchors * hsize * wsize, -1
        )
        grid = grid.view(1, -1, 2).cuda()
        # print(output.device,grid.device)
        output[..., :2] = (output[..., :2] + grid) * stride
        output[..., 2:4] = torch.exp(output[..., 2:4]) * stride
        return output, grid

    def decode_outputs(self, outputs, dtype, flevel=0):
        grids = []
        strides = []
        for (hsize, wsize), stride in zip(self.hw, self.strides):
            yv, xv = torch.meshgrid([torch.arange(hsize), torch.arange(wsize)])
            grid = torch.stack((xv, yv), 2).view(1, -1, 2)
            grids.append(grid)
            shape = grid.shape[:2]
            strides.append(torch.full((*shape, 1), stride))

        grids = torch.cat(grids, dim=1).type(dtype)
        strides = torch.cat(strides, dim=1).type(dtype)

        outputs[..., :2] = (outputs[..., :2] + grids) * strides
        outputs[..., 2:4] = torch.exp(outputs[..., 2:4]) * strides
        return outputs

    def find_feature_score(self, features, idxs):
        """
        (gframe,5376取决于三个特征层的大小和通道数,128特征层通道数)
        """
        features_cls = []
        for i, feature in enumerate(features):                  # 遍历每一帧，选择30个预测框
            features_cls.append(feature[idxs[i][:self.simN]])
        features_cls = torch.cat(features_cls)
        return features_cls

    def find_feature_score_coord(self, features, idxs):
        """
        (gframe,5376取决于三个特征层的大小和通道数,128特征层通道数)
        """
        features_cls = []
        for i, feature in enumerate(features):                  # 遍历每一帧，选择30个预测框
            features_cls.append(feature[idxs[i][:self.simN]])
        features_cls = torch.cat(features_cls)

        return features_cls

    def get_losses(
            self,
            imgs,
            x_shifts,
            y_shifts,
            expanded_strides,
            labels,
            outputs,
            origin_preds,
            dtype,
            refined_cls,
            idx,
            pred_res,
    ):
        bbox_preds = outputs[:, :, :4]  # [batch, n_anchors_all, 4]
        obj_preds = outputs[:, :, 4].unsqueeze(-1)  # [batch, n_anchors_all, 1]
        cls_preds = outputs[:, :, 5:]  # [batch, n_anchors_all, n_cls]

        # calculate targets
        mixup = labels.shape[2] > 5
        if mixup:
            label_cut = labels[..., :5]
        else:
            label_cut = labels
        nlabel = (label_cut.sum(dim=2) > 0).sum(dim=1)  # number of objects

        total_num_anchors = outputs.shape[1]
        x_shifts = torch.cat(x_shifts, 1)  # [1, n_anchors_all]
        y_shifts = torch.cat(y_shifts, 1)  # [1, n_anchors_all]
        expanded_strides = torch.cat(expanded_strides, 1)
        if self.use_l1:
            origin_preds = torch.cat(origin_preds, 1)

        cls_targets = []
        reg_targets = []
        l1_targets = []
        obj_targets = []
        fg_masks = []
        ref_targets = []
        num_fg = 0.0
        num_gts = 0.0
        ref_masks = []
        for batch_idx in range(outputs.shape[0]):
            num_gt = int(nlabel[batch_idx])
            num_gts += num_gt
            if num_gt == 0:
                cls_target = outputs.new_zeros((0, self.num_classes))
                reg_target = outputs.new_zeros((0, 4))
                l1_target = outputs.new_zeros((0, 4))
                obj_target = outputs.new_zeros((total_num_anchors, 1))
                fg_mask = outputs.new_zeros(total_num_anchors).bool()
                ref_target = outputs.new_zeros((idx[batch_idx].shape[0], self.num_classes + 1))
                ref_target[:, -1] = 1

            else:
                gt_bboxes_per_image = labels[batch_idx, :num_gt, 1:5]
                gt_classes = labels[batch_idx, :num_gt, 0]  # [batch,120,class+xywh]
                bboxes_preds_per_image = bbox_preds[batch_idx]

                try:
                    (gt_matched_classes, fg_mask, pred_ious_this_matching, matched_gt_inds,
                     num_fg_img,) = self.get_assignments(  # noqa
                        batch_idx,
                        num_gt,
                        total_num_anchors,
                        gt_bboxes_per_image,
                        gt_classes,
                        bboxes_preds_per_image,
                        expanded_strides,
                        x_shifts,
                        y_shifts,
                        cls_preds,
                        bbox_preds,
                        obj_preds,
                        labels,
                        imgs,
                    )
                except RuntimeError:
                    logger.error(
                        "OOM RuntimeError is raised due to the huge memory cost during label assignment. \
                           CPU mode is applied in this batch. If you want to avoid this issue, \
                           try to reduce the batch size or image size."
                    )
                    torch.cuda.empty_cache()
                    (gt_matched_classes, fg_mask, pred_ious_this_matching, matched_gt_inds,
                     num_fg_img,) = self.get_assignments(  # noqa
                        batch_idx,
                        num_gt,
                        total_num_anchors,
                        gt_bboxes_per_image,
                        gt_classes,
                        bboxes_preds_per_image,
                        expanded_strides,
                        x_shifts,
                        y_shifts,
                        cls_preds,
                        bbox_preds,
                        obj_preds,
                        labels,
                        imgs,
                        "cpu",
                    )

                torch.cuda.empty_cache()
                num_fg += num_fg_img

                cls_target_onehot = F.one_hot(
                    gt_matched_classes.to(torch.int64), self.num_classes
                )
                cls_target = cls_target_onehot * pred_ious_this_matching.unsqueeze(-1)
                fg_idx = torch.where(fg_mask)[0]

                obj_target = fg_mask.unsqueeze(-1)
                reg_target = gt_bboxes_per_image[matched_gt_inds]
                if self.use_l1:
                    l1_target = self.get_l1_target(
                        outputs.new_zeros((num_fg_img, 4)),
                        gt_bboxes_per_image[matched_gt_inds],
                        expanded_strides[0][fg_mask],
                        x_shifts=x_shifts[0][fg_mask],
                        y_shifts=y_shifts[0][fg_mask],
                    )
                ref_target = outputs.new_zeros((idx[batch_idx].shape[0], self.num_classes + 1))
                fg = 0

                gt_xyxy = box_cxcywh_to_xyxy(torch.tensor(reg_target))
                pred_box = pred_res[batch_idx][:, :4]
                cost_giou, iou = generalized_box_iou(pred_box, gt_xyxy)
                max_iou = torch.max(iou, dim=-1)
                for ele_idx, ele in enumerate(idx[batch_idx]):
                    loc = torch.where(fg_idx == ele)[0]

                    if len(loc):
                        ref_target[ele_idx, :self.num_classes] = cls_target[loc, :]
                        fg += 1
                        continue
                    if max_iou.values[ele_idx] >= 0.6:
                        max_idx = int(max_iou.indices[ele_idx])

                        ref_target[ele_idx, :self.num_classes] = cls_target_onehot[max_idx, :] * max_iou.values[ele_idx]
                        fg += 1
                    else:
                        ref_target[ele_idx, -1] = 1 - max_iou.values[ele_idx]

            cls_targets.append(cls_target)
            reg_targets.append(reg_target)
            obj_targets.append(obj_target.to(dtype))
            fg_masks.append(fg_mask)
            ref_targets.append(ref_target[:, :self.num_classes])
            ref_masks.append(ref_target[:, -1] == 0)
            if self.use_l1:
                l1_targets.append(l1_target)

        cls_targets = torch.cat(cls_targets, 0)
        reg_targets = torch.cat(reg_targets, 0)
        obj_targets = torch.cat(obj_targets, 0)
        ref_targets = torch.cat(ref_targets, 0)

        fg_masks = torch.cat(fg_masks, 0)
        ref_masks = torch.cat(ref_masks, 0)
        # print(sum(ref_masks)/ref_masks.shape[0])
        if self.use_l1:
            l1_targets = torch.cat(l1_targets, 0)

        num_fg = max(num_fg, 1)
        loss_iou = (
                       self.iou_loss(bbox_preds.view(-1, 4)[fg_masks], reg_targets)
                   ).sum() / num_fg
        loss_obj = (
                       self.bcewithlog_loss(obj_preds.view(-1, 1), obj_targets)
                   ).sum() / num_fg
        loss_cls = (
                       self.bcewithlog_loss(
                           cls_preds.view(-1, self.num_classes)[fg_masks], cls_targets
                       )
                   ).sum() / num_fg
        loss_ref = (
                       self.bcewithlog_loss(
                           refined_cls.view(-1, self.num_classes)[ref_masks], ref_targets[ref_masks]
                       )
                   ).sum() / num_fg

        if self.use_l1:
            loss_l1 = (
                          self.l1_loss(origin_preds.view(-1, 4)[fg_masks], l1_targets)
                      ).sum() / num_fg
        else:
            loss_l1 = 0.0

        reg_weight = 3.0
        ref_weight = 0  # 2
        loss = reg_weight * loss_iou + loss_obj + ref_weight * loss_ref + loss_l1 + loss_cls

        return (
            loss,
            reg_weight * loss_iou,
            loss_obj,
            loss_cls,  # ref_weight * loss_ref,
            loss_l1,
            num_fg / max(num_gts, 1),
        )

    def get_l1_target(self, l1_target, gt, stride, x_shifts, y_shifts, eps=1e-8):
        l1_target[:, 0] = gt[:, 0] / stride - x_shifts
        l1_target[:, 1] = gt[:, 1] / stride - y_shifts
        l1_target[:, 2] = torch.log(gt[:, 2] / stride + eps)
        l1_target[:, 3] = torch.log(gt[:, 3] / stride + eps)
        return l1_target

    @torch.no_grad()
    def get_assignments(
            self,
            batch_idx,
            num_gt,
            total_num_anchors,
            gt_bboxes_per_image,
            gt_classes,
            bboxes_preds_per_image,
            expanded_strides,
            x_shifts,
            y_shifts,
            cls_preds,
            bbox_preds,
            obj_preds,
            labels,
            imgs,
            mode="gpu",
    ):

        if mode == "cpu":
            print("------------CPU Mode for This Batch-------------")
            gt_bboxes_per_image = gt_bboxes_per_image.cpu().float()
            bboxes_preds_per_image = bboxes_preds_per_image.cpu().float()
            gt_classes = gt_classes.cpu().float()
            expanded_strides = expanded_strides.cpu().float()
            x_shifts = x_shifts.cpu()
            y_shifts = y_shifts.cpu()

        fg_mask, is_in_boxes_and_center = self.get_in_boxes_info(
            gt_bboxes_per_image,
            expanded_strides,
            x_shifts,
            y_shifts,
            total_num_anchors,
            num_gt,
        )

        bboxes_preds_per_image = bboxes_preds_per_image[fg_mask]
        cls_preds_ = cls_preds[batch_idx][fg_mask]
        obj_preds_ = obj_preds[batch_idx][fg_mask]
        num_in_boxes_anchor = bboxes_preds_per_image.shape[0]

        if mode == "cpu":
            gt_bboxes_per_image = gt_bboxes_per_image.cpu()
            bboxes_preds_per_image = bboxes_preds_per_image.cpu()

        pair_wise_ious = bboxes_iou(gt_bboxes_per_image, bboxes_preds_per_image, False)

        gt_cls_per_image = (
            F.one_hot(gt_classes.to(torch.int64), self.num_classes)
                .float()
                .unsqueeze(1)
                .repeat(1, num_in_boxes_anchor, 1)
        )
        pair_wise_ious_loss = -torch.log(pair_wise_ious + 1e-8)

        if mode == "cpu":
            cls_preds_, obj_preds_ = cls_preds_.cpu(), obj_preds_.cpu()

        with torch.cuda.amp.autocast(enabled=False):
            cls_preds_ = (
                    cls_preds_.float().unsqueeze(0).repeat(num_gt, 1, 1).sigmoid_()
                    * obj_preds_.unsqueeze(0).repeat(num_gt, 1, 1).sigmoid_()
            )
            pair_wise_cls_loss = F.binary_cross_entropy(
                cls_preds_.sqrt_(), gt_cls_per_image, reduction="none"
            ).sum(-1)
        del cls_preds_

        cost = (
                pair_wise_cls_loss
                + 3.0 * pair_wise_ious_loss
                + 100000.0 * (~is_in_boxes_and_center)
        )

        (
            num_fg,
            gt_matched_classes,
            pred_ious_this_matching,
            matched_gt_inds,
        ) = self.dynamic_k_matching(cost, pair_wise_ious, gt_classes, num_gt, fg_mask)
        del pair_wise_cls_loss, cost, pair_wise_ious, pair_wise_ious_loss

        if mode == "cpu":
            gt_matched_classes = gt_matched_classes.cuda()
            fg_mask = fg_mask.cuda()
            pred_ious_this_matching = pred_ious_this_matching.cuda()
            matched_gt_inds = matched_gt_inds.cuda()

        return (
            gt_matched_classes,
            fg_mask,
            pred_ious_this_matching,
            matched_gt_inds,
            num_fg,
        )

    def get_in_boxes_info(
            self,
            gt_bboxes_per_image,
            expanded_strides,
            x_shifts,
            y_shifts,
            total_num_anchors,
            num_gt,
    ):
        expanded_strides_per_image = expanded_strides[0]
        x_shifts_per_image = x_shifts[0] * expanded_strides_per_image
        y_shifts_per_image = y_shifts[0] * expanded_strides_per_image
        x_centers_per_image = (
            (x_shifts_per_image + 0.5 * expanded_strides_per_image)
                .unsqueeze(0)
                .repeat(num_gt, 1)
        )  # [n_anchor] -> [n_gt, n_anchor]
        y_centers_per_image = (
            (y_shifts_per_image + 0.5 * expanded_strides_per_image)
                .unsqueeze(0)
                .repeat(num_gt, 1)
        )

        gt_bboxes_per_image_l = (
            (gt_bboxes_per_image[:, 0] - 0.5 * gt_bboxes_per_image[:, 2])
                .unsqueeze(1)
                .repeat(1, total_num_anchors)
        )
        gt_bboxes_per_image_r = (
            (gt_bboxes_per_image[:, 0] + 0.5 * gt_bboxes_per_image[:, 2])
                .unsqueeze(1)
                .repeat(1, total_num_anchors)
        )
        gt_bboxes_per_image_t = (
            (gt_bboxes_per_image[:, 1] - 0.5 * gt_bboxes_per_image[:, 3])
                .unsqueeze(1)
                .repeat(1, total_num_anchors)
        )
        gt_bboxes_per_image_b = (
            (gt_bboxes_per_image[:, 1] + 0.5 * gt_bboxes_per_image[:, 3])
                .unsqueeze(1)
                .repeat(1, total_num_anchors)
        )

        b_l = x_centers_per_image - gt_bboxes_per_image_l
        b_r = gt_bboxes_per_image_r - x_centers_per_image
        b_t = y_centers_per_image - gt_bboxes_per_image_t
        b_b = gt_bboxes_per_image_b - y_centers_per_image
        bbox_deltas = torch.stack([b_l, b_t, b_r, b_b], 2)

        is_in_boxes = bbox_deltas.min(dim=-1).values > 0.0
        is_in_boxes_all = is_in_boxes.sum(dim=0) > 0
        # in fixed center

        center_radius = 4.5

        gt_bboxes_per_image_l = (gt_bboxes_per_image[:, 0]).unsqueeze(1).repeat(
            1, total_num_anchors
        ) - center_radius * expanded_strides_per_image.unsqueeze(0)
        gt_bboxes_per_image_r = (gt_bboxes_per_image[:, 0]).unsqueeze(1).repeat(
            1, total_num_anchors
        ) + center_radius * expanded_strides_per_image.unsqueeze(0)
        gt_bboxes_per_image_t = (gt_bboxes_per_image[:, 1]).unsqueeze(1).repeat(
            1, total_num_anchors
        ) - center_radius * expanded_strides_per_image.unsqueeze(0)
        gt_bboxes_per_image_b = (gt_bboxes_per_image[:, 1]).unsqueeze(1).repeat(
            1, total_num_anchors
        ) + center_radius * expanded_strides_per_image.unsqueeze(0)

        c_l = x_centers_per_image - gt_bboxes_per_image_l
        c_r = gt_bboxes_per_image_r - x_centers_per_image
        c_t = y_centers_per_image - gt_bboxes_per_image_t
        c_b = gt_bboxes_per_image_b - y_centers_per_image
        center_deltas = torch.stack([c_l, c_t, c_r, c_b], 2)
        is_in_centers = center_deltas.min(dim=-1).values > 0.0
        is_in_centers_all = is_in_centers.sum(dim=0) > 0

        # in boxes and in centers
        is_in_boxes_anchor = is_in_boxes_all | is_in_centers_all

        is_in_boxes_and_center = (
                is_in_boxes[:, is_in_boxes_anchor] & is_in_centers[:, is_in_boxes_anchor]
        )
        return is_in_boxes_anchor, is_in_boxes_and_center

    def dynamic_k_matching(self, cost, pair_wise_ious, gt_classes, num_gt, fg_mask):
        # Dynamic K
        # ---------------------------------------------------------------
        matching_matrix = torch.zeros_like(cost)

        ious_in_boxes_matrix = pair_wise_ious
        n_candidate_k = min(10, ious_in_boxes_matrix.size(1))
        topk_ious, _ = torch.topk(ious_in_boxes_matrix, n_candidate_k, dim=1)
        dynamic_ks = torch.clamp(topk_ious.sum(1).int(), min=1)
        for gt_idx in range(num_gt):
            _, pos_idx = torch.topk(
                cost[gt_idx], k=dynamic_ks[gt_idx].item(), largest=False
            )
            matching_matrix[gt_idx][pos_idx] = 1.0

        del topk_ious, dynamic_ks, pos_idx

        anchor_matching_gt = matching_matrix.sum(0)
        if (anchor_matching_gt > 1).sum() > 0:
            _, cost_argmin = torch.min(cost[:, anchor_matching_gt > 1], dim=0)
            matching_matrix[:, anchor_matching_gt > 1] *= 0.0
            matching_matrix[cost_argmin, anchor_matching_gt > 1] = 1.0
        fg_mask_inboxes = matching_matrix.sum(0) > 0.0
        num_fg = fg_mask_inboxes.sum().item()

        fg_mask[fg_mask.clone()] = fg_mask_inboxes

        matched_gt_inds = matching_matrix[:, fg_mask_inboxes].argmax(0)
        gt_matched_classes = gt_classes[matched_gt_inds]

        pred_ious_this_matching = (matching_matrix * pair_wise_ious).sum(0)[
            fg_mask_inboxes
        ]
        return num_fg, gt_matched_classes, pred_ious_this_matching, matched_gt_inds

    def postpro_woclass(self, prediction, nms_thre=0.75, topK=75, features=None):
        # find topK predictions, play the same role as RPN
        '''

        Args:
            prediction: [batch,feature_num,5+clsnum]
            num_classes:
            conf_thre:
            conf_thre_high:
            nms_thre:

        Returns:
            [batch,topK,5+clsnum]
        '''
        self.topK = topK
        box_corner = prediction.new(prediction.shape)
        box_corner[:, :, 0] = prediction[:, :, 0] - prediction[:, :, 2] / 2    # x1
        box_corner[:, :, 1] = prediction[:, :, 1] - prediction[:, :, 3] / 2    # y1
        box_corner[:, :, 2] = prediction[:, :, 0] + prediction[:, :, 2] / 2    # x2
        box_corner[:, :, 3] = prediction[:, :, 1] + prediction[:, :, 3] / 2    # y2
        prediction[:, :, :4] = box_corner[:, :, :4]
        output = [None for _ in range(len(prediction))]
        output_index = [None for _ in range(len(prediction))]
        features_list = []
        for i, image_pred in enumerate(prediction):

            if not image_pred.size(0):
                continue

            # Detections ordered as (x1, y1, x2, y2, obj_conf, class_conf, class_pred)
            detections = image_pred[:, :]

            conf_score = image_pred[:, 4] * image_pred[:, 5]
            top_pre = torch.topk(conf_score, k=self.Prenum)  # 750
            sort_idx = top_pre.indices[:self.Prenum]
            detections_temp = detections[sort_idx, :]
            nms_out_index = torchvision.ops.nms(
                detections_temp[:, :4],
                detections_temp[:, 4] * detections_temp[:, 5],
                nms_thre,
            )

            topk_idx = sort_idx[nms_out_index[:self.topK]]
            output[i] = detections[topk_idx, :]
            output_index[i] = topk_idx

        return output, output_index

    def find_index_for_spatialTemporal_MSA_and_fetch(self, f1_feat, f2_feat, indices):
        """

        :param f1_feat_flatten:  (4,6400,80,80) cls_feat1+coordAdd
        :param f2_feat_flatten:  (4,1600,40,40) cls_feat2++coordAdd
        :param index: 8000中的indices N
        :return: list 4 -- 4xlist[10]---10*(4x25,128) 10个框，每个框和自身周围4x25个特征进行MSA
        """
        f1_size = 4  # 半边长
        f2_size = 4

        feat = []
        for j in range(len(indices)):  # 遍历每一帧
            feat_this_frame = []
            for i in indices[j]:  # 遍历每一帧的框，每个框对应(4x81,128)
                if i < f1_feat.shape[3] * f1_feat.shape[2]:  # (80,80)  f1中
                    index_array_row = i // f1_feat.shape[3]
                    index_array_col = i % f1_feat.shape[2]
                    this_ind_feat = f1_feat[:, :, max(index_array_row - f1_size, 0): min(index_array_row + f1_size + 1, f1_feat.shape[2]),
                                    max(index_array_col - f1_size, 0): min(index_array_col + f1_size + 1, f1_feat.shape[2])]  # (4,128,9,9)
                    this_ind_feat = this_ind_feat.flatten(start_dim=2)  # (4,128,9,9)-->(4,128,81)
                    if this_ind_feat.shape[-1] < (2 * f1_size + 1) ** 2:
                        this_ind_feat = torch.cat([this_ind_feat, this_ind_feat[:, :, -1].unsqueeze(-1).repeat(1, 1, (
                                    2 * f1_size + 1) ** 2 - this_ind_feat.shape[-1])], -1)  # 重复，对齐点的数量
                    feat_this_frame.append(this_ind_feat.permute(0, 2, 1).flatten(start_dim=0,end_dim=1))  # (4,128,81)-->(4,81,128)-->(4x81,128) 一行接一行拉平

                else:  # (40,40)  f2中
                    index_array_row = (i - f1_feat.shape[3] * f1_feat.shape[2]) // f2_feat.shape[3]
                    index_array_col = (i - f1_feat.shape[3] * f1_feat.shape[2]) % f2_feat.shape[2]
                    this_ind_feat = f2_feat[:, :, max(index_array_row - f2_size, 0): min(index_array_row + f2_size + 1,
                                                                                         f2_feat.shape[2]),
                                    max(index_array_col - f2_size, 0): min(index_array_col + f2_size + 1,
                                                                           f2_feat.shape[2])]  # (4,128,3,3)
                    this_ind_feat = this_ind_feat.flatten(start_dim=2)  # (4,128,5,5)-->(4,128,25)
                    if this_ind_feat.shape[-1] < (2 * f2_size + 1) ** 2:
                        this_ind_feat = torch.cat([this_ind_feat, this_ind_feat[:, :, -1].unsqueeze(-1).repeat(1, 1, (
                                    2 * f2_size + 1) ** 2 - this_ind_feat.shape[-1])], -1)  # 重复，对齐点的数量
                    feat_this_frame.append(this_ind_feat.permute(0, 2, 1).flatten(start_dim=0,
                                                                                  end_dim=1))  # (4,128,25)-->(4,25,128)-->(4x25,128)
            feat.append(feat_this_frame)
        return feat

    def find_index_for_dark2(self, f1_feat, indices):
        """
        :param f1_feat_flatten:  (1,16,320,320) cls_feat1+coordAdd
        :param index: 8000中的indices N
        :return: (10,16x9) 10个框，16每个点通道数，9周围点数
        """
        f1_size = 1  # 半边长
        f2_size = 1

        j = -1
        feat_this_frame = []
        for i in indices[j]:  # 遍历最后一帧的框，每个框对应(16x9,16)
            if i < 80 * 80:  # (80,80)  f1中
                index_array_row = i // 80 * 2   # 在dark1中的位置
                index_array_col = i % 80 * 2
                this_ind_feat = f1_feat[-1, :, max(index_array_row - f1_size, 0): min(index_array_row + f1_size + 1, f1_feat.shape[2]),
                                max(index_array_col - f1_size, 0): min(index_array_col + f1_size + 1, f1_feat.shape[2])]  # (16,3,3)
                this_ind_feat = this_ind_feat.flatten(start_dim=1)  # (16,3,3)-->(16,9)
                if this_ind_feat.shape[-1] < (2 * f1_size + 1) ** 2:
                    this_ind_feat = torch.cat([this_ind_feat, this_ind_feat[:, -1].unsqueeze(-1).repeat(1, (2 * f2_size + 1) ** 2 - this_ind_feat.shape[-1])], -1)  # 重复，对齐点的数量
                feat_this_frame.append(this_ind_feat.permute(1, 0).flatten(start_dim=0,end_dim=1).unsqueeze(0))  # (16,9)-->(9,16)-->(16x9,)
            else:  # (40,40)  f2中
                index_array_row = (i - 80 * 80) // 40 * 4  # 在dark1中的位置
                index_array_col = (i - 80 * 80) % 40 * 4
                this_ind_feat = f1_feat[-1, :, max(index_array_row - f2_size, 0): min(index_array_row + f2_size + 1, f1_feat.shape[2]),
                                max(index_array_col - f2_size, 0): min(index_array_col + f2_size + 1, f1_feat.shape[2])]  # (16,3,3)
                this_ind_feat = this_ind_feat.flatten(start_dim=1)  # (16,3,3)-->(16,9)
                if this_ind_feat.shape[-1] < (2 * f2_size + 1) ** 2:
                    this_ind_feat = torch.cat([this_ind_feat, this_ind_feat[:, -1].unsqueeze(-1).repeat(1, (2 * f2_size + 1) ** 2 - this_ind_feat.shape[-1])], -1)  # 重复，对齐点的数量
                feat_this_frame.append(this_ind_feat.permute(1,0).flatten(start_dim=0, end_dim=1).unsqueeze(0))  # (16,9)-->(9,16)-->(16x9,)-->(1,16x9)
        feat_this_frame = torch.cat(feat_this_frame, 0)  # (10,16x9)
        return feat_this_frame

    def find_index_for_dark1(self, f1_feat, indices):
        """

        :param f1_feat_flatten:  (1,16,320,320) cls_feat1+coordAdd
        :param index: 8000中的indices N
        :return: (10,16x9) 10个框，16每个点通道数，9周围点数
        """
        f1_size = 1  # 半边长
        f2_size = 1

        j = -1
        feat_this_frame = []
        for i in indices[j]:  # 遍历最后一帧的框，每个框对应(16x9,16)
            if i < 80 * 80:  # (80,80)  f1中
                index_array_row = i // 80 * 4   # 在dark1中的位置
                index_array_col = i % 80 * 4
                this_ind_feat = f1_feat[-1, :, max(index_array_row - f1_size, 0): min(index_array_row + f1_size + 1, f1_feat.shape[2]),
                                max(index_array_col - f1_size, 0): min(index_array_col + f1_size + 1, f1_feat.shape[2])]  # (16,3,3)
                this_ind_feat = this_ind_feat.flatten(start_dim=1)  # (16,3,3)-->(16,9)
                if this_ind_feat.shape[-1] < (2 * f1_size + 1) ** 2:
                    this_ind_feat = torch.cat([this_ind_feat, this_ind_feat[:, -1].unsqueeze(-1).repeat(1, (2 * f2_size + 1) ** 2 - this_ind_feat.shape[-1])], -1)  # 重复，对齐点的数量
                feat_this_frame.append(this_ind_feat.permute(1, 0).flatten(start_dim=0,end_dim=1).unsqueeze(0))  # (16,9)-->(9,16)-->(16x9,)

            else:  # (40,40)  f2中
                index_array_row = (i - 80 * 80) // 40 * 8  # 在dark1中的位置
                index_array_col = (i - 80 * 80) % 40 * 8
                this_ind_feat = f1_feat[-1, :, max(index_array_row - f2_size, 0): min(index_array_row + f2_size + 1, f1_feat.shape[2]),
                                max(index_array_col - f2_size, 0): min(index_array_col + f2_size + 1, f1_feat.shape[2])]  # (16,3,3)
                this_ind_feat = this_ind_feat.flatten(start_dim=1)  # (16,3,3)-->(16,9)
                if this_ind_feat.shape[-1] < (2 * f2_size + 1) ** 2:
                    this_ind_feat = torch.cat([this_ind_feat, this_ind_feat[:, -1].unsqueeze(-1).repeat(1, (2 * f2_size + 1) ** 2 - this_ind_feat.shape[-1])], -1)  # 重复，对齐点的数量
                feat_this_frame.append(this_ind_feat.permute(1,0).flatten(start_dim=0, end_dim=1).unsqueeze(0))  # (16,9)-->(9,16)-->(16x9,)-->(1,16x9)
        feat_this_frame = torch.cat(feat_this_frame, 0)  # (10,16x9)
        return feat_this_frame

    def find_index_for_dark1_ROIPooling(self, f1_feat, indices, results):
        """
        :param f1_feat_flatten:  (1,32,320,320) cls_feat1+coordAdd
        :param index: 8000中的indices N
        :param results: list 4x(10,6)  (x1,y1,x2,y2,...) 640尺寸下的
        :return: (10,16x225) 10个框，16每个点通道数，225周围点数
        """
        coords = results[-1][:,0:4]/640   # (10,4) x1,y1,x2,y2
        feat_this_frame = []
        for i in range(coords.shape[0]):  # 遍历最后一帧的框，每个框对应(1x9,16)
            x1 = max(int(coords[i,0] * 320), 0)     # 列号
            y1 = max(int(coords[i,1] * 320), 0)     # 列号
            x2 = min(int(coords[i,2] * 320), 319)   # 行号
            y2 = min(int(coords[i,3] * 320), 319)   # 行号
            # tmp = self.show_detection_on_FeatMap(f1_feat, 0, x1, y1, x2, y2)
            this_ind_feat = f1_feat[-1, :, y1: y2, x1: x2]                 # (16,m,n)
            this_ind_feat = torch.nn.AdaptiveAvgPool2d(7)(this_ind_feat)  # (16,15,15)
            this_ind_feat = this_ind_feat.flatten(start_dim=1)  # (16,15,15)-->(16,225)
            feat_this_frame.append(this_ind_feat.permute(1, 0).flatten(start_dim=0,end_dim=1).unsqueeze(0))  # (16,225)-->(225,16)-->(16x225,)
        feat_this_frame = torch.cat(feat_this_frame, 0)  # (10,16x225)
        return feat_this_frame

    def find_index_for_dark2_ROIPooling(self, f1_feat, indices, results):
        """
        :param f1_feat_flatten:  (1,32,320,320) cls_feat1+coordAdd
        :param index: 8000中的indices N
        :param results: list4x(10,6)  (x1,y1,x2,y2,...) 640尺寸下的
        :return: (10,32x49) 10个框，16每个点通道数，49周围点数
        """
        coords = results[-1][:,0:4]/640   # x1,y1,x2,y2
        feat_this_frame = []
        for i in range(coords.shape[0]):  # 遍历最后一帧的框，每个框对应(1x9,16)
            x1 = max(int(coords[i,0] * 160), 0)     # 列号
            y1 = max(int(coords[i,1] * 160), 0)     # 列号
            x2 = min(int(coords[i,2] * 160), 159)   # 行号
            y2 = min(int(coords[i,3] * 160), 159)   # 行号
            # tmp = self.show_detection_on_FeatMap(f1_feat, 0, x1, y1, x2, y2)
            this_ind_feat = f1_feat[-1, :, y1: y2, x1: x2]                 # (32,m,n)
            this_ind_feat = torch.nn.AdaptiveAvgPool2d(7)(this_ind_feat)   # (32,7,7)  ROI pooling
            this_ind_feat = this_ind_feat.flatten(start_dim=1)             # (32,7,7)-->(16,49)
            feat_this_frame.append(this_ind_feat.permute(1, 0).flatten(start_dim=0,end_dim=1).unsqueeze(0))  # (32,49)-->(32,16)-->(32x49,)
        feat_this_frame = torch.cat(feat_this_frame, 0)  # (10,32x49)
        return feat_this_frame

    def show_detection_on_FeatMap(self, feat, channel_id, x1,y1,x2,y2):
        """
        :param feat:  (4,16,w,h)
        :param channel_id:
        :return:
        """
        import cv2
        array = feat.detach().cpu().numpy()[-1,channel_id]
        array = (array / array.max() * 255).astype(np.uint8)
        array_rgb = cv2.cvtColor(array, cv2.COLOR_GRAY2BGR)
        draw = cv2.rectangle(array_rgb, (x1, y1), (x2, y2), (0, 255, 0), 1)
        return draw

    def find_index_for_MSA_OF_Aware(self, f1_feat, f2_feat, indices, f1_OF, f2_OF):
        """
        对于一个group中的所有检测框，MSA的对象都是一样的，就是每帧中光流值最大的5个特征点，4帧一共就是20个
         f1_OF: (16,128,80,80)
         f2_OF: (16,64,40,40)
        :param 混合特征 f1_feat:  (16,128,80,80)
        :param 混合特征 f2_feat:  (16,64,40,40)
        :param index: 8000中的indices N  list[4x(10,)]
        :return: list 16 -- 16xlist[10]---10*(16x25,128) 10个框，每个框和自身周围16x25个特征进行MSA
        """
        num_per_frame = 5
        f1_of_sum = torch.sum(f1_OF, dim=1).flatten(start_dim=1, end_dim=2)   # sum: (16,80,80) flatten: (16,6400)
        f2_of_sum = torch.sum(f2_OF, dim=1).flatten(start_dim=1, end_dim=2)   # (16,40,40)  (16,1600)
        _, f1_of_indices = f1_of_sum.topk(num_per_frame, dim=1, largest=True, sorted=True)   # (16,10)  每帧的5个索引(在6400里)
        _, f2_of_indices = f2_of_sum.topk(num_per_frame, dim=1, largest=True, sorted=True)   # (16,10)  每帧的5个索引(在1600里)
        # print(f1_of_indices)
        f1_flatten = f1_feat.flatten(start_dim=2, end_dim=3).permute(0, 2, 1)   # (16,6400,128)
        f2_flatten = f2_feat.flatten(start_dim=2, end_dim=3).permute(0, 2, 1)   # (16,1600,128)

        f1 = []
        for i in range(f1_feat.shape[0]):
            for j in range(num_per_frame):
                f1.append(f1_flatten[i,f1_of_indices[i,j],...].unsqueeze(0))  # list 40: (1,128)
        f1 = torch.cat(f1,0)     # (40,128)

        f2 = []
        for i in range(f2_feat.shape[0]):
            for j in range(num_per_frame):
                f2.append(f2_flatten[i,f2_of_indices[i,j],...].unsqueeze(0))  # list 40: (1,128)
        f2 = torch.cat(f2,0)     # (40,128)

        feat = []
        for j in range(len(indices)):  # 遍历每一帧
            feat_this_frame = []
            for i in indices[j]:       # 遍历每一帧的框，每个框对应(4x5,128)
                if i < f1_feat.shape[3] * f1_feat.shape[2]:  # (80,80)  f1中
                    feat_this_frame.append(f1)  # (40,128)

                else:  # (40,40)  f2中
                    feat_this_frame.append(f2)  # (16,128,25)-->(16,25,128)-->(16x25,128)
            feat.append(feat_this_frame)
        return feat

    def find_index_for_MSA_OF_Aware2(self, f1_feat, f2_feat, indices, f1_OF, f2_OF):
        """
        对于一个group中的所有检测框，MSA的对象都是一样的，就是每帧中光流值最大的5个特征点，4帧一共就是20个. 不光只取5个点，还取其邻域
         f1_OF: (16,128,80,80)
         f2_OF: (16,64,40,40)
        :param 混合特征 f1_feat:  (16,128,80,80)
        :param 混合特征 f2_feat:  (16,64,40,40)
        :param index: 8000中的indices N  list[4x(10,)]
        :return: list 10 -- 10*(180,128) 最后一帧10个框，每个框和自身周围180(4x5x9)个光流选出来的特征进行MSA
        """
        r = 1
        num_per_frame = 5
        f1_of_sum = torch.sum(f1_OF, dim=1).flatten(start_dim=1, end_dim=2)   # sum: (4,80,80) flatten: (4,6400)
        f2_of_sum = torch.sum(f2_OF, dim=1).flatten(start_dim=1, end_dim=2)   # (4,40,40)  (4,1600)
        _, f1_of_indices = f1_of_sum.topk(num_per_frame, dim=1, largest=True, sorted=True)   # (4,5)  每帧的5个索引(在6400里)
        _, f2_of_indices = f2_of_sum.topk(num_per_frame, dim=1, largest=True, sorted=True)   # (4,5)  每帧的5个索引(在1600里)
        # print(f1_of_indices)
        # f1_flatten = f1_feat.flatten(start_dim=2, end_dim=3).permute(0, 2, 1)   # (4,6400,128)
        # f2_flatten = f2_feat.flatten(start_dim=2, end_dim=3).permute(0, 2, 1)   # (4,1600,128)

        f1 = []
        for i in range(f1_feat.shape[0]):   # 4个关键帧
            for j in range(num_per_frame):  # 每帧5个关键点，每个关键点取周围9个点
                index_array_row = f1_of_indices[i,j] // f1_feat.shape[3]
                index_array_col = f1_of_indices[i,j] % f1_feat.shape[2]
                temp = f1_feat[i, :, max(index_array_row - r, 0) : min(index_array_row + r + 1, f1_feat.shape[2]),
                                     max(index_array_col - r, 0) : min(index_array_col + r + 1, f1_feat.shape[2])]# (128,3,3)
                temp_flat = temp.flatten(start_dim=1)    # (128,9)
                if temp_flat.shape[-1] < (2 * r + 1) ** 2:
                    temp_flat = torch.cat([temp_flat,
                                           temp_flat[:, 0].unsqueeze(-1).repeat(1, (2 * r + 1) ** 2 - temp_flat.shape[-1])],  # (128, x)
                                           -1)  # 重复，对齐点的数量 --> (128,9)
                f1.append(temp_flat.permute(1, 0))     # list 4x5: (9,128)
        f1 = torch.cat(f1,0)     # (4x5x9,128)  (180,128)

        f2 = []
        for i in range(f2_feat.shape[0]):   # 4个关键帧
            for j in range(num_per_frame):  # 每帧5个关键点，每个关键点取周围9个点
                index_array_row = f2_of_indices[i,j] // f2_feat.shape[3]
                index_array_col = f2_of_indices[i,j] % f2_feat.shape[2]
                temp = f2_feat[i, :, max(index_array_row - r, 0) : min(index_array_row + r + 1, f2_feat.shape[2]),
                                     max(index_array_col - r, 0) : min(index_array_col + r + 1, f2_feat.shape[2])] # (128,3,3)
                temp_flat = temp.flatten(start_dim=1)    # (128,9)
                if temp_flat.shape[-1] < (2 * r + 1) ** 2:
                    temp_flat = torch.cat([temp_flat,
                                           temp_flat[:, 0].unsqueeze(-1).repeat(1, (2 * r + 1) ** 2 - temp_flat.shape[-1])],  # (128, x)
                                           -1)           # 重复，对齐点的数量 --> (128,9)
                f2.append(temp_flat.permute(1, 0))       # list 4x5: (9,128)
        f2 = torch.cat(f2,0)     # (4x5x9,128)  (180,128)

        feat_this_frame = []
        for i in indices[-1]:       # 遍历最后一帧的10个预测框，每个框对应(4x5,128)
            if i < f1_feat.shape[3] * f1_feat.shape[2]:  # (80,80)  预测框由f1中预测出
                feat_this_frame.append(f1)  # (180,128)

            else:                           # (40,40)  f2中
                feat_this_frame.append(f2)  # (180,128)

        return feat_this_frame   # (10,180,128)

    def postprocess_single_img(self, prediction, num_classes, conf_thre=0.001, nms_thre=0.5):

        output_ori = [None for _ in range(len(prediction))]
        prediction_ori = copy.deepcopy(prediction)
        for i, detections in enumerate(prediction):

            if not detections.size(0):
                continue

            detections_ori = prediction_ori[i]

            conf_mask = (detections_ori[:, 4] * detections_ori[:, 5] >= conf_thre).squeeze()
            detections_ori = detections_ori[conf_mask]
            nms_out_index = torchvision.ops.batched_nms(
                detections_ori[:, :4],
                detections_ori[:, 4] * detections_ori[:, 5],
                detections_ori[:, 6],
                nms_thre,
            )
            detections_ori = detections_ori[nms_out_index]
            output_ori[i] = detections_ori
        # print(output)
        return output_ori, output_ori

