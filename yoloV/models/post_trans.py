import cv2.cv2
from vit_pytorch.vit_for_small_dataset import PreNorm, FeedForward
import torch
import torch.nn as nn
from matplotlib import pyplot as plt
from yoloV.utils.box_op import box_cxcywh_to_xyxy, generalized_box_iou
import numpy as np
from einops import rearrange, repeat
from torch.autograd import Variable
import math

class Attention_msa(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., scale=25):
        # dim :input[batchsize,sequence length, input dimension]-->output[batchsize, sequence lenght, dim]
        # qkv_bias : Is it matter?
        # qk_scale, attn_drop,proj_drop will not be used
        # object = Attention(dim,num head)
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = scale  # qk_scale or head_dim ** -0.5

        self.qkv_cls = nn.Linear(dim, dim * 3, bias=qkv_bias)  # 128-->384  3的目的是对应分别分给q k v， 也就是q k v 都是由原向量导出来的
        self.qkv_reg = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)

    def forward(self, x_cls, x_reg, cls_score=None, fg_score=None, return_attention=False, ave=True, sim_thresh=0.75,
                use_mask=False):
        B, N, C = x_cls.shape  # 1, 480, 128

        qkv_cls = self.qkv_cls(x_cls).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)  # 3, B, num_head, N, c   (3,1,4,480,32)
        qkv_reg = self.qkv_reg(x_reg).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)    # (3,1,4,480,32)
        q_cls, k_cls, v_cls = qkv_cls[0], qkv_cls[1], qkv_cls[2]  # make torchscript happy (cannot use tensor as tuple) (3,1,4,480,32),(3,1,4,480,32),(3,1,4,480,32)
        q_reg, k_reg, v_reg = qkv_reg[0], qkv_reg[1], qkv_reg[2]  # (1,4,480,32),(1,4,480,32),(1,4,480,32)
        # 注意这里会出现除0
        q_cls = q_cls / (torch.norm(q_cls, dim=-1, keepdim=True) + torch.tensor([1e-8]).to('cuda')) # (1,4,480,32)  4就是4个头的意思，每个头处理32个维度，一共处理128维度，每个头都有自己的QKV
        k_cls = k_cls / (torch.norm(k_cls, dim=-1, keepdim=True) + torch.tensor([1e-8]).to('cuda')) # (1,4,480,32)
        q_reg = q_reg / (torch.norm(q_reg, dim=-1, keepdim=True) + torch.tensor([1e-8]).to('cuda'))  # (1,4,480,32)
        k_reg = k_reg / (torch.norm(k_reg, dim=-1, keepdim=True) + torch.tensor([1e-8]).to('cuda')) # (1,4,480,32)
        v_cls_normed = v_cls / (torch.norm(v_cls, dim=-1, keepdim=True)+ torch.tensor([1e-8]).to('cuda')) # (1,4,480,32)

        if cls_score == None:
            cls_score = 1
        else:
            cls_score = torch.reshape(cls_score, [1, 1, 1, -1]).repeat(1, self.num_heads, N, 1)   # (480,) 重复--> (1,4,480,480)

        if fg_score == None:
            fg_score = 1
        else:
            fg_score = torch.reshape(fg_score, [1, 1, 1, -1]).repeat(1, self.num_heads, N, 1)    # (480,) 重复--> (1,4,480,480)

        attn_cls_raw = v_cls_normed @ v_cls_normed.transpose(-2, -1)   # N(V)XN(V) (1,4,480,480) = (1,4,480,32) x (1,4,32,480)
        if use_mask:
            # only reference object with higher confidence..
            cls_score_mask = (cls_score > (cls_score.transpose(-2, -1) - 0.1)).type_as(cls_score)
            fg_score_mask = (fg_score > (fg_score.transpose(-2, -1) - 0.1)).type_as(fg_score)
        else:
            cls_score_mask = fg_score_mask = 1

        # cls_score_mask = (cls_score < (cls_score.transpose(-2, -1) + 0.1)).type_as(cls_score)
        # fg_score_mask = (fg_score < (fg_score.transpose(-2, -1) + 0.1)).type_as(fg_score)
        # visual_attention(cls_score[0, 0, :, :])
        # visual_attention(cls_score_mask[0,0,:,:])

        attn_cls = (q_cls @ k_cls.transpose(-2, -1)) * self.scale * cls_score * cls_score_mask   # (1,4,480,480) 标准的attention公式中的QKT/scale，乘K之前先乘上置信度 cls_score，即分类置信度(预测框所属的那个类别)作为K加权权重的一部分
        attn_cls = attn_cls.softmax(dim=-1)   # softmax() (1,4,480,480)
        attn_cls = self.attn_drop(attn_cls)   # dropout (1,4,480,480)

        attn_reg = (q_reg @ k_reg.transpose(-2, -1)) * self.scale * fg_score * fg_score_mask  # obj置信度作为reg attention的K加权权重的一部分
        attn_reg = attn_reg.softmax(dim=-1)   # softmax() (1,4,480,480)
        attn_reg = self.attn_drop(attn_reg)   # dropout (1,4,480,480)

        attn = (attn_reg + attn_cls) / 2      # (1,4,480,480) 回归特征和分类特征的平均后得到的attention权重，称为多头yolo回归分类相似度

        # 将同一帧里的预测框相关度置0
        attn_same_frame_set_zero = attn.detach().cpu().numpy()
        attn_same_frame_set_zero_ori = attn_same_frame_set_zero.copy()
        for i in range(attn_same_frame_set_zero.shape[1]):
            for j in range(attn_same_frame_set_zero.shape[2]):
                attn_same_frame_set_zero[0][i][j, j // 10 * 10:j // 10 * 10 + (10 - 1)] = 0
                attn_same_frame_set_zero[0][i][j, j] = attn_same_frame_set_zero_ori[0][i][j, j]

        attn = torch.from_numpy(attn_same_frame_set_zero).to(attn.device).to(attn.dtype)  # (1,4,160,160)

        x = (attn @ v_cls).transpose(1, 2).reshape(B, N, C)  # (1, 4, 480, 32)-->(1, 480, 4, 32)-->(1,480,128) #

        x_ori = v_cls.permute(0, 2, 1, 3).reshape(B, N, C)  # 原始的V，即cls特征向量(cls_feat2)直接一次fc过来的 (1,480,128)
        x_cls = torch.cat([x, x_ori], dim=-1)               # 拼接的V (1,480,256): (原始cls_feat2+1fc的)+(考虑了置信度的MSA后的)
        #

        if ave:  # 注意这里虽然提炼了一个预测框最终相似度，考虑了余弦相似度mask和考虑了置信度的MSA相似度， 但该相似度在x_cls中没有被使用，后面又进行了使用
            ones_matrix = torch.ones(attn.shape[2:]).to('cuda')   # (480,480)
            zero_matrix = torch.zeros(attn.shape[2:]).to('cuda')  # (480,480)

            attn_cls_raw = torch.sum(attn_cls_raw, dim=1, keepdim=False)[0] / self.num_heads  # (1,4,480,480) sum--> (1,480,480) [0]--> (480,480) /4--> (480,480) 4个头平均
            sim_mask = torch.where(attn_cls_raw > sim_thresh, ones_matrix, zero_matrix)       # 根据 attn_cls_raw 即N(V)XN(V)进行二值化 (480, 480)，称为余弦相似度mask
            sim_attn = torch.sum(attn, dim=1, keepdim=False)[0] / self.num_heads              # 特殊MSA得到的最终atteneion权重   (480, 480) 4个头平均， 称为考虑了置信度的MSA相似度

            sim_round2 = torch.softmax(sim_attn, dim=-1)  # 概率化 (480, 480) 480=16(seq中图像张数)x30(每张选择的预测框数量)
            sim_round2 = sim_mask * sim_round2 / ((torch.sum(sim_mask * sim_round2, dim=-1, keepdim=True)) + torch.tensor([1e-8]).to('cuda')) # 最终seq中任意两个预测框的相似度 (480,480),称为预测框最终相似度
            return x_cls, None, sim_round2  # 拼接的V (1,480,256): 原始+MSA后的
        else:
            return x_cls, None, None

class Attention_msa_coord(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., scale=25):
        # dim :input[batchsize,sequence length, input dimension]-->output[batchsize, sequence lenght, dim]
        # qkv_bias : Is it matter?
        # qk_scale, attn_drop,proj_drop will not be used
        # object = Attention(dim,num head)
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = scale  # qk_scale or head_dim ** -0.5

        self.qkv_cls = nn.Linear(dim, dim * 3, bias=qkv_bias)  # 128-->384  3的目的是对应分别分给q k v， 也就是q k v 都是由原向量导出来的
        self.qkv_reg = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)

    def forward(self, x_cls, x_reg, cls_score=None, fg_score=None, return_attention=False, ave=True, sim_thresh=0.75,
                use_mask=False, coord_attn=None):
        B, N, C = x_cls.shape  # 1, 480, 128

        qkv_cls = self.qkv_cls(x_cls).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)  # 3, B, num_head, N, c   (3,1,4,480,32)
        qkv_reg = self.qkv_reg(x_reg).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)    # (3,1,4,480,32)
        q_cls, k_cls, v_cls = qkv_cls[0], qkv_cls[1], qkv_cls[2]  # make torchscript happy (cannot use tensor as tuple) (3,1,4,480,32),(3,1,4,480,32),(3,1,4,480,32)
        q_reg, k_reg, v_reg = qkv_reg[0], qkv_reg[1], qkv_reg[2]  # (1,4,480,32),(1,4,480,32),(1,4,480,32)
        # 注意这里会出现除0
        q_cls = q_cls / (torch.norm(q_cls, dim=-1, keepdim=True) + torch.tensor([1e-8]).to('cuda')) # (1,4,480,32)  4就是4个头的意思，每个头处理32个维度，一共处理128维度，每个头都有自己的QKV
        k_cls = k_cls / (torch.norm(k_cls, dim=-1, keepdim=True) + torch.tensor([1e-8]).to('cuda')) # (1,4,480,32)
        q_reg = q_reg / (torch.norm(q_reg, dim=-1, keepdim=True) + torch.tensor([1e-8]).to('cuda'))  # (1,4,480,32)
        k_reg = k_reg / (torch.norm(k_reg, dim=-1, keepdim=True) + torch.tensor([1e-8]).to('cuda')) # (1,4,480,32)
        v_cls_normed = v_cls / (torch.norm(v_cls, dim=-1, keepdim=True)+ torch.tensor([1e-8]).to('cuda')) # (1,4,480,32)

        if cls_score == None:
            cls_score = 1
        else:
            cls_score = torch.reshape(cls_score, [1, 1, 1, -1]).repeat(1, self.num_heads, N, 1)   # (480,) 重复--> (1,4,480,480)

        if fg_score == None:
            fg_score = 1
        else:
            fg_score = torch.reshape(fg_score, [1, 1, 1, -1]).repeat(1, self.num_heads, N, 1)    # (480,) 重复--> (1,4,480,480)

        attn_cls_raw = v_cls_normed @ v_cls_normed.transpose(-2, -1)   # N(V)XN(V) (1,4,480,480) = (1,4,480,32) x (1,4,32,480)
        if use_mask:
            # only reference object with higher confidence..
            cls_score_mask = (cls_score > (cls_score.transpose(-2, -1) - 0.1)).type_as(cls_score)
            fg_score_mask = (fg_score > (fg_score.transpose(-2, -1) - 0.1)).type_as(fg_score)
        else:
            cls_score_mask = fg_score_mask = 1

        # cls_score_mask = (cls_score < (cls_score.transpose(-2, -1) + 0.1)).type_as(cls_score)
        # fg_score_mask = (fg_score < (fg_score.transpose(-2, -1) + 0.1)).type_as(fg_score)
        # visual_attention(cls_score[0, 0, :, :])
        # visual_attention(cls_score_mask[0,0,:,:])

        attn_cls = (q_cls @ k_cls.transpose(-2, -1)) * self.scale * cls_score * cls_score_mask   # (1,4,480,480) 标准的attention公式中的QKT/scale，乘K之前先乘上置信度 cls_score，即分类置信度(预测框所属的那个类别)作为K加权权重的一部分
        attn_cls = attn_cls.softmax(dim=-1)   # softmax() (1,4,480,480)
        attn_cls = self.attn_drop(attn_cls)   # dropout (1,4,480,480)

        attn_reg = (q_reg @ k_reg.transpose(-2, -1)) * self.scale * fg_score * fg_score_mask  # obj置信度作为reg attention的K加权权重的一部分
        attn_reg = attn_reg.softmax(dim=-1)   # softmax() (1,4,480,480)
        attn_reg = self.attn_drop(attn_reg)   # dropout (1,4,480,480)

        attn = (attn_reg + attn_cls) / 2 * coord_attn    # (1,4,480,480) 回归特征和分类特征的平均后得到的attention权重，称为多头yolo回归分类相似度

        # 将同一帧里的预测框相关度置0
        attn_same_frame_set_zero = attn.detach().cpu().numpy()
        attn_same_frame_set_zero_ori = attn_same_frame_set_zero.copy()
        for i in range(attn_same_frame_set_zero.shape[1]):
            for j in range(attn_same_frame_set_zero.shape[2]):
                attn_same_frame_set_zero[0][i][j, j // 10 * 10:j // 10 * 10 + (10 - 1)] = 0
                attn_same_frame_set_zero[0][i][j, j] = attn_same_frame_set_zero_ori[0][i][j, j]

        attn = torch.from_numpy(attn_same_frame_set_zero).to(attn.device).to(attn.dtype)  # (1,4,160,160)

        x = (attn @ v_cls).transpose(1, 2).reshape(B, N, C)  # (1, 4, 480, 32)-->(1, 480, 4, 32)-->(1,480,128) #

        x_ori = v_cls.permute(0, 2, 1, 3).reshape(B, N, C)  # 原始的V，即cls特征向量(cls_feat2)直接一次fc过来的 (1,480,128)
        x_cls = torch.cat([x, x_ori], dim=-1)               # 拼接的V (1,480,256): (原始cls_feat2+1fc的)+(考虑了置信度的MSA后的)
        #

        if ave:  # 注意这里虽然提炼了一个预测框最终相似度，考虑了余弦相似度mask和考虑了置信度的MSA相似度， 但该相似度在x_cls中没有被使用，后面又进行了使用
            ones_matrix = torch.ones(attn.shape[2:]).to('cuda')   # (480,480)
            zero_matrix = torch.zeros(attn.shape[2:]).to('cuda')  # (480,480)

            attn_cls_raw = torch.sum(attn_cls_raw, dim=1, keepdim=False)[0] / self.num_heads  # (1,4,480,480) sum--> (1,480,480) [0]--> (480,480) /4--> (480,480) 4个头平均
            sim_mask = torch.where(attn_cls_raw > sim_thresh, ones_matrix, zero_matrix)       # 根据 attn_cls_raw 即N(V)XN(V)进行二值化 (480, 480)，称为余弦相似度mask
            sim_attn = torch.sum(attn, dim=1, keepdim=False)[0] / self.num_heads              # 特殊MSA得到的最终atteneion权重   (480, 480) 4个头平均， 称为考虑了置信度的MSA相似度

            sim_round2 = torch.softmax(sim_attn, dim=-1)  # 概率化 (480, 480) 480=16(seq中图像张数)x30(每张选择的预测框数量)
            sim_round2 = sim_mask * sim_round2 / ((torch.sum(sim_mask * sim_round2, dim=-1, keepdim=True)) + torch.tensor([1e-8]).to('cuda')) # 最终seq中任意两个预测框的相似度 (480,480),称为预测框最终相似度
            return x_cls, None, sim_round2  # 拼接的V (1,480,256): 原始+MSA后的
        else:
            return x_cls, None, None

class Attention_msa_TwoStream(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., scale=25):
        # dim :input[batchsize,sequence length, input dimension]-->output[batchsize, sequence lenght, dim]
        # qkv_bias : Is it matter?
        # qk_scale, attn_drop,proj_drop will not be used
        # object = Attention(dim,num head)
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = scale  # qk_scale or head_dim ** -0.5

        # self.qkv_cls = nn.Linear(dim, dim * 2, bias=qkv_bias)  # 128-->384  3的目的是对应分别分给q k v， 也就是q k v 都是由原向量导出来的
        # self.v_fc = nn.Linear(dim, dim * 2, bias=qkv_bias)  # 128-->384  3的目的是对应分别分给q k v， 也就是q k v 都是由原向量导出来的
        self.attn_drop = nn.Dropout(attn_drop)
        self.qkv_cls = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.ReLU(True),
            nn.Linear(dim * 2, dim * 4),
            nn.ReLU(True),
            nn.Linear(dim * 4, dim * 2))

    def forward(self, x_cls, v_cls, cls_score=None, ave=True, sim_thresh=0.75, use_mask=False):
        """
        :param x_cls:   flatterned spacial feature  (1, 160, 64)
        :param v_cls:   flatterned spatiotemporal cls2-feat (1, 160, 64)
        :param cls_score:
        :param ave:
        :param sim_thresh:
        :param use_mask:
        :return:
        """
        B, N, C = x_cls.shape  # 1, 160, 64

        qkv_cls = self.qkv_cls(x_cls).reshape(B, N, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)  # 2, B, num_head, N, c   (2,1,4,160,16)
        v_cls   = v_cls.reshape(B, N, 1, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)[0]   # (1,1,4,160,16)

        q_cls, k_cls = qkv_cls[0], qkv_cls[1]     # make torchscript happy (cannot use tensor as tuple) (3,1,4,160,32),(3,1,4,160,32),(3,1,4,160,32)
        # 注意这里会出现除0,所以要加上1e-8
        q_cls = q_cls / (torch.norm(q_cls, dim=-1, keepdim=True) + torch.tensor([1e-8]).to('cuda'))       # (1,4,160,32)  4就是4个头的意思，每个头处理32个维度，一共处理128维度，每个头都有自己的QKV
        k_cls = k_cls / (torch.norm(k_cls, dim=-1, keepdim=True) + torch.tensor([1e-8]).to('cuda'))       # (1,4,160,32)

        v_cls_normed = v_cls / (torch.norm(v_cls, dim=-1, keepdim=True)+ torch.tensor([1e-8]).to('cuda')) # (1,4,160,32)

        if cls_score == None:
            cls_score = 1
        else:
            cls_score = torch.reshape(cls_score, [1, 1, 1, -1]).repeat(1, self.num_heads, N, 1)   # (160,) 行重复--> (1,4,160,160)

        attn_cls_raw = v_cls_normed @ v_cls_normed.transpose(-2, -1)   # N(V)XN(V) (1,4,160,160) = (1,4,160,32) x (1,4,32,160)
        if use_mask:
            # only reference object with higher confidence..
            cls_score_mask = (cls_score > (cls_score.transpose(-2, -1) - 0.1)).type_as(cls_score)
        else:
            cls_score_mask = fg_score_mask = 1

        # cls_score_mask = (cls_score < (cls_score.transpose(-2, -1) + 0.1)).type_as(cls_score)
        # fg_score_mask = (fg_score < (fg_score.transpose(-2, -1) + 0.1)).type_as(fg_score)
        # visual_attention(cls_score[0, 0, :, :])
        # visual_attention(cls_score_mask[0,0,:,:])

        attn_cls = (q_cls @ k_cls.transpose(-2, -1)) * self.scale * cls_score * cls_score_mask   # (1,4,160,160) 标准的attention公式中的QKT/scale，乘K之前先乘上置信度 cls_score，即分类置信度(预测框所属的那个类别)作为K加权权重的一部分
        attn_cls = attn_cls.softmax(dim=-1)   # softmax() (1,4,160,160)
        attn_cls = self.attn_drop(attn_cls)   # dropout (1,4,160,160)

        attn = attn_cls      # (1,4,160,160) 回归特征和分类特征的平均后得到的attention权重，称为多头yolo回归分类相似度

        # 将同一帧里的预测框相关度置0
        attn_same_frame_set_zero = attn_cls_raw.detach().cpu().numpy()     # cosine 相似度
        # attn_same_frame_set_zero = attn.detach().cpu().numpy()           # MSA 相似度
        attn_same_frame_set_zero_ori = attn_same_frame_set_zero.copy()
        for i in range(attn_same_frame_set_zero.shape[1]):
            for j in range(attn_same_frame_set_zero.shape[2]):
                attn_same_frame_set_zero[0][i][j, j // 10 * 10:j // 10 * 10 + (10 - 1)] = 0
                attn_same_frame_set_zero[0][i][j, j] = attn_same_frame_set_zero_ori[0][i][j, j]

        attn = torch.from_numpy(attn_same_frame_set_zero).to(attn.device).to(attn.dtype)   # (1,4,160,160)

        x = (attn @ v_cls).transpose(1, 2).reshape(B, N, C)  # (1, 4, 160, 32)-->(1, 160, 4, 32)-->(1,160,128) #

        x_ori = v_cls.permute(0, 2, 1, 3).reshape(B, N, C)  # 原始的V，即cls特征向量(cls_feat2)直接一次fc过来的 (1,160,128)
        x_cls = torch.cat([x, x_ori], dim=-1)               # 拼接的V (1,160,256): (原始cls_feat2+1fc的)+(考虑了置信度的MSA后的)
        #

        if ave:  # 注意这里虽然提炼了一个预测框最终相似度，考虑了余弦相似度mask和考虑了置信度的MSA相似度， 但该相似度在x_cls中没有被使用，后面又进行了使用
            ones_matrix = torch.ones(attn.shape[2:]).to('cuda')   # (160,160)
            zero_matrix = torch.zeros(attn.shape[2:]).to('cuda')  # (160,160)

            attn_cls_raw = torch.sum(attn_cls_raw, dim=1, keepdim=False)[0] / self.num_heads  # (1,4,160,160) sum--> (1,160,160) [0]--> (160,160) /4--> (160,160) 4个头平均
            sim_mask = torch.where(attn_cls_raw > sim_thresh, ones_matrix, zero_matrix)       # 根据 attn_cls_raw 即N(V)XN(V)进行二值化 (160, 160)，称为余弦相似度mask
            sim_attn = torch.sum(attn, dim=1, keepdim=False)[0] / self.num_heads              # 特殊MSA得到的最终atteneion权重   (160, 160) 4个头平均， 称为考虑了置信度的MSA相似度

            sim_round2 = torch.softmax(sim_attn, dim=-1)  # 概率化 (160, 160) 160=16(seq中图像张数)x10(每张选择的预测框数量)
            sim_round2 = sim_mask * sim_round2 / ((torch.sum(sim_mask * sim_round2, dim=-1, keepdim=True)) + torch.tensor([1e-8]).to('cuda')) # 最终seq中任意两个预测框的相似度 (160,160),称为预测框最终相似度
            return x_cls, None, sim_round2  # 拼接的V (1,160,256): 原始+MSA后的
        else:
            return x_cls, None, None

class Attention_msa_Mutual(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., scale=25, prior_head_feats=None):
        # dim :input[batchsize,sequence length, input dimension]-->output[batchsize, sequence lenght, dim]
        # qkv_bias : Is it matter?
        # qk_scale, attn_drop,proj_drop will not be used
        # object = Attention(dim,num head)
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = scale  # qk_scale or head_dim ** -0.5

        self.qkv_cls = nn.Linear(dim, dim * 3, bias=qkv_bias)  # 64-->64*3  3的目的是对应分别分给q k v， 也就是q k v 都是由原向量导出来的
        self.qkv_reg = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)

        self.prior_head_feats = torch.from_numpy(np.load(prior_head_feats))   # (n,128)

    def forward(self, x_cls, x_reg):
        B, N, C = x_cls.shape  # 1, 480, 64

        qkv_cls = self.qkv_cls(x_cls).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)    # 3, B, num_head, N, c   (3,1,4,480,16)
        qkv_reg = self.qkv_reg(x_reg).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)    # (3,1,4,480,16)
        q_cls, k_cls, v_cls = qkv_cls[0], qkv_cls[1], qkv_cls[2]  # make torchscript happy (cannot use tensor as tuple) (3,1,4,480,16),(3,1,4,480,16),(3,1,4,480,16)
        q_reg, k_reg, v_reg = qkv_reg[0], qkv_reg[1], qkv_reg[2]  # (1,4,480,16),(1,4,480,16),(1,4,480,16)
        # 注意这里会出现除0
        q_cls = q_cls / (torch.norm(q_cls, dim=-1, keepdim=True) + torch.tensor([1e-8]).to('cuda')) # (1,4,480,16)  4就是4个头的意思，每个头处理32个维度，一共处理128维度，每个头都有自己的QKV
        q_reg = q_reg / (torch.norm(q_reg, dim=-1, keepdim=True) + torch.tensor([1e-8]).to('cuda'))  # (1,4,480,16)

        self.prior_head_feats = self.prior_head_feats.to(self.qkv_cls.weight.device).to(self.qkv_cls.weight.dtype)

        x_cls_prior = self.prior_head_feats[:, 0:64].unsqueeze(0)  #  1, xxx, 64   xxx: 50114
        x_reg_prior = self.prior_head_feats[:, 64:].unsqueeze(0)   #  1, xxx, 64
        _, N_prior, _ = x_cls_prior.shape  # 1, 480, 64
        qkv_cls_prior = self.qkv_cls(x_cls_prior).reshape(B, N_prior, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)    #  3, B, num_head, N, c   (3,1,4,xxx,16=64/4)
        qkv_reg_prior = self.qkv_reg(x_reg_prior).reshape(B, N_prior, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)    # (3,1,4,xxx,16=64/4)
        q_cls_prior, k_cls_prior, v_cls_prior = qkv_cls_prior[0], qkv_cls_prior[1], qkv_cls_prior[2]  # make torchscript happy (cannot use tensor as tuple) (3,1,4,xxx,16),(3,1,4,xxx,16),(3,1,4,xxx,16)
        q_reg_prior, k_reg_prior, v_reg_prior = qkv_reg_prior[0], qkv_reg_prior[1], qkv_reg_prior[2]  # (1,4,xxx,16),(1,4,xxx,16),(1,4,xxx,16)
        k_cls_prior = k_cls_prior / (torch.norm(k_cls_prior, dim=-1, keepdim=True) + torch.tensor([1e-8]).to('cuda')) # (1,4,xxx,16)
        k_reg_prior = k_reg_prior / (torch.norm(k_reg_prior, dim=-1, keepdim=True) + torch.tensor([1e-8]).to('cuda')) # (1,4,xxx,16)

        # 互监督
        attn_cls = (q_cls @ k_cls_prior.transpose(-2, -1)) * self.scale   # (1,4,480,xxx) 标准的attention公式中的QKT/scale，乘K之前先乘上置信度 cls_score，即分类置信度(预测框所属的那个类别)作为K加权权重的一部分
        attn_cls = attn_cls.softmax(dim=-1)   # softmax() (1,4,480,xxx)
        attn_cls = self.attn_drop(attn_cls)   # dropout (1,4,480,xxx)

        attn_reg = (q_reg @ k_reg_prior.transpose(-2, -1)) * self.scale  # obj置信度作为reg attention的K加权权重的一部分
        attn_reg = attn_reg.softmax(dim=-1)   # softmax() (1,4,480,xxx)
        attn_reg = self.attn_drop(attn_reg)   # dropout (1,4,480,xxx)

        attn = (attn_reg + attn_cls) / 2      # (1,4,480,xxx) 回归特征和分类特征的平均后得到的attention权重，称为多头yolo回归分类相似度
        x = (attn @ v_cls_prior).transpose(1, 2).reshape(B, N, C)  # (1, 4, 480, 16)-->(1, 480, 4, 16)-->(1,480,64) #

        x_ori = v_cls.permute(0, 2, 1, 3).reshape(B, N, C)  # 原始的V，即cls特征向量(cls_feat2)直接一次fc过来的 (1,480,64)
        x_cls = torch.cat([x, x_ori], dim=-1)               # 拼接的V (1,480,64+64): (原始cls_feat2+1fc的)+(考虑了置信度的MSA后的)

        return x_cls

class Native_Attention_msa(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., scale=25, proj_drop=0.):
        # dim :input[batchsize,sequence length, input dimension]-->output[batchsize, sequence lenght, dim]
        # qkv_bias : Is it matter?
        # qk_scale, attn_drop,proj_drop will not be used
        # object = Attention(dim,num head)
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = head_dim ** -0.5  # qk_scale or head_dim ** -0.5

        self.qkv_cls = nn.Linear(dim, dim * 3, bias=qkv_bias)  # 128-->384  3的目的是对应分别分给q k v， 也就是q k v 都是由原向量导出来的
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x_cls, cls_score=None, use_mask=False):
        """
        :param x_cls:
        :param cls_score:
        :param use_mask:
        :return:
        """
        B, N, C = x_cls.shape  # 1, 480, 128

        qkv_cls = self.qkv_cls(x_cls).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1,4)  # 3, B, num_head, N, c   (3,1,4,480,32)
        q_cls, k_cls, v_cls = qkv_cls[0], qkv_cls[1], qkv_cls[2]  # make torchscript happy (cannot use tensor as tuple) (3,1,4,480,32),(3,1,4,480,32),(3,1,4,480,32)
        # 注意这里会出现除0
        q_cls = q_cls / (torch.norm(q_cls, dim=-1, keepdim=True) + torch.tensor([1e-8]).to('cuda')) # (1,4,480,32)  4就是4个头的意思，每个头处理32个维度，一共处理128维度，每个头都有自己的QKV
        k_cls = k_cls / (torch.norm(k_cls, dim=-1, keepdim=True) + torch.tensor([1e-8]).to('cuda')) # (1,4,480,32)

        if cls_score == None:
            cls_score = 1
        else:
            cls_score = torch.reshape(cls_score, [1, 1, 1, -1]).repeat(1, self.num_heads, N, 1)   # (480,) 重复--> (1,4,480,480)

        if use_mask:
            # only reference object with higher confidence..
            cls_score_mask = (cls_score > (cls_score.transpose(-2, -1) - 0.1)).type_as(cls_score)
        else:
            cls_score_mask  = 1

        attn_cls = (q_cls @ k_cls.transpose(-2, -1)) * self.scale * cls_score * cls_score_mask   # (1,4,480,480) 标准的attention公式中的QKT/scale，乘K之前先乘上置信度 cls_score，即分类置信度(预测框所属的那个类别)作为K加权权重的一部分
        attn_cls = attn_cls.softmax(dim=-1)   # softmax() (1,4,480,480)
        attn_cls = self.attn_drop(attn_cls)   # dropout (1,4,480,480)

        attn = attn_cls      # (1,4,480,480) 回归特征和分类特征的平均后得到的attention权重，称为多头yolo回归分类相似度
        x = (attn @ v_cls).transpose(1, 2).reshape(B, N, C)  # (1, 4, 480, 32)-->(1, 480, 4, 32)-->(1,480,128) #

        # (B, N, C) -> (B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x

class Native_Attention_msa_onlyAttn(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., scale=25, proj_drop=0.):
        # dim :input[batchsize,sequence length, input dimension]-->output[batchsize, sequence lenght, dim]
        # qkv_bias : Is it matter?
        # qk_scale, attn_drop,proj_drop will not be used
        # object = Attention(dim,num head)
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = scale  # qk_scale or head_dim ** -0.5

        self.qkv_cls = nn.Linear(dim, dim * 2, bias=qkv_bias)  # 128-->384  3的目的是对应分别分给q k v， 也就是q k v 都是由原向量导出来的

    def forward(self, x_cls):
        """
        :param x_cls:
        :param cls_score:
        :param use_mask:
        :return:
        """
        B, N, C = x_cls.shape  # 1, 160, 128

        qkv_cls = self.qkv_cls(x_cls).reshape(B, N, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1,4)  # 3, B, num_head, N, c   (2,1,4,480,32)
        q_cls, k_cls = qkv_cls[0], qkv_cls[1] # make torchscript happy (cannot use tensor as tuple) (3,1,4,480,32),(3,1,4,480,32),(3,1,4,480,32)
        # 注意这里会出现除0
        q_cls = q_cls / (torch.norm(q_cls, dim=-1, keepdim=True) + torch.tensor([1e-8]).to('cuda')) # (1,4,480,32)  4就是4个头的意思，每个头处理32个维度，一共处理128维度，每个头都有自己的QKV
        k_cls = k_cls / (torch.norm(k_cls, dim=-1, keepdim=True) + torch.tensor([1e-8]).to('cuda')) # (1,4,480,32)

        attn_cls = (q_cls @ k_cls.transpose(-2, -1)) * self.scale    # (1,4,480,480) 标准的attention公式中的QKT/scale，乘K之前先乘上置信度 cls_score，即分类置信度(预测框所属的那个类别)作为K加权权重的一部分
        attn_cls = attn_cls.softmax(dim=-1)   # softmax() (1,4,480,480)

        return attn_cls

class Attention_msa_visual(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False,attn_drop=0., scale=25):
        # dim :input[batchsize,sequence length, input dimension]-->output[batchsize, sequence lenght, dim]
        # qkv_bias : Is it matter?
        # qk_scale, attn_drop,proj_drop will not be used
        # object = Attention(dim,num head)
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = 30#scale  # qk_scale or head_dim ** -0.5

        self.qkv_cls = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.qkv_reg = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)

    def forward(self, x_cls, x_reg, cls_score=None, fg_score=None,img = None, pred = None):
        B, N, C = x_cls.shape

        qkv_cls = self.qkv_cls(x_cls).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1,
                                                                                                    4)  # 3, B, num_head, N, c
        qkv_reg = self.qkv_reg(x_reg).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q_cls, k_cls, v_cls = qkv_cls[0], qkv_cls[1], qkv_cls[2]  # make torchscript happy (cannot use tensor as tuple)
        q_reg, k_reg, v_reg = qkv_reg[0], qkv_reg[1], qkv_reg[2]

        q_cls = q_cls / torch.norm(q_cls, dim=-1, keepdim=True)
        k_cls = k_cls / torch.norm(k_cls, dim=-1, keepdim=True)
        q_reg = q_reg / torch.norm(q_reg, dim=-1, keepdim=True)
        k_reg = k_reg / torch.norm(k_reg, dim=-1, keepdim=True)
        v_cls_normed = v_cls / torch.norm(v_cls,dim=-1,keepdim=True)

        if cls_score == None:
            cls_score = 1
        else:
            cls_score = torch.reshape(cls_score,[1,1,1,-1]).repeat(1,self.num_heads,N, 1)

        if fg_score == None:
            fg_score = 1
        else:
            fg_score = torch.reshape(fg_score, [1, 1, 1, -1]).repeat(1,self.num_heads,N, 1)

        attn_cls_raw = v_cls_normed @ v_cls_normed.transpose(-2, -1)

        attn_cls = (q_cls @ k_cls.transpose(-2, -1)) * self.scale * cls_score #* cls_score
        attn_cls = attn_cls.softmax(dim=-1)
        attn_cls = self.attn_drop(attn_cls)

        attn_reg = (q_reg @ k_reg.transpose(-2, -1)) * self.scale * fg_score
        attn_reg = attn_reg.softmax(dim=-1)
        attn_reg = self.attn_drop(attn_reg)

        attn = (attn_cls_raw*25).softmax(dim=-1)#attn_cls#(attn_reg + attn_cls) / 2 #attn_reg#(attn_reg + attn_cls) / 2
        x = (attn @ v_cls).transpose(1, 2).reshape(B, N, C)

        x_ori = v_cls.permute(0,2,1,3).reshape(B, N, C)
        x_cls = torch.cat([x, x_ori], dim=-1)

        ones_matrix = torch.ones(attn.shape[2:]).to('cuda')
        zero_matrix = torch.zeros(attn.shape[2:]).to('cuda')

        attn_cls_raw = torch.sum(attn_cls_raw,dim=1,keepdim=False)[0] / self.num_heads
        sim_mask = torch.where(attn_cls_raw > 0.75, ones_matrix, zero_matrix)
        sim_attn = torch.sum(attn, dim=1, keepdim=False)[0] / self.num_heads

        sim_round2 = torch.softmax(sim_attn, dim=-1)
        sim_round2 = sim_mask*sim_round2/(torch.sum(sim_mask*sim_round2,dim=-1,keepdim=True))
        from yoloV.models.post_process import visual_sim
        attn_total = torch.sum(attn,dim=1,keepdim=False)[0] / self.num_heads
        visual_sim(attn_total,img,30,pred,attn_cls_raw)
        return x_cls,None,sim_round2

class Attention_msa_online(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False,attn_drop=0., scale=25):
        # dim :input[batchsize,sequence length, input dimension]-->output[batchsize, sequence lenght, dim]
        # qkv_bias : Is it matter?
        # qk_scale, attn_drop,proj_drop will not be used
        # object = Attention(dim,num head)
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = scale  # qk_scale or head_dim ** -0.5
        self.qkv_cls = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.qkv_reg = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)

    def forward(self, x_cls, x_reg, cls_score=None, fg_score=None, return_attention=False,ave = True):
        B, N, C = x_cls.shape

        qkv_cls = self.qkv_cls(x_cls).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1,
                                                                                                    4)  # 3, B, num_head, N, c
        qkv_reg = self.qkv_reg(x_reg).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q_cls, k_cls, v_cls = qkv_cls[0], qkv_cls[1], qkv_cls[2]  # make torchscript happy (cannot use tensor as tuple)
        q_reg, k_reg, v_reg = qkv_reg[0], qkv_reg[1], qkv_reg[2]

        q_cls = q_cls / torch.norm(q_cls, dim=-1, keepdim=True)
        k_cls = k_cls / torch.norm(k_cls, dim=-1, keepdim=True)
        q_reg = q_reg / torch.norm(q_reg, dim=-1, keepdim=True)
        k_reg = k_reg / torch.norm(k_reg, dim=-1, keepdim=True)
        v_cls_normed = v_cls / torch.norm(v_cls,dim=-1,keepdim=True)

        if cls_score == None:
            cls_score = 1
        else:
            cls_score = torch.reshape(cls_score,[1,1,1,-1]).repeat(1,self.num_heads,N, 1)

        if fg_score == None:
            fg_score = 1
        else:
            fg_score = torch.reshape(fg_score, [1, 1, 1, -1]).repeat(1,self.num_heads,N, 1)

        attn_cls_raw = v_cls_normed @ v_cls_normed.transpose(-2, -1)

        attn_cls = (q_cls @ k_cls.transpose(-2, -1)) * self.scale * cls_score
        attn_cls = attn_cls.softmax(dim=-1)
        attn_cls = self.attn_drop(attn_cls)

        attn_reg = (q_reg @ k_reg.transpose(-2, -1)) * self.scale * fg_score
        attn_reg = attn_reg.softmax(dim=-1)
        attn_reg = self.attn_drop(attn_reg)

        attn = (attn_reg + attn_cls) / 2
        x = (attn @ v_cls).transpose(1, 2).reshape(B, N, C)

        x_ori = v_cls.permute(0,2,1,3).reshape(B, N, C)
        x_cls = torch.cat([x, x_ori], dim=-1)
        if ave:
            ones_matrix = torch.ones(attn.shape[2:]).to('cuda')
            zero_matrix = torch.zeros(attn.shape[2:]).to('cuda')

            attn_cls_raw = torch.sum(attn_cls_raw,dim=1,keepdim=False)[0] / self.num_heads
            sim_mask = torch.where(attn_cls_raw > 0.75, ones_matrix, zero_matrix)
            sim_attn = torch.sum(attn, dim=1, keepdim=False)[0] / self.num_heads

            sim_round2 = torch.softmax(sim_attn, dim=-1)
            sim_round2 = sim_mask*sim_round2/(torch.sum(sim_mask*sim_round2,dim=-1,keepdim=True))
            return x_cls,None,sim_round2
        else:
            return x_cls

class MSA_yolov(nn.Module):

    def __init__(self, dim, out_dim, num_heads=4, qkv_bias=False, attn_drop=0., scale=25):
        super().__init__()
        # self.layerNorm = nn.LayerNorm(dim)
        # self.multiMSA = Native_MSA_block(dim, num_heads, qkv_bias, attn_drop, scale=scale)

        self.msa = Attention_msa(dim, num_heads, qkv_bias, attn_drop, scale=scale)
        self.linear1 = nn.Linear(2 * dim, 2 * dim)
        self.linear2 = nn.Linear(4 * dim, out_dim)
        self.linear3 = nn.Linear(2 * dim, out_dim)

    def find_similar_round2(self, features, sort_results):  # features: (1,480,256), sort_results:(480,480)所有预测框互相之间的相似度
        key_feature = features[0]       # (480,256)
        support_feature = features[0]
        if not self.training:
            sort_results = sort_results.to(features.dtype)
        soft_sim_feature = (sort_results @ support_feature)  # (480,256) 根据相似度进行加权求和，即论文中的”average pooling over reference features“
        cls_feature = torch.cat([soft_sim_feature, key_feature], dim=-1) # (480,512) 最终一个预测框的特征：128原始分类回归特征+128考虑了置信度的MSA+256所有向量(128+128)根据预测框最终相似度加权平均的特征
        return cls_feature

    def forward(self, x_cls, x_reg, cls_score=None, fg_score=None, sim_thresh=0.75, ave=True, use_mask=False):
        # x_cls = self.layerNorm(x_cls)
        # x_reg = self.layerNorm(x_reg)
        trans_cls, trans_reg, sim_round2 = self.msa(x_cls, x_reg, cls_score, fg_score, sim_thresh=sim_thresh, ave=ave, use_mask=use_mask)   # trans_cls: (1,480,256) x_cls:(1,480,128), x_reg:(1,480,128), cls_score:(480,), fg_score:(480,)
        msa = self.linear1(trans_cls)                      # (1,480,256)-->(1,480,256)

        if ave:
            msa = self.find_similar_round2(msa, sim_round2)    # (1,480,256) --> (480,512)
            out = self.linear2(msa)  # (480,512)--->(480,512)  最终的分类特征： cls_feat2+1fc的128维 + (cls_feat2的考虑了cls置信度的MSA特征+reg_feat的考虑了obj置信度的MSA特征)/2的128维 + 对前两者拼接的(256)按预测框最终置信度进行加权平均的256维
        else:
            out = self.linear3(msa)  # (480,512)--->(480,512)  最终的分类特征： cls_feat2+1fc的128维 + (cls_feat2的考虑了cls置信度的MSA特征+reg_feat的考虑了obj置信度的MSA特征)/2的128维 + 对前两者拼接的(256)按预测框最终置信度进行加权平均的256维

        # msa = self.find_similar_round2(msa, sim_round2)    # (1,480,256) --> (480,512)
        # out = self.linear2(msa)  # (480,512)--->(480,512)  最终的分类特征： cls_feat2+1fc的128维 + (cls_feat2的考虑了cls置信度的MSA特征+reg_feat的考虑了obj置信度的MSA特征)/2的128维 + 对前两者拼接的(256)按预测框最终置信度进行加权平均的256维

        return out

class MSA_yolov_coord(nn.Module):

    def __init__(self, dim, out_dim, num_heads=4, qkv_bias=False, attn_drop=0., scale=25):
        super().__init__()
        # self.layerNorm = nn.LayerNorm(dim)
        # self.multiMSA = Native_MSA_block(dim, num_heads, qkv_bias, attn_drop, scale=scale)

        self.msa = Attention_msa_coord(dim, num_heads, qkv_bias, attn_drop, scale=scale)
        self.linear1 = nn.Linear(2 * dim, 2 * dim)
        self.linear2 = nn.Linear(4 * dim, out_dim)
        self.linear3 = nn.Linear(2 * dim, out_dim)

    def find_similar_round2(self, features, sort_results):  # features: (1,480,256), sort_results:(480,480)所有预测框互相之间的相似度
        key_feature = features[0]       # (480,256)
        support_feature = features[0]
        if not self.training:
            sort_results = sort_results.to(features.dtype)
        soft_sim_feature = (sort_results @ support_feature)  # (480,256) 根据相似度进行加权求和，即论文中的”average pooling over reference features“
        cls_feature = torch.cat([soft_sim_feature, key_feature], dim=-1) # (480,512) 最终一个预测框的特征：128原始分类回归特征+128考虑了置信度的MSA+256所有向量(128+128)根据预测框最终相似度加权平均的特征
        return cls_feature

    def forward(self, x_cls, x_reg, cls_score=None, fg_score=None, coord_attn=None, sim_thresh=0.75, ave=True, use_mask=False):
        # x_cls = self.layerNorm(x_cls)
        # x_reg = self.layerNorm(x_reg)
        trans_cls, trans_reg, sim_round2 = self.msa(x_cls, x_reg, cls_score, fg_score, coord_attn = coord_attn, sim_thresh=sim_thresh, ave=ave, use_mask=use_mask)   # trans_cls: (1,480,256) x_cls:(1,480,128), x_reg:(1,480,128), cls_score:(480,), fg_score:(480,)
        msa = self.linear1(trans_cls)                      # (1,480,256)-->(1,480,256)

        if ave:
            msa = self.find_similar_round2(msa, sim_round2)    # (1,480,256) --> (480,512)
            out = self.linear2(msa)  # (480,512)--->(480,512)  最终的分类特征： cls_feat2+1fc的128维 + (cls_feat2的考虑了cls置信度的MSA特征+reg_feat的考虑了obj置信度的MSA特征)/2的128维 + 对前两者拼接的(256)按预测框最终置信度进行加权平均的256维
        else:
            out = self.linear3(msa)  # (480,512)--->(480,512)  最终的分类特征： cls_feat2+1fc的128维 + (cls_feat2的考虑了cls置信度的MSA特征+reg_feat的考虑了obj置信度的MSA特征)/2的128维 + 对前两者拼接的(256)按预测框最终置信度进行加权平均的256维

        # msa = self.find_similar_round2(msa, sim_round2)    # (1,480,256) --> (480,512)
        # out = self.linear2(msa)  # (480,512)--->(480,512)  最终的分类特征： cls_feat2+1fc的128维 + (cls_feat2的考虑了cls置信度的MSA特征+reg_feat的考虑了obj置信度的MSA特征)/2的128维 + 对前两者拼接的(256)按预测框最终置信度进行加权平均的256维

        return out

class LSA(nn.Module):
    def __init__(self, dim, heads = 4, dim_head = 32, dropout = 0.5):
        super().__init__()
        inner_dim = dim_head *  heads
        self.heads = heads
        self.temperature = nn.Parameter(torch.log(torch.tensor(dim_head ** -0.5)))

        self.attend = nn.Softmax(dim = -1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):   # mutual q-kv
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)   # (1,4,16X25+1,32)

        dots = torch.matmul(q[0,:,0,:].unsqueeze(0).unsqueeze(2), k.transpose(-1, -2)) * self.temperature.exp()   # [4,32] --> [1,4,1,32] * [1,4,32,16X25+1] = [1,4,1,401]

        mask = torch.zeros((1,dots.shape[-1]), device = dots.device, dtype = torch.bool)  # (1,401)
        mask[0,0] = torch.tensor([1],device = dots.device, dtype = torch.bool)
        mask_value = -torch.finfo(dots.dtype).max
        dots = dots.masked_fill(mask, mask_value)

        attn = self.attend(dots)     # softmax
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)    # [1,4,1,401] * (1,4,401,32) = (1,4,1,32)
        out = rearrange(out, 'b h n d -> b n (h d)')  # (1,4,1,32) --> (1,1,128)
        return self.to_out(out)        # (1,1,128) --> (1,1,128)

    # def forward(self, x):
    #     qkv = self.to_qkv(x).chunk(3, dim = -1)
    #     q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)
    #
    #     dots = torch.matmul(q, k.transpose(-1, -2)) * self.temperature.exp()
    #
    #     mask = torch.eye(dots.shape[-1], device = dots.device, dtype = torch.bool)
    #     mask_value = -torch.finfo(dots.dtype).max
    #     dots = dots.masked_fill(mask, mask_value)
    #
    #     attn = self.attend(dots)
    #     attn = self.dropout(attn)
    #
    #     out = torch.matmul(attn, v)
    #     out = rearrange(out, 'b h n d -> b n (h d)')
    #     return self.to_out(out)

def visualize_attn_1head(attn,k,h,t):
    a = attn[k,h,0,1:]               # (324,) 第一个数：第几个candidate, 第二个：第几个head, 第三个：当前那个candidate, 第四个：1+w*w*N
                                     # 排列的顺序为 第一帧一行接一行，第二帧，一行接一行，...
    a = a.reshape((4,9,9))
    a = a.detach().cpu().numpy()
    return a[t]

def visualize_attn_4head(attn,k):
    """
    :param attn:  [p,4,1,N]
    :param k:
    :return:
    """
    m = []
    for i in range(4):                   # 遍历每一个头
        a = attn[k,i,0,1:]               # (324,) 第一个数：第几个candidate, 第二个：第几个head, 第三个：当前那个candidate, 第四个：1+w*w*N
                                         # 排列的顺序为 第一帧一行接一行，第二帧，一行接一行，...
        a = a.reshape((4,9,9))
        a = a.detach().cpu().numpy()
        # for j in range(4):               # 遍历每一帧
        a = np.hstack((a[0],a[1],a[2],a[3]))
        m.append(a)

    max = m.max()
    min = m.min()
    m = ((m-min) / (max-min) * 255).astype(np.uint8)
    m_r = cv2.cv2.resize(m,(640*4,640*4))

    return m,m_r


def visualize_attn_4head2(attn, k):
    """
    :param attn:  [p,4,1,N]
    :param k:
    :return:
    """
    m = []
    aa = np.zeros((4, 9, 9))
    for i in range(4):  # 遍历每一个头
        a = attn[k, i, 0, 1:]  # (324,) 第一个数：第几个candidate, 第二个：第几个head, 第三个：当前那个candidate, 第四个：1+w*w*N
        # 排列的顺序为 第一帧一行接一行，第二帧，一行接一行，...
        a = a.detach().cpu().numpy()
        # a = a.reshape((4, 9, 9))
        for p in range(4):
            for q in range(9):
                for r in range(9):
                    aa[p,q,r] = a[p*81+q*9+r]
        a = aa

        a_r = np.zeros((4,160,160),dtype=np.uint8)
        for j in range(4):               # 遍历每一帧
            max = a[j].max()
            min = a[j].min()
            a[j] = ((a[j] - min) / (max - min) * 255).astype(np.uint8)
            a_r[j] = cv2.cv2.resize(a[j], (160, 160))
        a_r = np.hstack((a_r[0], a_r[1], a_r[2], a_r[3]))
        m.append(a_r)
    m = np.vstack((m[0], m[1], m[2], m[3]))
    return m

class LSA_parallel(nn.Module):
    def __init__(self, dim, heads = 4, dim_head = 32, dropout = 0.5):
        super().__init__()
        inner_dim = dim_head *  heads
        self.heads = heads
        self.temperature = nn.Parameter(torch.log(torch.tensor(dim_head ** -0.5)))

        self.attend = nn.Softmax(dim = -1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):   # cross attention q-kv  x:(p,N,64)
        qkv = self.to_qkv(x).chunk(3, dim = -1)  # tuple3: (p,N,256)   chunk: 分块
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)   # (p,4,N,64)

        dots = torch.matmul(q[:,:,0,:].unsqueeze(2), k.transpose(-1, -2)) * self.temperature.exp()   # 第一个向量和其他所有进行attn, [p,4,32] --> [p,4,1,64] * [p,4,64,N] = [p,4,1,N]

        mask = torch.zeros((1,dots.shape[-1]), device = dots.device, dtype = torch.bool)             # (1,N)
        mask[:,0] = torch.tensor([1],device = dots.device, dtype = torch.bool)                       # [0,0]位置为True,其他为False
        mask_value = -torch.finfo(dots.dtype).max                                                    # [0,0] 位置为负无穷，attn时不考虑自己
        dots = dots.masked_fill(mask, mask_value)     # [p,4,1,N]

        attn = self.attend(dots)                      # softmax  [p,4,1,N]
        # visual_r = visualize_attn_4head2(attn,k=0)
        # visual_r1 = visualize_attn_4head2(attn,k=1)
        # visual_r2 = visualize_attn_4head2(attn,k=2)
        # visual_r3 = visualize_attn_4head2(attn,k=3)
        # visual_r4 = visualize_attn_4head2(attn,k=4)
        # visual_r5 = visualize_attn_4head2(attn,k=5)
        # visual_r6 = visualize_attn_4head2(attn,k=6)
        # visual_r7 = visualize_attn_4head2(attn,k=7)
        # visual_r8 = visualize_attn_4head2(attn,k=8)
        # visual_r9 = visualize_attn_4head2(attn,k=9)

        attn = self.dropout(attn)                     # [p,4,1,N]

        out = torch.matmul(attn, v)                   # [p,4,1,N] * (p,4,N,64) = (p,4,1,64)
        out = rearrange(out, 'b h n d -> b n (h d)')  # (p,4,1,64) --> (p,1,256)
        return self.to_out(out)                       # (p,1,256) --> (p,1,64)

    # def forward(self, x):
    #     qkv = self.to_qkv(x).chunk(3, dim = -1)
    #     q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)
    #
    #     dots = torch.matmul(q, k.transpose(-1, -2)) * self.temperature.exp()
    #
    #     mask = torch.eye(dots.shape[-1], device = dots.device, dtype = torch.bool)
    #     mask_value = -torch.finfo(dots.dtype).max
    #     dots = dots.masked_fill(mask, mask_value)
    #
    #     attn = self.attend(dots)
    #     attn = self.dropout(attn)
    #
    #     out = torch.matmul(attn, v)
    #     out = rearrange(out, 'b h n d -> b n (h d)')
    #     return self.to_out(out)

class LSA_Native(nn.Module):
    def __init__(self, dim, heads = 4, dim_head = 32, dropout = 0.5):
        super().__init__()
        inner_dim = dim_head *  heads
        self.heads = heads
        self.temperature = nn.Parameter(torch.log(torch.tensor(dim_head ** -0.5)))

        self.attend = nn.Softmax(dim = -1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)   # (1,4,16X25+1,32)=(1,4,401,32)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.temperature.exp()                 # [4,32] --> [1,4,401,32] * [1,4,32,401] = [1,4,401,401]

        mask = torch.eye(dots.shape[-1], device = dots.device, dtype = torch.bool)           # (401,401)
        mask_value = -torch.finfo(dots.dtype).max
        dots = dots.masked_fill(mask, mask_value)      # (1,4,325,325)

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)                    # [1,4,401,401] * (1,4,401,32) = (1,4,401,32)
        out = rearrange(out, 'b h n d -> b n (h d)')   # (1,4,401,32) --> (1,401,128)
        return self.to_out(out)

class MSA_yolov_Mutual_ViT(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, LSA(dim, heads=heads, dim_head=dim_head, dropout=dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout))
            ]))

    def forward(self, x):   # mutual
        for attn, ff in self.layers:
            x = attn(x) + x[0,0].unsqueeze(0).unsqueeze(0)     # attn(x): (1,1,64)  + (1,1,64)
            x = ff(x) + x
        return x

class MSA_yolov_Mutual_ViT_parallel(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, LSA_parallel(dim, heads=heads, dim_head=dim_head, dropout=dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout))
            ]))

    def forward(self, x):   # mutual    x: (p, 401, 64)
        for attn, ff in self.layers:
            x = attn(x) + x[:,[0],:]     # attn(x): (p,1,64)  + (p,1,64)
            x = ff(x) + x
        return x

def SinCosin_pos_emb(N, C, group_num):
    # 计算pe编码
    pe = torch.zeros(N, C)  # 建立空表，每行代表一个词的位置，每列代表一个编码位 (L,C)
    position = torch.arange(0, N).unsqueeze(1)  # 建个arrange表示词的位置以便公式计算，size=(max_len,1)
    div_term = torch.exp(torch.arange(0, C, 2) * -(math.log(10000.0) / C))  # 计算公式中10000**（2i/d_model)

    pe[:, 0::2] = torch.sin(position * div_term)  # 计算偶数维度的pe值
    pe[:, 1::2] = torch.cos(position * div_term)  # 计算奇数维度的pe值
    pe = pe.unsqueeze(0)                          # size=(1, L, d_model)，为了后续与word_embedding相加,意为batch维度下的操作相同
    pe = pe.repeat(group_num, 1, 1).flatten(start_dim=0, end_dim=1).unsqueeze(0)   # (group_num, L, d_model)-->(group_num * L, d_model)-->(1, group_num * L, d_model)
    return pe  # (1, group_num * N, C)

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout, max_len=5000):
        """
        :param d_model: pe编码维度，一般与word embedding相同，方便相加
        :param dropout: dorp out
        :param max_len: 语料库中最长句子的长度，即word embedding中的L
        """
        super(PositionalEncoding, self).__init__()
        # 定义drop out
        self.dropout = nn.Dropout(p=dropout)
        # 计算pe编码
        pe = torch.zeros(max_len, d_model)               # 建立空表，每行代表一个词的位置，每列代表一个编码位 (L,C)
        position = torch.arange(0, max_len).unsqueeze(1) # 建个arrange表示词的位置以便公式计算，size=(max_len,1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))   # 计算公式中10000**（2i/d_model)

        pe[:, 0::2] = torch.sin(position * div_term)  # 计算偶数维度的pe值
        pe[:, 1::2] = torch.cos(position * div_term)  # 计算奇数维度的pe值
        pe = pe.unsqueeze(0)  # size=(1, L, d_model)，为了后续与word_embedding相加,意为batch维度下的操作相同
        self.register_buffer('pe', pe)  # pe值是不参加训练的

    def forward(self, x):
        # 输入的最终编码 = word_embedding + positional_embedding
        # x = x + Variable(self.pe[:, :x.size(1)],requires_grad=False) #size = [batch, L, d_model]
        # return self.dropout(x) # size = [batch, L, d_model]

        x = Variable(self.pe[:, :x.size(1)], requires_grad=False) #size = [batch, L, d_model]
        return x # size = [batch, L, d_model]

class MSA_yolov_Mutual_ViT_parallel_positionEmbedding(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0., fsize=4, group_num=4):
        super().__init__()
        self.layers = nn.ModuleList([])
        pe = SinCosin_pos_emb((fsize*2+1)**2,dim,group_num)   # SinCosin公式用来初始化 (1, group_num*N, C)
        self.pos_embedding3D = nn.Parameter(pe)  # 可学习的 (1, group_num*N, C)
        self.fsize = fsize
        self.loc = ((2*fsize)+1)**2*(group_num-1) + ((2*fsize)+1)*fsize+fsize             # 最后一帧的中间位置
        self.dropout = nn.Dropout(dropout)
        self.pos_embedding = PositionalEncoding(dim,dropout,(fsize*2+1)**2*group_num)     # (1,N,c) sin-cosin的

        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, LSA_parallel(dim, heads=heads, dim_head=dim_head, dropout=dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout))
            ]))

    def forward(self, x):   # mutual    x: (p, 401, 64)
        # 可学习的 或者 sin-cosin
        x += torch.cat([self.pos_embedding3D[[[0]],[[self.loc]],:], self.pos_embedding3D], 1)        # 可学习的。 把最后一帧(当前帧)中间的那个位置编码取出来
        # x += torch.cat([self.pos_embedding(x)[[[0]],[[self.loc]],:], self.pos_embedding(x)], 1)    # sin-cosin的 把最后一帧(当前帧)中间的那个位置编码取出来
        # x = self.dropout(x)

        for attn, ff in self.layers:
            x = attn(x) + x[:,[0],:]     # attn(x): (p,1,64)  + (p,1,64)
            x = ff(x) + x
        return x

class MSA_yolov_Mutual_ViT_parallel_block(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, LSA_parallel(dim, heads=heads, dim_head=dim_head, dropout=dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout))
            ]))

    def forward(self, x):   # mutual    x: (p, 101, 64)
        x_ori = x  # (p,101,64)
        for attn, ff in self.layers:
            x = attn(x_ori) + x_ori[:,0].unsqueeze(1)     # attn(x): (p,1,64)  + (p,1,64)
            x = ff(x) + x                                 # (p,1,64)
            x_ori[:, [0]] = x                             #
        return x

class MSA_yolov_Mutual_ViT_Block(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, LSA(dim, heads=heads, dim_head=dim_head, dropout=dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout))
            ]))

    def forward(self, x):   # mutual
        x_ori = x           # (1,101,64)
        for attn, ff in self.layers:
            x = attn(x_ori) + x_ori[0,0].unsqueeze(0).unsqueeze(0)     # attn(x): (1,1,64)  + (1,1,64)
            x = ff(x) + x                                              # (1,1,64) + (1,1,64)
            x_ori[0,0] = x
        return x

class MSA_yolov_Native_ViT(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, LSA_Native(dim, heads=heads, dim_head=dim_head, dropout=dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout))
            ]))

    def forward(self, x):   # mutual
        for attn, ff in self.layers:
            x = attn(x) + x     # attn(x): (1,401,128)  + (1,401,128)
            x = ff(x) + x
        return x

class MSA_yolov_TwoStream(nn.Module):

    def __init__(self, dim, out_dim, num_heads=4, qkv_bias=False, attn_drop=0., scale=25):
        super().__init__()
        # self.layerNorm = nn.LayerNorm(dim)
        # self.multiMSA = Native_MSA_block(dim, num_heads, qkv_bias, attn_drop, scale=scale)

        self.msa = Attention_msa_TwoStream(dim, num_heads, qkv_bias, attn_drop, scale=scale)
        self.linear1 = nn.Linear(2 * dim, 2 * dim)
        self.linear2 = nn.Linear(4 * dim, out_dim)
        self.linear3 = nn.Linear(2 * dim, out_dim)

    def find_similar_round2(self, features, sort_results):  # features: (1,480,256), sort_results:(480,480)所有预测框互相之间的相似度
        key_feature = features[0]       # (480,256)
        support_feature = features[0]
        if not self.training:
            sort_results = sort_results.to(features.dtype)
        soft_sim_feature = (sort_results @ support_feature)              # (480,256) 根据相似度进行加权求和，即论文中的”average pooling over reference features“
        cls_feature = torch.cat([soft_sim_feature, key_feature], dim=-1) # (480,512) 最终一个预测框的特征：128原始分类回归特征+128考虑了置信度的MSA+256所有向量(128+128)根据预测框最终相似度加权平均的特征
        return cls_feature

    def forward(self, x_cls, v_cls, cls_score=None, sim_thresh=0.75, ave=True, use_mask=False):
        # x_cls = self.layerNorm(x_cls)
        # x_reg = self.layerNorm(x_reg)
        trans_cls, trans_reg, sim_round2 = self.msa(x_cls, v_cls, cls_score, sim_thresh=sim_thresh, ave=ave, use_mask=use_mask)   # trans_cls: (1,480,256) x_cls:(1,480,128), x_reg:(1,480,128), cls_score:(480,), fg_score:(480,)
        msa = self.linear1(trans_cls)                      # (1,480,256)-->(1,480,256)

        if ave:
            msa = self.find_similar_round2(msa, sim_round2)    # (1,480,256) --> (480,512)
            out = self.linear2(msa)  # (480,512)--->(480,512)  最终的分类特征： cls_feat2+1fc的128维 + (cls_feat2的考虑了cls置信度的MSA特征+reg_feat的考虑了obj置信度的MSA特征)/2的128维 + 对前两者拼接的(256)按预测框最终置信度进行加权平均的256维
        else:
            out = self.linear3(msa)  # (480,512)--->(480,512)  最终的分类特征： cls_feat2+1fc的128维 + (cls_feat2的考虑了cls置信度的MSA特征+reg_feat的考虑了obj置信度的MSA特征)/2的128维 + 对前两者拼接的(256)按预测框最终置信度进行加权平均的256维

        # msa = self.find_similar_round2(msa, sim_round2)    # (1,480,256) --> (480,512)
        # out = self.linear2(msa)  # (480,512)--->(480,512)  最终的分类特征： cls_feat2+1fc的128维 + (cls_feat2的考虑了cls置信度的MSA特征+reg_feat的考虑了obj置信度的MSA特征)/2的128维 + 对前两者拼接的(256)按预测框最终置信度进行加权平均的256维

        return out

class MSA_yolov_Mutual(nn.Module):

    def __init__(self, dim, out_dim, num_heads=4, qkv_bias=False, attn_drop=0., scale=25, prior_head_feats=False):
        super().__init__()
        # self.layerNorm = nn.LayerNorm(dim)
        # self.multiMSA = Native_MSA_block(dim, num_heads, qkv_bias, attn_drop, scale=scale)

        self.msa = Attention_msa_Mutual(dim, num_heads, qkv_bias, attn_drop, scale=scale, prior_head_feats=prior_head_feats)
        self.linear1 = nn.Linear(2 * dim, 2 * dim)
        self.linear2 = nn.Linear(2 * dim, out_dim)

    def forward(self, x_cls, x_reg):
        # x_cls = self.layerNorm(x_cls)
        # x_reg = self.layerNorm(x_reg)
        trans_cls = self.msa(x_cls, x_reg)   # trans_cls: (1,480,128) x_cls:(1,480,64), x_reg:(1,480,64)
        msa = self.linear1(trans_cls)        # (1,480,128)-->(1,480,128)
        out = self.linear2(msa)              # (480,128)--->(480,64)  最终的分类特征： cls_feat2+1fc的128维 + (cls_feat2的考虑了cls置信度的MSA特征+reg_feat的考虑了obj置信度的MSA特征)/2的128维
        return out

class Native_MSA_block(nn.Module):
    def __init__(self, dim, depth=1, num_heads=4, qkv_bias=False, attn_drop=0., scale=25):
        super().__init__()
        self.msa = Native_Attention_msa(dim, num_heads, qkv_bias, attn_drop, scale=scale)  # MSA + fc(dim,dim) + drop   (BNC)-->(BNC)
        self.layer_norm = nn.LayerNorm(dim)      # (BNC)-->(BNC)
        self.layers = nn.ModuleList([])
        self.act = nn.GELU()
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                self.layer_norm,
                self.msa,
                self.act
            ]))

    def forward(self, x):   # (BNC)-->(BNC)
        for ln, msa, act in self.layers:
            x = act(msa(ln(x)) + x)
        return x

class get_Attn(nn.Module):
    def __init__(self, dim, num_heads=4, qkv_bias=False, attn_drop=0., scale=25):
        super().__init__()
        self.msa = Native_Attention_msa_onlyAttn(dim, num_heads, qkv_bias, attn_drop, scale=scale)  # MSA + fc(dim,dim) + drop   (BNC)-->(BNC)

    def forward(self, x):   # (BNC)-->(1,4,160,160)
        attn = self.msa(x)
        return attn


class MSA_yolov_visual(nn.Module):

    def __init__(self, dim,out_dim, num_heads=4, qkv_bias=False, attn_drop=0.,scale=25):
        super().__init__()
        self.msa = Attention_msa_visual(dim,num_heads,qkv_bias,attn_drop,scale=scale)
        self.linear1 = nn.Linear(2 * dim,2 * dim)
        self.linear2 =  nn.Linear(4 * dim,out_dim)

    def ave_pooling_over_ref(self,features,sort_results):
        key_feature = features[0]
        support_feature = features[0]
        if not self.training:
            sort_results = sort_results.to(features.dtype)
        soft_sim_feature = (sort_results@support_feature)#.transpose(1, 2)#torch.sum(softmax_value * most_sim_feature, dim=1)
        cls_feature = torch.cat([soft_sim_feature,key_feature],dim=-1)
        return cls_feature

    def forward(self,x_cls, x_reg, cls_score = None, fg_score = None, img = None, pred = None):
        trans_cls, trans_reg, sim_round2 = self.msa(x_cls,x_reg,cls_score,fg_score,img,pred)
        msa = self.linear1(trans_cls)
        ave = self.ave_pooling_over_ref(msa,sim_round2)
        out = self.linear2(ave)
        return out


class MSA_yolov_online(nn.Module):

    def __init__(self, dim,out_dim, num_heads=4, qkv_bias=False, attn_drop=0.,scale=25):
        super().__init__()
        self.msa = Attention_msa_online(dim,num_heads,qkv_bias,attn_drop,scale=scale)
        self.linear1 = nn.Linear(2 * dim,2 * dim)
        self.linear2 =  nn.Linear(4 * dim,out_dim)

    def ave_pooling_over_ref(self,features,sort_results):
        key_feature = features[0]
        support_feature = features[0]
        if not self.training:
            sort_results = sort_results.to(features.dtype)
        soft_sim_feature = (sort_results@support_feature)#.transpose(1, 2)#torch.sum(softmax_value * most_sim_feature, dim=1)
        cls_feature = torch.cat([soft_sim_feature,key_feature],dim=-1)

        return cls_feature

    def compute_geo_sim(self,key_preds,ref_preds):
        key_boxes = key_preds[:,:4]
        ref_boxes = ref_preds[:,:4]
        cost_giou, iou = generalized_box_iou(key_boxes.to(torch.float32), ref_boxes.to(torch.float32))

        return iou.to(torch.float16)

    def local_agg(self,features,local_results,boxes,cls_score,fg_score):
        local_features = local_results['msa']
        local_features_n = local_features / torch.norm(local_features, dim=-1, keepdim=True)
        features_n = features /torch.norm(features, dim=-1, keepdim=True)
        cos_sim = features_n@local_features_n.transpose(0,1)

        geo_sim = self.compute_geo_sim(boxes,local_results['boxes'])
        N = local_results['cls_scores'].shape[0]
        M = cls_score.shape[0]
        pre_scores = cls_score*fg_score
        pre_scores = torch.reshape(pre_scores, [-1, 1]).repeat(1, N)
        other_scores = local_results['cls_scores']*local_results['reg_scores']
        other_scores = torch.reshape(other_scores, [1, -1]).repeat(M, 1)
        ones_matrix = torch.ones([M,N]).to('cuda')
        zero_matrix = torch.zeros([M,N]).to('cuda')
        thresh_map = torch.where(other_scores-pre_scores>-0.3,ones_matrix,zero_matrix)
        local_sim = torch.softmax(25*cos_sim*thresh_map,dim=-1)*geo_sim
        local_sim = local_sim / torch.sum(local_sim, dim=-1, keepdim=True)
        local_sim = local_sim.to(features.dtype)
        sim_features = local_sim @ local_features

        return (sim_features+features)/2

    def forward(self,x_cls, x_reg, cls_score = None, fg_score = None,other_result = {},boxes=None, simN=30):
        trans_cls, trans_reg, sim_round2 = self.msa(x_cls,x_reg,cls_score,fg_score)
        msa = self.linear1(trans_cls)
        # if other_result != []:
        #     other_msa = other_result['msa'].unsqueeze(0)
        #     msa = torch.cat([msa,other_msa],dim=1)
        ave = self.ave_pooling_over_ref(msa,sim_round2)
        out = self.linear2(ave)
        if other_result != [] and other_result['local_results'] != []:
            lout = self.local_agg(out[:simN],other_result['local_results'],boxes[:simN],cls_score[:simN],fg_score[:simN])
            return lout,out
        return out,out

def visual_attention(data):
    data = data.cpu()
    data = data.detach().numpy()

    plt.xlabel('x')
    plt.ylabel('score')
    plt.imshow(data)
    plt.show()













