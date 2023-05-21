import torch
import torch.nn as nn

from .darknet import BaseConv, CSPDarknet, CSPLayer, DWConv
from .yolovp_msa import YOLOXHead_PriorBox_RegionalMSA1_OF_KeyFrame_Gray_or_OF_fineFeat


class YOLOXHead_original(nn.Module):
    def __init__(self, num_classes, width = 1.0, features = "234", act = "silu", depthwise = False):
        super().__init__()
        Conv            = DWConv if depthwise else BaseConv
        if features == "123":
            channel_factor = 0.5 * 0.5 * 2
            in_channels = [int(256 * channel_factor), int(512 * channel_factor), int(1024 * channel_factor)]
        if features == "234":
            channel_factor = 0.5 * 2
            in_channels = [int(256 * channel_factor), int(512 * channel_factor), int(1024 * channel_factor)]
        if features == "345":
            channel_factor = 1 * 2
            in_channels = [int(256 * channel_factor), int(512 * channel_factor), int(1024 * channel_factor)]
        if features == "23":
            channel_factor = 0.5 * 2
            in_channels = [int(256 * channel_factor), int(512 * channel_factor)]
        if features == "2":
            self.in_features = ("dark2")
            channel_factor = 0.5 * 2
            in_channels = [int(256 * channel_factor)]
        if features == "3":
            self.in_features = ("dark3")
            channel_factor = 1 * 2
            in_channels = [int(256 * channel_factor)]
        if features == "34":
            self.in_features = ("dark3", "dark4")
            channel_factor = 1 * 2
            in_channels = [int(256 * channel_factor), int(512 * channel_factor)]
        if features == "2345":
            channel_factor = 0.5 * 2
            in_channels = [int(256 * channel_factor), int(512 * channel_factor), int(1024 * channel_factor), int(2048 * channel_factor)]

        self.cls_convs  = nn.ModuleList()
        self.reg_convs  = nn.ModuleList()
        self.cls_preds  = nn.ModuleList()
        self.reg_preds  = nn.ModuleList()
        self.obj_preds  = nn.ModuleList()
        self.stems      = nn.ModuleList()

        for i in range(len(in_channels)):
            self.stems.append(BaseConv(in_channels = int(in_channels[i] * width), out_channels = int(256 * width), ksize = 1, stride = 1, act = act))
            self.cls_convs.append(nn.Sequential(*[
                Conv(in_channels = int(256 * width), out_channels = int(256 * width), ksize = 3, stride = 1, act = act), 
                Conv(in_channels = int(256 * width), out_channels = int(256 * width), ksize = 3, stride = 1, act = act), 
            ]))
            self.cls_preds.append(
                nn.Conv2d(in_channels = int(256 * width), out_channels = num_classes, kernel_size = 1, stride = 1, padding = 0)
            )
            

            self.reg_convs.append(nn.Sequential(*[
                Conv(in_channels = int(256 * width), out_channels = int(256 * width), ksize = 3, stride = 1, act = act), 
                Conv(in_channels = int(256 * width), out_channels = int(256 * width), ksize = 3, stride = 1, act = act)
            ]))
            self.reg_preds.append(
                nn.Conv2d(in_channels = int(256 * width), out_channels = 4, kernel_size = 1, stride = 1, padding = 0)
            )
            self.obj_preds.append(
                nn.Conv2d(in_channels = int(256 * width), out_channels = 1, kernel_size = 1, stride = 1, padding = 0)
            )

    def forward(self, inputs):
        #---------------------------------------------------#
        #   inputs输入
        #   P3_out  80, 80, 256
        #   P4_out  40, 40, 512
        #   P5_out  20, 20, 1024
        #---------------------------------------------------#
        outputs = []
        for k, x in enumerate(inputs):
            #---------------------------------------------------#
            #   利用1x1卷积进行通道整合
            #---------------------------------------------------#
            x       = self.stems[k](x)
            #---------------------------------------------------#
            #   利用两个卷积标准化激活函数来进行特征提取
            #---------------------------------------------------#
            cls_feat    = self.cls_convs[k](x)
            #---------------------------------------------------#
            #   判断特征点所属的种类
            #   80, 80, num_classes
            #   40, 40, num_classes
            #   20, 20, num_classes
            #---------------------------------------------------#
            cls_output  = self.cls_preds[k](cls_feat)

            #---------------------------------------------------#
            #   利用两个卷积标准化激活函数来进行特征提取
            #---------------------------------------------------#
            reg_feat    = self.reg_convs[k](x)
            #---------------------------------------------------#
            #   特征点的回归系数
            #   reg_pred 80, 80, 4
            #   reg_pred 40, 40, 4
            #   reg_pred 20, 20, 4
            #---------------------------------------------------#
            reg_output  = self.reg_preds[k](reg_feat)
            #---------------------------------------------------#
            #   判断特征点是否有对应的物体
            #   obj_pred 80, 80, 1
            #   obj_pred 40, 40, 1
            #   obj_pred 20, 20, 1
            #---------------------------------------------------#
            obj_output  = self.obj_preds[k](reg_feat)

            output      = torch.cat([reg_output, obj_output, cls_output], 1)   # 拼接同一特征层的三种预测头，所以拼完第二个维度为7
            outputs.append(output)
        return outputs   # list 3: [[16,7,80,80],[16,7,40,40],[16,7,20,20]]


class YOLOPAFPN(nn.Module):
    def __init__(self, depth = 1.0, width = 1.0, FPN_feature = "234", depthwise = False, act = "silu", MultiInputs = False, PAFPN_use=False):
        super().__init__()
        Conv                = DWConv if depthwise else BaseConv

        if FPN_feature == "123":
            self.in_features = ("stem", "dark2", "dark3")
            backbone_out_features = ("stem", "dark2", "dark3")
            channel_factor = 0.5 * 0.5
        if FPN_feature == "23":
            self.in_features = ("dark2", "dark3")
            backbone_out_features = ("dark2", "dark3")
            channel_factor = 0.5 * 0.5
        if FPN_feature == "2":
            self.in_features = ("dark2")
            backbone_out_features = ("dark2")
            channel_factor = 0.5
        if FPN_feature == "3":
            self.in_features = ("dark3")
            backbone_out_features = ("dark3")
            channel_factor = 1
        if FPN_feature == "34":
            self.in_features = ("dark3", "dark4")
            backbone_out_features = ("dark3", "dark4")
            channel_factor = 1
        if FPN_feature == "234":
            self.in_features = ("dark2", "dark3", "dark4")
            backbone_out_features = ("dark2", "dark3", "dark4")
            channel_factor = 0.5
        if FPN_feature == "345":
            self.in_features = ("dark3", "dark4", "dark5")
            backbone_out_features = ("dark3", "dark4", "dark5")
            channel_factor = 1
        if FPN_feature == "2345":
            channel_factor = 0.5
            self.in_features = ("dark2", "dark3", "dark4", "dark5")
            backbone_out_features = ("dark2", "dark3", "dark4", "dark5")

        in_channels = [int(256 * channel_factor), int(512 * channel_factor), int(1024 * channel_factor)]

        self.backbone       = CSPDarknet(depth, width, depthwise = depthwise, act = act, out_features=backbone_out_features)
        self.upsample       = nn.Upsample(scale_factor=2, mode="nearest")
        self.MultiInputs = MultiInputs
        self.PAFPN_use = PAFPN_use
        #-------------------------------------------#
        #   20, 20, 1024 -> 20, 20, 512
        #-------------------------------------------#
        self.lateral_conv0  = BaseConv(int(in_channels[2] * width), int(in_channels[1] * width), 1, 1, act=act)
    
        #-------------------------------------------#
        #   40, 40, 1024 -> 40, 40, 512
        #-------------------------------------------#
        self.C3_p4 = CSPLayer(
            int(2 * in_channels[1] * width),
            int(in_channels[1] * width),
            round(3 * depth),
            False,
            depthwise = depthwise,
            act = act,
        )  

        #-------------------------------------------#
        #   40, 40, 512 -> 40, 40, 256
        #-------------------------------------------#
        self.reduce_conv1   = BaseConv(int(in_channels[1] * width), int(in_channels[0] * width), 1, 1, act=act)
        #-------------------------------------------#
        #   80, 80, 512 -> 80, 80, 256
        #-------------------------------------------#
        self.C3_p3 = CSPLayer(
            int(2 * in_channels[0] * width),
            int(in_channels[0] * width),
            round(3 * depth),
            False,
            depthwise = depthwise,
            act = act,
        )

        #-------------------------------------------#
        #   80, 80, 256 -> 40, 40, 256
        #-------------------------------------------#
        self.bu_conv2       = Conv(int(in_channels[0] * width), int(in_channels[0] * width), 3, 2, act=act)
        #-------------------------------------------#
        #   40, 40, 256 -> 40, 40, 512
        #-------------------------------------------#
        self.C3_n3 = CSPLayer(
            int(2 * in_channels[0] * width),
            int(in_channels[1] * width),
            round(3 * depth),
            False,
            depthwise = depthwise,
            act = act,
        )

        #-------------------------------------------#
        #   40, 40, 512 -> 20, 20, 512
        #-------------------------------------------#
        self.bu_conv1       = Conv(int(in_channels[1] * width), int(in_channels[1] * width), 3, 2, act=act)
        #-------------------------------------------#
        #   20, 20, 1024 -> 20, 20, 1024
        #-------------------------------------------#
        self.C3_n4 = CSPLayer(
            int(2 * in_channels[1] * width),
            int(in_channels[2] * width),
            round(3 * depth),
            False,
            depthwise = depthwise,
            act = act,
        )

    def forward(self, input):
        if self.MultiInputs:
            out_features            = self.backbone.forward(input[0])   # input = [images, images_ids]
        else:
            out_features = self.backbone.forward(input)

        if len(self.in_features) == 4:
            [feat1, feat2, feat3, feat4]   = [out_features[f] for f in self.in_features]    # 选取哪几个特征层，默认 dark3,4,5
        if len(self.in_features) == 3:
            [feat1, feat2, feat3]   = [out_features[f] for f in self.in_features]    # 选取哪几个特征层，默认 dark3,4,5
        if len(self.in_features) == 2:
            [feat1, feat2]   = [out_features[f] for f in self.in_features]    # 选取哪几个特征层，默认 dark3,4,5
        if len(self.in_features) == 1:
            [feat1]   = [out_features[f] for f in self.in_features]    # 选取哪几个特征层，默认 dark3,4,5

        if self.PAFPN_use:
            #-------------------------------------------#
            #   20, 20, 1024 -> 20, 20, 512
            #-------------------------------------------#
            P5          = self.lateral_conv0(feat3)           # (128,40,40)
            #-------------------------------------------#
            #  20, 20, 512 -> 40, 40, 512
            #-------------------------------------------#
            P5_upsample = self.upsample(P5)
            #-------------------------------------------#
            #  40, 40, 512 + 40, 40, 512 -> 40, 40, 1024
            #-------------------------------------------#
            P5_upsample = torch.cat([P5_upsample, feat2], 1)
            #-------------------------------------------#
            #   40, 40, 1024 -> 40, 40, 512
            #-------------------------------------------#
            P5_upsample = self.C3_p4(P5_upsample)

            #-------------------------------------------#
            #   40, 40, 512 -> 40, 40, 256
            #-------------------------------------------#
            P4          = self.reduce_conv1(P5_upsample)
            #-------------------------------------------#
            #   40, 40, 256 -> 80, 80, 256
            #-------------------------------------------#
            P4_upsample = self.upsample(P4)
            #-------------------------------------------#
            #   80, 80, 256 + 80, 80, 256 -> 80, 80, 512
            #-------------------------------------------#
            P4_upsample = torch.cat([P4_upsample, feat1], 1)
            #-------------------------------------------#
            #   80, 80, 512 -> 80, 80, 256
            #-------------------------------------------#
            P3_out      = self.C3_p3(P4_upsample)

            #-------------------------------------------#
            #   80, 80, 256 -> 40, 40, 256
            #-------------------------------------------#
            P3_downsample   = self.bu_conv2(P3_out)
            #-------------------------------------------#
            #   40, 40, 256 + 40, 40, 256 -> 40, 40, 512
            #-------------------------------------------#
            P3_downsample   = torch.cat([P3_downsample, P4], 1)
            #-------------------------------------------#
            #   40, 40, 256 -> 40, 40, 512
            #-------------------------------------------#
            P4_out          = self.C3_n3(P3_downsample)

            #-------------------------------------------#
            #   40, 40, 512 -> 20, 20, 512
            #-------------------------------------------#
            P4_downsample   = self.bu_conv1(P4_out)
            #-------------------------------------------#
            #   20, 20, 512 + 20, 20, 512 -> 20, 20, 1024
            #-------------------------------------------#
            P4_downsample   = torch.cat([P4_downsample, P5], 1)
            #-------------------------------------------#
            #   20, 20, 1024 -> 20, 20, 1024
            #-------------------------------------------#
            P5_out          = self.C3_n4(P4_downsample)

            if self.MultiInputs:
                return (P3_out, P4_out, P5_out, input[1])   # input[1]:images_ids
            else:
                return (P3_out, P4_out, P5_out)   # (80,80,64) (40,40,128) (20,20,256)

        else:
            if self.MultiInputs:
                if len(self.in_features) == 4:
                    return (feat1, feat2, feat3, feat4, input[1])   # input[1]:images_ids
                if len(self.in_features) == 3:
                    return (feat1, feat2, feat3, input[1])   # input[1]:images_ids
                if len(self.in_features) == 2:
                    return (feat1, feat2, input[1])
                if len(self.in_features) == 1:
                    return (feat1, input[1])
            else:
                if len(self.in_features) == 4:
                    return (feat1, feat2, feat3, feat4)   # (80,80,64) (40,40,128) (20,20,256)
                if len(self.in_features) == 3:
                    return (feat1, feat2, feat3)   # (80,80,64) (40,40,128) (20,20,256)
                if len(self.in_features) == 2:
                    return (feat1, feat2)
                if len(self.in_features) == 1:
                    return (feat1)


class YOLOPAFPN_TwoStream(nn.Module):
    def __init__(self, depth=1.0, width=1.0, features="234", depthwise=False, act="silu", MultiInputs=True,
                 PAFPN_use=False):
        super().__init__()
        Conv = DWConv if depthwise else BaseConv

        if PAFPN_use == False:
            if features == "123":
                self.in_features = ("stem", "dark2", "dark3")
                backbone_out_features = ("stem", "dark2", "dark3")
                channel_factor = 0.5 * 0.5
            if features == "23":
                self.in_features = ("dark2", "dark3")
                backbone_out_features = ("dark2", "dark3")
                channel_factor = 0.5 * 0.5
            if features == "2":
                self.in_features = ("dark2")
                backbone_out_features = ("dark2")
                channel_factor = 0.5
            if features == "3":
                self.in_features = ("dark3")
                backbone_out_features = ("dark3")
                channel_factor = 1
            if features == "34":
                self.in_features = ("dark3", "dark4")
                backbone_out_features = ("stem", "dark2", "dark3", "dark4")
                channel_factor = 1
            if features == "234":
                self.in_features = ("dark2", "dark3", "dark4")
                backbone_out_features = ("dark2", "dark3", "dark4")
                channel_factor = 0.5
            if features == "345":
                self.in_features = ("dark3", "dark4", "dark5")
                backbone_out_features = ("dark3", "dark4", "dark5")
                channel_factor = 1
            if features == "2345":
                channel_factor = 0.5
                self.in_features = ("dark2", "dark3", "dark4", "dark5")
                backbone_out_features = ("dark2", "dark3", "dark4", "dark5")

        in_channels = [int(256 * channel_factor), int(512 * channel_factor), int(1024 * channel_factor)]

        self.backbone_out_features = backbone_out_features
        self.backbone = CSPDarknet(depth, width, depthwise=depthwise, act=act, out_features=backbone_out_features)
        self.backbone_Of = CSPDarknet(depth, width, depthwise=depthwise, act=act, out_features=backbone_out_features)
        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")
        self.MultiInputs = MultiInputs
        self.PAFPN_use = PAFPN_use
        # -------------------------------------------#
        #   20, 20, 1024 -> 20, 20, 512
        # -------------------------------------------#
        self.lateral_conv0 = BaseConv(int(in_channels[2] * width), int(in_channels[1] * width), 1, 1, act=act)

        # -------------------------------------------#
        #   40, 40, 1024 -> 40, 40, 512
        # -------------------------------------------#
        self.C3_p4 = CSPLayer(
            int(2 * in_channels[1] * width),
            int(in_channels[1] * width),
            round(3 * depth),
            False,
            depthwise=depthwise,
            act=act,
        )

        # -------------------------------------------#
        #   40, 40, 512 -> 40, 40, 256
        # -------------------------------------------#
        self.reduce_conv1 = BaseConv(int(in_channels[1] * width), int(in_channels[0] * width), 1, 1, act=act)
        # -------------------------------------------#
        #   80, 80, 512 -> 80, 80, 256
        # -------------------------------------------#
        self.C3_p3 = CSPLayer(
            int(2 * in_channels[0] * width),
            int(in_channels[0] * width),
            round(3 * depth),
            False,
            depthwise=depthwise,
            act=act,
        )

        # -------------------------------------------#
        #   80, 80, 256 -> 40, 40, 256
        # -------------------------------------------#
        self.bu_conv2 = Conv(int(in_channels[0] * width), int(in_channels[0] * width), 3, 2, act=act)
        # -------------------------------------------#
        #   40, 40, 256 -> 40, 40, 512
        # -------------------------------------------#
        self.C3_n3 = CSPLayer(
            int(2 * in_channels[0] * width),
            int(in_channels[1] * width),
            round(3 * depth),
            False,
            depthwise=depthwise,
            act=act,
        )

        # -------------------------------------------#
        #   40, 40, 512 -> 20, 20, 512
        # -------------------------------------------#
        self.bu_conv1 = Conv(int(in_channels[1] * width), int(in_channels[1] * width), 3, 2, act=act)
        # -------------------------------------------#
        #   20, 20, 1024 -> 20, 20, 1024
        # -------------------------------------------#
        self.C3_n4 = CSPLayer(
            int(2 * in_channels[1] * width),
            int(in_channels[2] * width),
            round(3 * depth),
            False,
            depthwise=depthwise,
            act=act,
        )

    def forward(self, input):
        if self.MultiInputs:
            out_features = self.backbone.forward(input[0][:,0:3,:,:])  # input = [images, images_ids]
            out_features_Of = self.backbone_Of.forward(input[0][:,3:,:,:])  # input = [images, images_ids]
        else:
            out_features = self.backbone.forward(input[:,0:3,:,:])
            out_features_Of = self.backbone_Of.forward(input[:,3:,:,:])

        if len(self.in_features) == 4:
            [feat1, feat2, feat3, feat4] = [out_features[f] for f in self.in_features]                 # 选取哪几个特征层，默认 dark3,4,5
            [feat1_Of, feat2_Of, feat3_Of, feat4_Of] = [out_features_Of[f] for f in self.in_features]  # 选取哪几个特征层，默认 dark3,4,5
            feat1 = torch.cat([feat1, feat1_Of],1)
            feat2 = torch.cat([feat2, feat2_Of],1)
            feat3 = torch.cat([feat3, feat3_Of],1)
            feat4 = torch.cat([feat4, feat4_Of],1)

        if len(self.in_features) == 3:
            [feat1, feat2, feat3] = [out_features[f] for f in self.in_features]  # 选取哪几个特征层，默认 dark3,4,5
            [feat1_Of, feat2_Of, feat3_Of] = [out_features_Of[f] for f in self.in_features]  # 选取哪几个特征层，默认 dark3,4,5
            feat1 = torch.cat([feat1, feat1_Of],1)
            feat2 = torch.cat([feat2, feat2_Of],1)
            feat3 = torch.cat([feat3, feat3_Of],1)

        if len(self.in_features) == 2:
            [stem, dark2, feat1, feat2] = [out_features[f] for f in self.backbone_out_features]  # 选取哪几个特征层，默认 dark3,4,5
            [stem_of, dark2_of, feat1_Of, feat2_Of] = [out_features_Of[f] for f in self.backbone_out_features]  # 选取哪几个特征层，默认 dark3,4,5
            feat1 = torch.cat([feat1, feat1_Of],1)
            feat2 = torch.cat([feat2, feat2_Of],1)

        if len(self.in_features) == 1:
            [feat1] = [out_features[f] for f in self.in_features]  # 选取哪几个特征层，默认 dark3,4,5
            [feat1_Of] = [out_features_Of[f] for f in self.in_features]  # 选取哪几个特征层，默认 dark3,4,5
            feat1 = torch.cat([feat1, feat1_Of],1)

        if self.PAFPN_use:
            # -------------------------------------------#
            #   20, 20, 1024 -> 20, 20, 512
            # -------------------------------------------#
            P5 = self.lateral_conv0(feat3)  # (128,40,40)
            # -------------------------------------------#
            #  20, 20, 512 -> 40, 40, 512
            # -------------------------------------------#
            P5_upsample = self.upsample(P5)
            # -------------------------------------------#
            #  40, 40, 512 + 40, 40, 512 -> 40, 40, 1024
            # -------------------------------------------#
            P5_upsample = torch.cat([P5_upsample, feat2], 1)
            # -------------------------------------------#
            #   40, 40, 1024 -> 40, 40, 512
            # -------------------------------------------#
            P5_upsample = self.C3_p4(P5_upsample)

            # -------------------------------------------#
            #   40, 40, 512 -> 40, 40, 256
            # -------------------------------------------#
            P4 = self.reduce_conv1(P5_upsample)
            # -------------------------------------------#
            #   40, 40, 256 -> 80, 80, 256
            # -------------------------------------------#
            P4_upsample = self.upsample(P4)
            # -------------------------------------------#
            #   80, 80, 256 + 80, 80, 256 -> 80, 80, 512
            # -------------------------------------------#
            P4_upsample = torch.cat([P4_upsample, feat1], 1)
            # -------------------------------------------#
            #   80, 80, 512 -> 80, 80, 256
            # -------------------------------------------#
            P3_out = self.C3_p3(P4_upsample)

            # -------------------------------------------#
            #   80, 80, 256 -> 40, 40, 256
            # -------------------------------------------#
            P3_downsample = self.bu_conv2(P3_out)
            # -------------------------------------------#
            #   40, 40, 256 + 40, 40, 256 -> 40, 40, 512
            # -------------------------------------------#
            P3_downsample = torch.cat([P3_downsample, P4], 1)
            # -------------------------------------------#
            #   40, 40, 256 -> 40, 40, 512
            # -------------------------------------------#
            P4_out = self.C3_n3(P3_downsample)

            # -------------------------------------------#
            #   40, 40, 512 -> 20, 20, 512
            # -------------------------------------------#
            P4_downsample = self.bu_conv1(P4_out)
            # -------------------------------------------#
            #   20, 20, 512 + 20, 20, 512 -> 20, 20, 1024
            # -------------------------------------------#
            P4_downsample = torch.cat([P4_downsample, P5], 1)
            # -------------------------------------------#
            #   20, 20, 1024 -> 20, 20, 1024
            # -------------------------------------------#
            P5_out = self.C3_n4(P4_downsample)

            if self.MultiInputs:
                return (P3_out, P4_out, P5_out, input[1])  # input[1]:images_ids
            else:
                return (P3_out, P4_out, P5_out)  # (80,80,64) (40,40,128) (20,20,256)

        else:
            if self.MultiInputs:
                if len(self.in_features) == 4:
                    return (feat1, feat2, feat3, feat4, input[1])  # input[1]:images_ids
                if len(self.in_features) == 3:
                    return (feat1, feat2, feat3, input[1])  # input[1]:images_ids
                if len(self.in_features) == 2:
                    return (feat1, feat2, input[1], stem, dark2, stem_of, dark2_of)
                if len(self.in_features) == 1:
                    return (feat1, input[1])
            else:
                if len(self.in_features) == 4:
                    return (feat1, feat2, feat3, feat4)  # (80,80,64) (40,40,128) (20,20,256)
                if len(self.in_features) == 3:
                    return (feat1, feat2, feat3)  # (80,80,64) (40,40,128) (20,20,256)
                if len(self.in_features) == 2:
                    return (feat1, feat2)
                if len(self.in_features) == 1:
                    return (feat1)


class YoloBody(nn.Module):
    def __init__(self, num_classes, phi, MultiInputs = False, group_num = 16, features="234", PAFPN_use=False):
        super().__init__()
        depth_dict = {'nano': 0.33, 'tiny': 0.33, 's' : 0.33, 'm' : 0.67, 'l' : 1.00, 'x' : 1.33,}        # 决定层数  只影响backbone+fpn
        width_dict = {'nano': 0.25, 'tiny': 0.375, 's' : 0.50, 'm' : 0.75, 'l' : 1.00, 'x' : 1.25,}       # 决定通道数 只影响head
        depth, width    = depth_dict[phi], width_dict[phi]
        depthwise       = True if phi == 'nano' else False

        self.MultiInputs = MultiInputs
        self.backbone   = YOLOPAFPN_TwoStream(depth, width, depthwise=depthwise, features = features, PAFPN_use=PAFPN_use, MultiInputs = MultiInputs)        # 要设置PAFPN不同头的输入通道数
        if MultiInputs:
            self.head = YOLOXHead_PriorBox_RegionalMSA1_OF_KeyFrame_Gray_or_OF_fineFeat(num_classes, width, features=features, heads=1, depthwise=depthwise,group_num=group_num)
        else:
            self.head = YOLOXHead_original(num_classes, width, features=features, depthwise=depthwise)


    def forward(self, x):
        fpn_outs    = self.backbone.forward(x)   # (feat1, feat2, input[1], stem, dark2, feat1_Of, feat2_Of)
        if self.MultiInputs:
            outputs, VID_fc, pred_result, pred_idx = self.head.forward(fpn_outs)
            return outputs, VID_fc, pred_result, pred_idx
        else:
            outputs     = self.head.forward(fpn_outs)
            return outputs

