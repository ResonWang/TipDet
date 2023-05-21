import os
from loguru import logger
import torch
import matplotlib
matplotlib.use('Agg')
import scipy.signal
from matplotlib import pyplot as plt
from torch.utils.tensorboard import SummaryWriter
import torchvision
import shutil
import numpy as np

from PIL import Image
from tqdm import tqdm
from .utils import cvtColor, preprocess_input, resize_image, preprocess_input_of
from .utils_bbox import decode_outputs, non_max_suppression
from .utils_map import get_coco_map, get_map
import copy
import datetime

class LossHistory():
    def __init__(self, log_dir, model, input_shape):
        self.log_dir    = log_dir
        self.losses     = []
        self.val_loss   = []
        
        os.makedirs(self.log_dir)
        self.writer     = SummaryWriter(self.log_dir)
        try:
            dummy_input     = torch.randn(2, 3, input_shape[0], input_shape[1])
            self.writer.add_graph(model, dummy_input)
        except:
            pass

    def append_loss(self, epoch, loss, val_loss):
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)

        self.losses.append(loss)
        self.val_loss.append(val_loss)

        with open(os.path.join(self.log_dir, "epoch_loss.txt"), 'a') as f:
            f.write(str(loss))
            f.write("\n")
        with open(os.path.join(self.log_dir, "epoch_val_loss.txt"), 'a') as f:
            f.write(str(val_loss))
            f.write("\n")

        self.writer.add_scalar('loss', loss, epoch)
        self.writer.add_scalar('val_loss', val_loss, epoch)
        self.loss_plot()

    def loss_plot(self):
        iters = range(len(self.losses))

        plt.figure()
        plt.plot(iters, self.losses, 'red', linewidth = 2, label='train loss')
        plt.plot(iters, self.val_loss, 'coral', linewidth = 2, label='val loss')
        try:
            if len(self.losses) < 25:
                num = 5
            else:
                num = 15
            
            plt.plot(iters, scipy.signal.savgol_filter(self.losses, num, 3), 'green', linestyle = '--', linewidth = 2, label='smooth train loss')
            plt.plot(iters, scipy.signal.savgol_filter(self.val_loss, num, 3), '#8B4513', linestyle = '--', linewidth = 2, label='smooth val loss')
        except:
            pass

        plt.grid(True)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend(loc="upper right")

        plt.savefig(os.path.join(self.log_dir, "epoch_loss.png"))

        plt.cla()
        plt.close("all")

class EvalCallback():
    def __init__(self, net, input_shape, class_names, num_classes, val_lines, log_dir, cuda, \
            map_out_path=".temp_map_out", max_boxes=100, confidence=0.05, nms_iou=0.5, letterbox_image=True, MINOVERLAP=0.5, eval_flag=True, period=1, MultiInputs=True, group_num=4):
        super(EvalCallback, self).__init__()

        current_time = datetime.datetime.now()

        self.net                = net
        self.input_shape        = input_shape
        self.class_names        = class_names
        self.num_classes        = num_classes
        self.val_lines          = val_lines
        self.log_dir            = log_dir
        self.cuda               = cuda
        self.map_out_path       = map_out_path + str(current_time)
        self.max_boxes          = max_boxes
        self.confidence         = confidence
        self.nms_iou            = nms_iou
        self.letterbox_image    = letterbox_image
        self.MINOVERLAP         = MINOVERLAP
        self.eval_flag          = eval_flag
        self.period             = period
        self.MultiInputs    = MultiInputs
        self.group_num = group_num
        
        self.maps       = [0]
        self.epoches    = [0]
        if self.eval_flag:
            with open(os.path.join(self.log_dir, "epoch_map.txt"), 'a') as f:
                f.write(str(0))
                f.write("\n")

    def yolo_correct_boxes(self, box_xy, box_wh, input_shape, image_shape, letterbox_image):
        # -----------------------------------------------------------------#
        #   把y轴放前面是因为方便预测框和图像的宽高进行相乘
        # -----------------------------------------------------------------#
        box_yx = box_xy[..., ::-1]          # 倒序
        box_hw = box_wh[..., ::-1]
        input_shape = np.array(input_shape)
        image_shape = np.array(image_shape)

        if letterbox_image:
            # -----------------------------------------------------------------#
            #   这里求出来的offset是图像有效区域相对于图像左上角的偏移情况
            #   new_shape指的是宽高缩放情况
            # -----------------------------------------------------------------#
            new_shape = np.round(image_shape * np.min(input_shape / image_shape))  # [480,640]
            offset = (input_shape - new_shape) / 2. / input_shape
            scale = input_shape / new_shape

            box_yx = (box_yx - offset) * scale
            box_hw *= scale

        box_mins = box_yx - (box_hw / 2.)
        box_maxes = box_yx + (box_hw / 2.)
        boxes = np.concatenate([box_mins[..., 0:1], box_mins[..., 1:2], box_maxes[..., 0:1], box_maxes[..., 1:2]],
                               axis=-1)  # 转为(x1,y1,x2,y2)
        boxes *= np.concatenate([image_shape, image_shape], axis=-1)
        return boxes

    def fuse_VID_cls_pred(self, prediction, num_classes, fc_outputs, input_shape, image_shape, letterbox_image, conf_thre=0.001, nms_thre=0.5, print_process=True):
        # 根据yolox预测结果prediction以及VID分类结果fc_outputs，得到最终预测结果[list 16] 用于存放每张图片中的VID预测框结果，每个元素是(XXX,7)
        # 遍历每一帧，坐标归一化，用VID分类结果替换yolox分类结果
        # prediction: list 16, 每个元素是(10,9) 9: (x1, y1, x2, y2, obj_conf_yolox, class_conf_yolox, class_pred_yolox, each_class_conf_yolox) 其中x1, y1, x2, y2为640尺寸下的绝对值
        # fc_outputs: (16,10,2)
        output = [None for _ in range(len(prediction))]  # [16xNone] 用于存放每张图片中的原始YOLOX预测框结果，每个元素是(XX,7)
        output_ori = [None for _ in range(len(prediction))]  # [16xNone] 用于存放每张图片中的VID预测框结果，每个元素是(XXX,7)
        prediction_ori = copy.deepcopy(prediction)  # [16x(10,7)] 10:单帧预测框数量 7:(x1, y1, x2, y2, obj_conf_yolox, class_conf_yolox, class_pred_yolox, each_class_conf_yolox)
        # cls_conf, cls_pred = torch.max(fc_outputs, -1, keepdim=False)  # (16,10) 每个框的VID cls 预测类别 及 对应类别的logits
        obj_refined_conf = fc_outputs.squeeze(-1)  # (16,10) 每个框的VID cls 预测类别 及 对应类别的logits
        for i, detections in enumerate(prediction):  # detections: (10,7) 10:每帧10个框 7： 2个类别+bbox+obj
            # 遍历每一帧
            if not detections.size(0):
                continue

            # -----------------#
            #   归一化bbox
            # -----------------#
            detections[..., [0, 2]] = detections[..., [0, 2]] / input_shape[1]
            detections[..., [1, 3]] = detections[..., [1, 3]] / input_shape[0]

            prediction_ori[i][..., [0, 2]] = prediction_ori[i][..., [0, 2]] / input_shape[1]
            prediction_ori[i][..., [1, 3]] = prediction_ori[i][..., [1, 3]] / input_shape[0]

            # cls refine
            # detections[:, 5] = cls_conf[i].sigmoid()  # 将VID的分类logits转为概率并替换
            # detections[:, 6] = cls_pred[i]  # 将VID分类结果(属于哪一个类)替换原来yolox的
            # tmp_cls_score = fc_outputs[i].sigmoid()  # [10,num_class] 单帧所有10个预测框的2个类VID分类概率
            # cls_mask = tmp_cls_score >= conf_thre  # [10,num_class] VID分类概率是否大于阈值0.001
            # cls_loc = torch.where(cls_mask)  # 找出非0元素的位置，tuple 2：((m,),(m,)) 行号，列号，这里m=20，即全满足条件
            # scores = torch.gather(tmp_cls_score[cls_loc[0]], dim=-1, index=cls_loc[1].unsqueeze(1))  # 根据cls_loc取值，(20,1): box1_vid_cls0_prob, box1_vid_cls1_prob, box2_vid_cls0_prob, box2_vid_cls1_prob, ...
            # detections[:, -num_classes:] = tmp_cls_score  # 将原始Yolox的2类分类概率改为VID分类后的 (10,7)
            # detections_raw = detections[:,:7]  # detections_raw: (10,7) (x1, y1, x2, y2, obj_conf_yolox, class_conf_VID, class_pred_VID) 其中class_conf, class_pred为VID分类后的
            # new_detetions = detections_raw[cls_loc[0]]  # (10,7)-->(20,7) 其中每2行的7个值都是相同的，即2行、2行的重复，共重复10轮
            # new_detetions[:, -1] = cls_loc[1]  # (20,7) 第7列改为num_class个类别的类别ID，0~(num_class-1)循环
            # new_detetions[:,5] = scores.squeeze()  # (20,7) 第6列cls_prob替换为VID分类概率  new_detetions:(10xnum_class,7) 7:(x1, y1, x2, y2, obj_conf_yolox, class_conf_VID, cls_loc[1]),注意，每2行的前5个值，即bbox+obj相同，但第一行的第6列为class 0 概率，第7列为0， 第二行的第6列为class 1 概率，第7列为1
            # detections_high = new_detetions  # (20,7) new_detetions

            # obj refine
            detections[:, 4] = obj_refined_conf[i].sigmoid()  # 将VID的分类logits转为概率并替换
            detections_high = detections[:,:]  # (20,7) new_detetions

            detections_ori = prediction_ori[i]  # yolox部分的预测结果：(10,7) 7:(x1, y1, x2, y2, obj_conf_yolox, class_conf_yolox, class_pred_yolox, each_class_conf_yolox)
            # print(len(detections_high.shape))

            conf_mask = (detections_high[:, 4] * detections_high[:,5] >= conf_thre).squeeze()  # (10xnum_class,) (objxcls_prob_vid)>阈值的
            detections_high = detections_high[conf_mask]  # (xx,7) 单帧VID预测框(未NMS之前的)
            # if print_process:
            #     print("---------------frame {} eval--------------------".format(i))
            #     print("---------------VID before NMS-------------------")
            #     print("{} predicted box above conf_thes {}".format(detections_high.shape[0], conf_thre))
            #     print("obj_prob:", detections_high[:, 4].detach().cpu().numpy(), "VID_cls_prob:",detections_high[:, 5].detach().cpu().numpy())
            #     print("------------------------------------------------")

            if not detections_high.shape[0]:
                continue
            if len(detections_high.shape) == 3:
                detections_high = detections_high[0]
            # VID NMS
            nms_out_index = torchvision.ops.nms(
                # 逐类别对VID分类后的预测框nms，对于第一类的，也就是第0行，第2行，第4行,...的框拿过来nms；对于第二类，也就是第1行，第3行，第5行,...的框拿过来nms。相当于对所有框进行nms,不考虑类别
                detections_high[:, :4],
                detections_high[:, 4] * detections_high[:, 5],
                nms_thre,
            )
            # 对原始Yolox预测结果进行逐类别NMS. 这里为什么还要进行一次NMS，以为第一次NMS的阈值为0.75，比较粗糙的筛选，这里为0.5，筛掉的框更多。单注意这里是逐类别筛选，上面VID NMS是不分类别的
            detections_high = detections_high[nms_out_index]  # (n,7) 7: (x1, y1, x2, y2, obj_conf_yolox, class_conf_VID, cls_loc[1])

            # if print_process:
            #     print("---------------VID after NMS-------------------")
            #     print("{} predicted box above conf_thes {}".format(detections_high.shape[0], conf_thre))
            #     print("obj_prob:{},VID_cls_prob:{},ID:{}".format(detections_high[:, 4].detach().cpu().numpy(),detections_high[:, 5].detach().cpu().numpy(),detections_high[:, 6].detach().cpu().numpy()))
            #     print("-----------------------------------------------")

            output[i] = detections_high  # output: list 16  list[i] = (200,7)

            detections_ori = detections_ori[:, :]  # (10,7) 10: 10个预测框
            conf_mask = (detections_ori[:, 4] * detections_ori[:, 5] >= conf_thre).squeeze()
            detections_ori = detections_ori[conf_mask]

            # if print_process:
            #     print("------------------yolox before NMS-------------------")
            #     print("{} predicted box above conf_thes {}".format(detections_ori.shape[0], conf_thre))
            #     print("obj_prob:", detections_ori[:, 4].detach().cpu().numpy(), "VID_cls_prob:",detections_ori[:, 5].detach().cpu().numpy())
            #     print("-----------------------------------------------------")
            # yolox NMS
            nms_out_index = torchvision.ops.nms(
                detections_ori[:, :4],
                detections_ori[:, 4] * detections_ori[:, 5],
                nms_thre,
            )
            detections_ori = detections_ori[nms_out_index]  # (x,7)
            output_ori[i] = detections_ori

            # if print_process:
            #     print("------------------yolox after NMS0.5-------------------")
            #     print("{} predicted box above conf_thes {}".format(detections_ori.shape[0], conf_thre))
            #     print("obj_prob:", detections_ori[:, 4].detach().cpu().numpy(), "VID_cls_prob:",detections_ori[:, 5].detach().cpu().numpy())
            #     print("-------------------------------------------------------")

        output_cpu = []
        output_ori_cpu = []
        for i in range(len(output)):
            if output[i] is None:
                output_cpu.append(output[i])
            else:
                output[i] = output[i].cpu().numpy()   # (xxx, 7)
                box_xy, box_wh = (output[i][:, 0:2] + output[i][:, 2:4]) / 2, output[i][:, 2:4] - output[i][:,0:2]  # 640下归一化的(x1,y1,x2,y2) 转 640下归一化的 x,y,w,h
                output[i][:, :4] = self.yolo_correct_boxes(box_xy, box_wh, input_shape, image_shape, letterbox_image)  # 还原到原图尺寸下的(x1,y1,x2,y2)，注意不是640x640，而是原始图像的大小image_shape
                output_cpu.append(output[i])
        for i in range(len(output_ori)):
            if output_ori[i] is None:
                output_ori_cpu.append(output_ori[i])
            else:
                output_ori[i] = output_ori[i].cpu().numpy()   # (xxx, 7)
                box_xy, box_wh = (output_ori[i][:, 0:2] + output_ori[i][:, 2:4]) / 2, output_ori[i][:, 2:4] - output_ori[i][:,0:2]  # 640下归一化的(x1,y1,x2,y2) 转 640下归一化的 x,y,w,h
                output_ori[i][:, :4] = self.yolo_correct_boxes(box_xy, box_wh, input_shape, image_shape, letterbox_image)  # 还原到原图尺寸下的(x1,y1,x2,y2)，注意不是640x640，而是原始图像的大小image_shape
                output_ori_cpu.append(output_ori[i])

        return output_cpu, output_ori_cpu  # [list 16] 用于存放每张图片中的VID预测框结果，每个元素是(XXX,7) [list 16] 用于存放每张图片中

    def get_map_txt(self, image_id, image, of_rgb, class_names, map_out_path):
        f = open(os.path.join(map_out_path, "detection-results/"+image_id+".txt"),"w") 
        image_shape = np.array(np.shape(image)[0:2])
        #---------------------------------------------------------#
        #   在这里将图像转换成RGB图像，防止灰度图在预测时报错。
        #   代码仅仅支持RGB图像的预测，所有其它类型的图像都会转化成RGB
        #---------------------------------------------------------#
        image       = cvtColor(image)
        of_rgb = cvtColor(of_rgb)
        #---------------------------------------------------------#
        #   给图像增加灰条，实现不失真的resize
        #   也可以直接resize进行识别
        #---------------------------------------------------------#
        image_data  = resize_image(image, (self.input_shape[1],self.input_shape[0]), self.letterbox_image)
        of_rgb  = resize_image(of_rgb, (self.input_shape[1],self.input_shape[0]), self.letterbox_image)

        #---------------------------------------------------------#
        #   添加上batch_size维度
        #---------------------------------------------------------#
        image = preprocess_input(np.array(image_data, dtype=np.float32)) # preprocess_input: 用coco数据集进行标准化, /255 -mean /std
        of_rgb = preprocess_input(np.array(of_rgb, dtype=np.float32)) # preprocess_input: 用coco数据集进行标准化, /255 -mean /std


        if self.coordConv:
            x = np.linspace(-1, 1, image.shape[0])
            y = np.linspace(-1, 1, image.shape[1])
            X,Y = np.meshgrid(x,y)
            coords = np.zeros((image.shape[0],image.shape[1],2))
            coords[...,0] = X
            coords[...,1] = Y
            image = np.concatenate((image,coords),axis=-1)
        #---------------------------------------------------------#
        #   添加上batch_size维度
        #---------------------------------------------------------#
        image_data  = np.expand_dims(np.transpose(image, (2, 0, 1)), 0).astype(np.float32)
        of_rgb_data = np.expand_dims(np.transpose(of_rgb, (2, 0, 1)), 0).astype(np.float32)
        image_data = np.concatenate([image_data,of_rgb_data],1)

        with torch.no_grad():
            images = torch.from_numpy(image_data)
            if self.cuda:
                images = images.cuda()
            # ---------------------------------------------------------#
            #   将图像输入网络当中进行预测！
            # ---------------------------------------------------------#
            outputs = self.net(images)                           # 三个特征层的预测头输出 list 3: [(16,7,80,80),(16,7,40,40),(16,7,20,20)]
            outputs = decode_outputs(outputs, self.input_shape)  # (16,8400,7) 7：(x,y,w,h,obj_prob,c0_prob,c1_prob) 其中x,y,w,h为640下的框坐标，但/640进行了归一化
            # ---------------------------------------------------------#
            #   将预测框进行堆叠，然后进行非极大抑制. 先从x,y,w,h-->x1,y1,x2,y2(归一化的)-->8400里的每一个，根据obj_prob*cls_prob进行初筛(conf_thres)-->NMS再筛(nms_thres)-->x,y,w,h(还是归一化的)-->x1,y1,x2,y2：原始图像上的绝对值(image_shape)
            #   result: list:1, 1个元素为(xxx,7)  7的内容为：原图尺寸下的x1, y1, x2, y2, obj_conf, class_conf, class_pred
            # ---------------------------------------------------------#
            results = non_max_suppression(outputs, self.num_classes, self.input_shape,
                                          image_shape, self.letterbox_image, conf_thres=self.confidence,
                                          nms_thres=self.nms_iou)

            if results[0] is None:
                return

            top_label = np.array(results[0][:, 6], dtype='int32')  # 框类别ID
            top_conf = results[0][:, 4] * results[0][:, 5]     # 最终置信度为obj * cls
            top_boxes = results[0][:, :4]                      # 最终边界框为原始图像大小(非640)下的x1, y1, x2, y2 (列，行)

        top_100 = np.argsort(top_label)[::-1][:self.max_boxes]  # 从里面最多选100个
        top_boxes = top_boxes[top_100]
        top_conf = top_conf[top_100]
        top_label = top_label[top_100]

        for i, c in list(enumerate(top_label)):
            predicted_class = self.class_names[int(c)]      # 类别ID 转为实际类别名
            box = top_boxes[i]
            score = str(top_conf[i])

            top, left, bottom, right = box
            if predicted_class not in class_names:
                continue

            f.write("%s %s %s %s %s %s\n" % (
            predicted_class, score[:6], str(int(left)), str(int(top)), str(int(right)), str(int(bottom))))

        f.close()
        return

    def get_map_txt_group(self, image_id_group, image_group, class_names, map_out_path):
        images = []
        imgs_ids = []
        for index,image_fuse in enumerate(image_group):
            # ---------------------------------------------------------#
            #   在这里将图像转换成RGB图像，防止灰度图在预测时报错。
            #   代码仅仅支持RGB图像的预测，所有其它类型的图像都会转化成RGB
            # ---------------------------------------------------------#
            image = image_fuse[..., 0:3]
            of_rgb = image_fuse[..., 3:]
            image = Image.fromarray(image, mode='RGB')
            of_rgb = Image.fromarray(of_rgb, mode='RGB')
            image = cvtColor(image)
            of_rgb = cvtColor(of_rgb)

            image_shape = [image.height, image.width]
            # ---------------------------------------------------------#
            #   给图像增加灰条，实现不失真的resize (letterbox_image=true)
            #   也可以直接resize进行识别
            # ---------------------------------------------------------#
            image = resize_image(image, (self.input_shape[1], self.input_shape[0]), self.letterbox_image)
            of_rgb = resize_image(of_rgb, (self.input_shape[1], self.input_shape[0]), self.letterbox_image)

            # ---------------------------------------------------------#
            #   添加上batch_size维度
            # ---------------------------------------------------------#
            image = preprocess_input(np.array(image, dtype=np.float32))  # preprocess_input: 用coco数据集进行标准化, /255 -mean /std
            of_rgb = preprocess_input_of(np.array(of_rgb, dtype=np.float32))  # preprocess_input: 用coco数据集进行标准化, /255 -mean /std

            if self.coordConv:
                x = np.linspace(-1, 1, image.shape[0])
                y = np.linspace(-1, 1, image.shape[1])
                X, Y = np.meshgrid(x, y)
                coords = np.zeros((image.shape[0], image.shape[1], 2))
                coords[..., 0] = X
                coords[..., 1] = Y
                image = np.concatenate((image, coords), axis=-1)

            if self.yolov:
                frame_id = int(image_id_group[index].split("_")[-1])
                imgs_ids.append(frame_id)
            # ---------------------------------------------------------#
            #   添加上batch_size维度
            # ---------------------------------------------------------#
            image = np.expand_dims(np.transpose(image, (2, 0, 1)), 0).astype(np.float32)   # (0, 3, h, w)
            of_rgb = np.expand_dims(np.transpose(of_rgb, (2, 0, 1)), 0).astype(np.float32)   # (0, 3, h, w)
            fuse_image = np.concatenate([image,of_rgb],1)

            images.append(fuse_image)

        image_data = np.concatenate(images,axis=0)    # (16, 3, h, w)
        # image_shape = np.array(np.shape(image_data)[2:4])

        with torch.no_grad():
            images = torch.from_numpy(image_data)
            if self.cuda:
                images = images.cuda()

            if self.yolov:
                imgs_ids = torch.from_numpy(np.array(imgs_ids))
                if self.cuda:
                    imgs_ids = imgs_ids.cuda()
                inputs = [images,imgs_ids]
            else:
                inputs = images
            # ---------------------------------------------------------#
            #   将图像输入网络当中进行预测！
            # ---------------------------------------------------------#
            outputs, VID_fc, pred_result, pred_idx = self.net(inputs)    # for yolox 三个特征层的预测头输出 list 3: [(16,7,80,80),(16,7,40,40),(16,7,20,20)]
                                                                         # for yolov: outputs, VID_fc, pred_result, pred_idx
                                                                         # 其中, outputs:  三个特征层的预测头输出 list 3: [(16,7,80,80),(16,7,40,40),(16,7,20,20)]
                                                                         # VID_fc: (bs,10,2)  每帧10个预测框，每个框的VID分类logits
                                                                         # pred_result: yolox预测头 先nms(0.75thres)，再保留概率最高的10个 list[16x(10,7)] 7:(x1, y1, x2, y2, obj_conf_yolox, class_conf_yolox, class_pred_yolox, each_class_conf_yolox) x1,y1,x2,y2为640尺寸下的绝对值
                                                                         # pred_idx: 相对于8400的索引，从中选择10个，list[16x(10,)] 用于后续VID的MSA
            results, results_original_yolox = self.fuse_VID_cls_pred(copy.deepcopy(pred_result), self.num_classes, VID_fc, self.input_shape, image_shape, self.letterbox_image, nms_thre=self.nms_iou, conf_thre=self.confidence)

            # outputs = decode_outputs(outputs, self.input_shape)  # 输入三个特征层 [(),(),()] 输出(16,8400,7) 7：(x,y,w,h,obj_prob,c0_prob,c1_prob) 其中x,y,w,h为640下的框坐标，但/640进行了归一化
            # ---------------------------------------------------------#
            #   NMS: 遍历每一张图片进行NMS，先从x,y,w,h-->x1,y1,x2,y2(归一化的)-->8400里的每一个，根据obj_prob*cls_prob进行初筛(conf_thres)-->NMS再筛(nms_thres)-->x,y,w,h(还是归一化的)-->x1,y1,x2,y2：原始图像上的绝对值(image_shape)
            #   result: list:16, 每个元素为(xxx,7)  7的内容为：原图尺寸下的x1, y1, x2, y2, obj_conf, class_conf, class_pred
            # ---------------------------------------------------------#
            # results = non_max_suppression(outputs, self.num_classes, self.input_shape, image_shape, self.letterbox_image, conf_thres=self.confidence, nms_thres=self.nms_iou)
        # results = results_original_yolox
        # 依次解析每一帧预测结果，并写入各自的txt
        for i, item in enumerate(results):
            if (i+1) % (self.group_num) == 0:   # 隔4帧统计预测结果
                f = open(os.path.join(map_out_path, "detection-results/" + image_id_group[i] + ".txt"), "w")

                if item is None:
                    f.close()
                    continue   # 跳过，不写入txt

                # top_label = np.array(results[i][:, 6], dtype='int32')  # 框类别ID
                top_label = 1 - np.array(results[i][:, 4] * results[i][:, 5] >= self.confidence, dtype='int32')  # 框类别ID
                top_conf = results[i][:, 4] * results[i][:, 5]         # 最终置信度为obj * cls
                top_boxes = results[i][:, :4]                          # 最终边界框为原始图像大小(非640)下的x1, y1, x2, y2 (列，行)

                top_100 = np.argsort(top_label)[::-1][:self.max_boxes]  # 从里面最多选100个
                top_boxes = top_boxes[top_100]
                top_conf = top_conf[top_100]
                top_label = top_label[top_100]

                for j, c in list(enumerate(top_label)):
                    predicted_class = self.class_names[int(c)]      # 类别ID 转为实际类别名
                    box = top_boxes[j]
                    score = str(top_conf[j])

                    top, left, bottom, right = box
                    if predicted_class not in class_names:
                        continue
                    f.write("%s %s %s %s %s %s\n" % (predicted_class, score[:6], str(int(left)), str(int(top)), str(int(right)), str(int(bottom))))

                f.close()
        return

    def on_epoch_end(self, epoch, model_eval):
        if epoch % self.period == 0 and self.eval_flag:    # 统计MAP
            self.net = model_eval
            if not os.path.exists(self.map_out_path):
                os.makedirs(self.map_out_path)
            if not os.path.exists(os.path.join(self.map_out_path, "ground-truth")):
                os.makedirs(os.path.join(self.map_out_path, "ground-truth"))
            if not os.path.exists(os.path.join(self.map_out_path, "detection-results")):
                os.makedirs(os.path.join(self.map_out_path, "detection-results"))
            print("Get map.")
            for annotation_line in tqdm(self.val_lines):
                line        = annotation_line.split()
                image_id    = os.path.basename(line[0]).split('.')[0]
                #------------------------------#
                #   读取图像并转换成RGB图像
                #------------------------------#
                fuse_image = np.load(line[0])
                image = fuse_image[..., :3]
                of_rgb = fuse_image[..., 3:]
                image = Image.fromarray(image, mode='RGB')
                of_rgb = Image.fromarray(of_rgb, mode='RGB')
                #------------------------------#
                #   获得预测框
                #------------------------------#
                gt_boxes    = np.array([np.array(list(map(int,box.split(',')))) for box in line[1:]])
                #------------------------------#
                #   获得预测txt
                #------------------------------#
                self.get_map_txt(image_id, image, of_rgb, self.class_names, self.map_out_path)
                
                #------------------------------#
                #   生成真实框txt
                #------------------------------#
                with open(os.path.join(self.map_out_path, "ground-truth/"+image_id+".txt"), "w") as new_f:
                    for box in gt_boxes:
                        left, top, right, bottom, obj = box
                        obj_name = self.class_names[obj]
                        new_f.write("%s %s %s %s %s\n" % (obj_name, left, top, right, bottom))
                        
            print("Calculate Map.")
            try:
                temp_map = get_coco_map(class_names = self.class_names, path = self.map_out_path)[1]
                # temp_map = get_map(self.MINOVERLAP, False, path=self.map_out_path, score_threhold=0.2)   # 只计算指定的AP(MINOVERLAP)
            except:
                temp_map = get_map(self.MINOVERLAP, False, path = self.map_out_path)
            self.maps.append(temp_map)
            self.epoches.append(epoch)
            print("MAP:", temp_map)

            with open(os.path.join(self.log_dir, "epoch_map.txt"), 'a') as f:
                f.write(str(temp_map))
                f.write("\n")
            
            plt.figure()
            plt.plot(self.epoches, self.maps, 'red', linewidth = 2, label='train map')

            plt.grid(True)
            plt.xlabel('Epoch')
            plt.ylabel('Map %s'%str(self.MINOVERLAP))
            plt.title('A Map Curve')
            plt.legend(loc="upper right")

            plt.savefig(os.path.join(self.log_dir, "epoch_map.png"))
            plt.cla()
            plt.close("all")

            print("Get map done.")
            shutil.rmtree(self.map_out_path)

    def on_epoch_end_yolov(self, epoch, model_eval, group_num):
        if epoch % self.period == 0 and self.eval_flag:  # 统计MAP
            self.net = model_eval
            if not os.path.exists(self.map_out_path):
                os.makedirs(self.map_out_path)
            if not os.path.exists(os.path.join(self.map_out_path, "ground-truth")):
                os.makedirs(os.path.join(self.map_out_path, "ground-truth"))
            if not os.path.exists(os.path.join(self.map_out_path, "detection-results")):
                os.makedirs(os.path.join(self.map_out_path, "detection-results"))
            print("Get map.")
            image_id_group = []
            image_group = []
            group_num = group_num
            for i, annotation_line in enumerate(tqdm(self.val_lines)):    # val.txt list
                line = annotation_line.split()
                image_id = os.path.basename(line[0]).split('.')[0]
                # ------------------------------#
                #   读取图像并转换成RGB图像
                # ------------------------------#
                image = np.load(line[0])

                # ------------------------------#
                #   获得标注预测框
                # ------------------------------#
                gt_boxes = np.array([np.array(list(map(int, box.split(',')))) for box in line[1:]])
                # ------------------------------#
                #   生成真实框txt
                # ------------------------------#
                if (i + 1) % group_num == 0:     # 隔4帧统计ground-truth
                    with open(os.path.join(self.map_out_path, "ground-truth/" + image_id + ".txt"), "w") as new_f:
                        for box in gt_boxes:
                            left, top, right, bottom, obj = box
                            obj_name = self.class_names[obj]
                            new_f.write("%s %s %s %s %s\n" % (obj_name, left, top, right, bottom))

                image_id_group.append(image_id)
                image_group.append(image)

                if (i + 1) % group_num == 0:   # i=15 image_id_group 16 i=30
                    # ------------------------------#
                    #   获得预测txt
                    # ------------------------------#
                    self.get_map_txt_group(image_id_group, image_group, self.class_names, self.map_out_path)
                    image_id_group = []
                    image_group = []

            print("Calculate Map.")
            try:
                temp_map = get_coco_map(class_names=self.class_names, path=self.map_out_path)[1]  # 标准coco接口
                # temp_map = get_map(self.MINOVERLAP, False, path=self.map_out_path, score_threhold=0.2)
            except:
                temp_map = get_map(self.MINOVERLAP, False, path=self.map_out_path, score_threhold=0.2)   # 只计算指定的AP(MINOVERLAP)
            self.maps.append(temp_map)
            self.epoches.append(epoch)
            print("MAP{}:{}".format(self.MINOVERLAP, temp_map))
            logger.info(temp_map)

            with open(os.path.join(self.log_dir, "epoch_map.txt"), 'a') as f:
                f.write(str(temp_map))
                f.write("\n")

            plt.figure()
            plt.plot(self.epoches, self.maps, 'red', linewidth=2, label='train map')

            plt.grid(True)
            plt.xlabel('Epoch')
            plt.ylabel('Map %s' % str(self.MINOVERLAP))
            plt.title('A Map Curve')
            plt.legend(loc="upper right")

            plt.savefig(os.path.join(self.log_dir, "epoch_map.png"))
            plt.cla()
            plt.close("all")

            print("Get map done.")
            shutil.rmtree(self.map_out_path)
