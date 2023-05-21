import colorsys
import os
import cv2
import numpy as np
import torch
import torch.nn as nn
from PIL import Image

from nets.yolo import YoloBody
from utils.utils import (cvtColor, get_classes, preprocess_input, resize_image, show_config)
from utils.utils_bbox import decode_outputs, non_max_suppression
import copy
import torchvision
from tqdm import tqdm

class YOLO_server(object):
    _defaults = {
        "model_path"  : None,
        "classes_path": None,
        "input_shape": [640, 640],
        "phi": 'nano',
        "confidence": 0.8,
        "nms_iou": 0.3,
        "letterbox_image": True,
        "cuda": True,
        "MultiInputs": False,
        "max_boxes": 100,
        "features" : "234",
        "PAFPN_use": False,
        "group_num": 4,
    }

    @classmethod
    def get_defaults(cls, n):
        if n in cls._defaults:
            return cls._defaults[n]
        else:
            return "Unrecognized attribute name '" + n + "'"

    # ---------------------------------------------------#
    #   初始化YOLO
    # ---------------------------------------------------#
    def __init__(self, **kwargs):
        self.__dict__.update(self._defaults)
        for name, value in kwargs.items():
            setattr(self, name, value)
            print(name, value)

        # ---------------------------------------------------#
        #   获得种类和先验框的数量
        # ---------------------------------------------------#
        self.class_names, self.num_classes = get_classes(self.classes_path)
        self.cons_num = 0

        # ---------------------------------------------------#
        #   画框设置不同的颜色
        # ---------------------------------------------------#
        hsv_tuples = [(x / self.num_classes, 1., 1.) for x in range(self.num_classes)]
        self.colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        self.colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), self.colors))
        self.generate()

        show_config(**self._defaults)

    # ---------------------------------------------------#
    #   生成模型
    # ---------------------------------------------------#
    def generate(self, onnx=False):
        self.net = YoloBody(self.num_classes, self.phi, self.MultiInputs, features=self.features, PAFPN_use=self.PAFPN_use, group_num=self.group_num)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.net.load_state_dict(torch.load(self.model_path, map_location=device))
        self.net = self.net.eval()
        print('{} model, and classes loaded.'.format(self.model_path))
        if not onnx:
            if self.cuda:
                self.net = nn.DataParallel(self.net)
                self.net = self.net.cuda()

    def convert_to_onnx(self, simplify, model_path):
        import onnx
        self.generate(onnx=True)

        im = torch.zeros(1, 3, *self.input_shape).to('cpu')  # image size(1, 3, 512, 512) BCHW
        input_layer_names = ["images"]
        output_layer_names = ["output"]

        # Export the model
        print(f'Starting export with onnx {onnx.__version__}.')
        torch.onnx.export(self.net,
                          im,
                          f=model_path,
                          verbose=False,
                          opset_version=12,
                          training=torch.onnx.TrainingMode.EVAL,
                          do_constant_folding=True,
                          input_names=input_layer_names,
                          output_names=output_layer_names,
                          dynamic_axes=None)

        # Checks
        model_onnx = onnx.load(model_path)  # load onnx model
        onnx.checker.check_model(model_onnx)  # check onnx model

        # Simplify onnx
        if simplify:
            import onnxsim
            print(f'Simplifying with onnx-simplifier {onnxsim.__version__}.')
            model_onnx, check = onnxsim.simplify(
                model_onnx,
                dynamic_input_shape=False)
            assert check, 'assert check failed'
            onnx.save(model_onnx, model_path)

        print('Onnx model save as {}'.format(model_path))

    def yolo_correct_boxes(self, box_xy, box_wh, input_shape, image_shape, letterbox_image):
        # -----------------------------------------------------------------#
        #   把y轴放前面是因为方便预测框和图像的宽高进行相乘
        # -----------------------------------------------------------------#
        box_yx = box_xy[..., ::-1]
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

    def fuse_VID_cls_pred(self, prediction, fc_outputs, input_shape, image_shape, letterbox_image, conf_thre=0.001, nms_thre=0.5):
        output = [None for _ in range(len(prediction))]
        output_ori = [None for _ in range(len(prediction))]
        prediction_ori = copy.deepcopy(prediction)
        obj_refined_conf = fc_outputs.squeeze(-1)

        for i, detections in enumerate(prediction):
            if not detections.size(0):
                continue

            # -----------------#
            #   归一化bbox
            # -----------------#
            detections[..., [0, 2]] = detections[..., [0, 2]] / input_shape[1]
            detections[..., [1, 3]] = detections[..., [1, 3]] / input_shape[0]

            prediction_ori[i][..., [0, 2]] = prediction_ori[i][..., [0, 2]] / input_shape[1]
            prediction_ori[i][..., [1, 3]] = prediction_ori[i][..., [1, 3]] / input_shape[0]

            # obj refine
            detections[:, 4] = obj_refined_conf[i].sigmoid()
            detections_high = detections[:,:]
            detections_ori = prediction_ori[i]
            conf_mask = (detections_high[:, 4] * detections_high[:,5] >= conf_thre).squeeze()
            detections_high = detections_high[conf_mask]

            if not detections_high.shape[0]:
                continue
            if len(detections_high.shape) == 3:
                detections_high = detections_high[0]

            nms_out_index = torchvision.ops.nms(
                detections_high[:, :4],
                detections_high[:, 4] * detections_high[:, 5],
                nms_thre,
            )

            detections_high = detections_high[nms_out_index]


            output[i] = detections_high

            detections_ori = detections_ori[:, :7]
            conf_mask = (detections_ori[:, 4] * detections_ori[:, 5]>= conf_thre).squeeze()
            detections_ori = detections_ori[conf_mask]

            nms_out_index = torchvision.ops.nms(
                detections_ori[:, :4],
                detections_ori[:, 4] * detections_ori[:, 5],
                nms_thre,
            )

            detections_ori = detections_ori[nms_out_index]
            output_ori[i] = detections_ori

        output_cpu = []
        output_ori_cpu = []
        for i in range(len(output)):
            if output[i] is None:
                output_cpu.append(output[i])
            else:
                output[i] = output[i].cpu().numpy()
                box_xy, box_wh = (output[i][:, 0:2] + output[i][:, 2:4]) / 2, output[i][:, 2:4] - output[i][:,0:2]
                output[i][:, :4] = self.yolo_correct_boxes(box_xy, box_wh, input_shape, image_shape, letterbox_image)
                output_cpu.append(output[i])
        for i in range(len(output_ori)):
            if output_ori[i] is None:
                output_ori_cpu.append(output_ori[i])
            else:
                output_ori[i] = output_ori[i].cpu().numpy()   # (xxx, 7)
                box_xy, box_wh = (output_ori[i][:, 0:2] + output_ori[i][:, 2:4]) / 2, output_ori[i][:, 2:4] - output_ori[i][:,0:2]
                output_ori[i][:, :4] = self.yolo_correct_boxes(box_xy, box_wh, input_shape, image_shape, letterbox_image)
                output_ori_cpu.append(output_ori[i])

        return output_cpu, output_ori_cpu


    def get_map_txt_group(self, image_id_group, image_group, class_names, map_out_path):
        images = []
        imgs_ids = []

        for index,image_fuse in enumerate(image_group):   # 200
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
            image_data = resize_image(image, (self.input_shape[1], self.input_shape[0]), self.letterbox_image)
            of_rgb = resize_image(of_rgb, (self.input_shape[1], self.input_shape[0]), self.letterbox_image)
            # ---------------------------------------------------------#
            #   添加上batch_size维度
            # ---------------------------------------------------------#
            image = preprocess_input(np.array(image_data, dtype=np.float32))
            of_rgb = preprocess_input(np.array(of_rgb, dtype=np.float32))

            if self.MultiInputs:
                frame_id = int(image_id_group[index].split("_")[-1])
                imgs_ids.append(frame_id)
            # ---------------------------------------------------------#
            #   添加上batch_size维度
            # ---------------------------------------------------------#
            image = np.expand_dims(np.transpose(image, (2, 0, 1)), 0).astype(np.float32)   # (0, 3, h, w)
            of_rgb = np.expand_dims(np.transpose(of_rgb, (2, 0, 1)), 0).astype(np.float32)   # (0, 3, h, w)
            fuse_image = np.concatenate([image,of_rgb],1)
            images.append(fuse_image)

        image_data = np.concatenate(images,axis=0)    # (group_num, 3, h, w)

        with torch.no_grad():
            images = torch.from_numpy(image_data)
            if self.cuda:
                images = images.cuda()

            if self.MultiInputs:
                imgs_ids = torch.from_numpy(np.array(imgs_ids))
                if self.cuda:
                    imgs_ids = imgs_ids.cuda()
                inputs = [images, imgs_ids]
            else:
                inputs = images
            # ---------------------------------------------------------#
            #   将图像输入网络当中进行预测！
            # ---------------------------------------------------------#
            outputs, VID_fc, pred_result, pred_idx = self.net(inputs)
            results, results_original_yolox = self.fuse_VID_cls_pred(copy.deepcopy(pred_result), self.num_classes, VID_fc, self.input_shape, image_shape, self.letterbox_image, nms_thre=self.nms_iou, conf_thre=self.confidence)

        results = results_original_yolox

        for i, item in enumerate(results):
            if (i + 1) % (self.group_num) == 0:
                f = open(os.path.join(map_out_path, "detection-results/" + image_id_group[i] + ".txt"), "w")

                if item is None:
                    f.close()
                    continue

                top_label = 1 - np.array(results[i][:, 4] * results[i][:, 5] >= self.confidence, dtype='int32')
                top_conf = results[i][:, 4] * results[i][:, 5]
                top_boxes = results[i][:, :4]

                top_100 = np.argsort(top_label)[::-1][:self.max_boxes]
                top_boxes = top_boxes[top_100]
                top_conf = top_conf[top_100]
                top_label = top_label[top_100]

                for j, c in list(enumerate(top_label)):
                    predicted_class = self.class_names[int(c)]
                    box = top_boxes[j]
                    score = str(top_conf[j])

                    top, left, bottom, right = box
                    if predicted_class not in class_names:
                        continue
                    f.write("%s %s %s %s %s %s\n" % (
                    predicted_class, score[:6], str(int(left)), str(int(top)), str(int(right)), str(int(bottom))))

                f.close()
        return

    def get_map_txt_group_batch2(self, image_id_group, image_group, class_names, map_out_path, group_num, test_gen):
        imgs_shape = []
        image_id_group_real = []
        results = []
        results_original_yolox = []
        for j in range(len(image_group)):
            imgs_shape.append([image_group[j].shape[0],image_group[j].shape[1]])

        num = 0
        with torch.no_grad():
            for iter, batch in enumerate(tqdm(test_gen)):
                images = batch[0]["images"]       # torch.tensor device=cuda, (bs, c, h, w)
                imgs_ids = batch[0]["frames_id"]  # torch.tensor device=cuda, (bs,)
                num += images.shape[0]
                for i in range(images.shape[0]):            # 遍历bs 0~20
                    if (i+1) % group_num == 0:              # 3  7  11  15
                        image_id_group_real.append(image_id_group[i+iter*images.shape[0]])
                        image_shape = imgs_shape[i+iter*images.shape[0]]
                        inputs = [images[i-group_num+1:i+1], imgs_ids[i-group_num+1:i+1]]   # 取4个
                        outputs, VID_fc, pred_result, pred_idx = self.net(inputs)

                        VID_fc = VID_fc[[-1]]
                        pred_result = [pred_result[-1]]
                        result, result_original_yolox = self.fuse_VID_cls_pred(copy.deepcopy(pred_result), VID_fc, self.input_shape, image_shape, self.letterbox_image, nms_thre=self.nms_iou, conf_thre=self.confidence)
                        results = results + result
                        results_original_yolox = results_original_yolox + result_original_yolox

        print("num:",num)
        print("len(results):",len(results))
        for i, item in enumerate(results):
            f = open(os.path.join(map_out_path, "detection-results/" + image_id_group_real[i] + ".txt"), "w")

            if item is None:
                f.close()
                continue
            top_label = 1 - np.array(results[i][:, 4] * results[i][:, 5] >= self.confidence, dtype='int32')
            top_conf = results[i][:, 4] * results[i][:, 5]
            top_boxes = results[i][:, :4]

            top_100 = np.argsort(top_conf)[::-1][:self.max_boxes]
            top_boxes = top_boxes[top_100]
            top_conf = top_conf[top_100]
            top_label = top_label[top_100]

            for j, c in list(enumerate(top_label)):
                predicted_class = self.class_names[int(c)]
                box = top_boxes[j]
                score = str(top_conf[j])

                top, left, bottom, right = box
                if predicted_class not in class_names:
                    continue
                f.write("%s %s %s %s %s %s\n" % (
                predicted_class, score[:6], str(int(left)), str(int(top)), str(int(right)), str(int(bottom))))

            f.close()
        return




