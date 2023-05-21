import time
from random import sample, shuffle
import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data.dataset import Dataset
from utils.utils import cvtColor, preprocess_input, preprocess_input_of
import os
from nvidia.dali.pipeline import pipeline_def
import nvidia.dali.types as types
import nvidia.dali.fn as fn
from nvidia.dali.plugin.pytorch import DALIGenericIterator, LastBatchPolicy
from tqdm import tqdm
import cupy as cp
import gc

def prior_resize(image, resize, bbox, pad_value):
    # image = Image.fromarray(img, mode='RGB')
    iw, ih = image.shape[1],image.shape[0]  # 600,800
    h, w = resize[0], resize[1]  # 640,640
    box = bbox

    scale = min(w / iw, h / ih)
    nw = int(iw * scale)  # 640
    nh = int(ih * scale)  # 640
    dx = (w - nw) // 2
    dy = (h - nh) // 2

    # ---------------------------------#
    #   将图像多余的部分加上灰条
    # ---------------------------------#
    # image = image.resize((nw, nh), Image.BILINEAR)   # BILINEAR  BICUBIC
    image = cv2.resize(image,(nw,nh),interpolation=cv2.INTER_LINEAR)
    # new_image = Image.new('RGB', (w, h), (pad_value, pad_value, pad_value))
    new_image = np.ones((resize[0],resize[1],3),dtype=np.uint8)*pad_value
    # new_image.paste(image, (dx, dy))
    new_image[dy:dy+image.shape[0],dx:dx+image.shape[1]] = image
    # image_data = np.array(new_image, np.float32)

    # del image,new_image
    # gc.collect()

    # ---------------------------------#
    #   对真实框进行调整
    # ---------------------------------#
    if box is not None:
        box[:, [0, 2]] = box[:, [0, 2]] * nw / iw + dx
        box[:, [1, 3]] = box[:, [1, 3]] * nh / ih + dy
        box[:, 0:2][box[:, 0:2] < 0] = 0
        box[:, 2][box[:, 2] > w] = w
        box[:, 3][box[:, 3] > h] = h
        box_w = box[:, 2] - box[:, 0]
        box_h = box[:, 3] - box[:, 1]
        box = box[np.logical_and(box_w > 1, box_h > 1)]  # discard invalid box

        box[:, 2:4] = box[:, 2:4] - box[:, 0:2]      # 0-1 (w, h)
        # box[:, 0:2] = box[:, 0:2] + box[:, 2:4] / 2  # 0-1 (xc,yc)
        box[:, 0:2] = box[:, 0:2]                    # 0-1 (x1,y1)

        return new_image, box / resize[0] # box: [0,1]尺寸下的[left,top,right,bottom,class_id] class_id=0/1
    else:
        return new_image, None

def prior_resize_PIL(image, resize, bbox, pad_value):
    image = Image.fromarray(image, mode='RGB')
    # iw, ih = image.shape[1],image.shape[0]  # 600,800
    iw, ih = image.size
    h, w = resize[0], resize[1]  # 640,640
    box = bbox

    scale = min(w / iw, h / ih)
    nw = int(iw * scale)  # 640
    nh = int(ih * scale)  # 640
    dx = (w - nw) // 2
    dy = (h - nh) // 2

    # ---------------------------------#
    #   将图像多余的部分加上灰条
    # ---------------------------------#
    image = image.resize((nw, nh), Image.BILINEAR)   # BILINEAR  BICUBIC
    # image = cv2.resize(image,(nw,nh),interpolation=cv2.INTER_LINEAR)
    new_image = Image.new('RGB', (w, h), (pad_value, pad_value, pad_value))
    # new_image = np.ones((resize[0],resize[1],3),dtype=np.uint8)*pad_value
    new_image.paste(image, (dx, dy))
    # new_image[dy:dy+image.shape[0],dx:dx+image.shape[1]] = image
    image_data = np.array(new_image, np.float32)

    # del image,new_image
    # gc.collect()

    # ---------------------------------#
    #   对真实框进行调整
    # ---------------------------------#
    if box is not None:
        box[:, [0, 2]] = box[:, [0, 2]] * nw / iw + dx
        box[:, [1, 3]] = box[:, [1, 3]] * nh / ih + dy
        box[:, 0:2][box[:, 0:2] < 0] = 0
        box[:, 2][box[:, 2] > w] = w
        box[:, 3][box[:, 3] > h] = h
        box_w = box[:, 2] - box[:, 0]
        box_h = box[:, 3] - box[:, 1]
        box = box[np.logical_and(box_w > 1, box_h > 1)]  # discard invalid box

        box[:, 2:4] = box[:, 2:4] - box[:, 0:2]      # (w, h)
        box[:, 0:2] = box[:, 0:2] #+ box[:, 2:4] / 2  # (xc,yc)

        return image_data, box/resize[0]  # box: [0,1]尺寸下的[left,top,right,bottom,class_id] class_id=0/1
    else:
        return image_data, None

def prior_resize_equal(img, resize, bbox, pad_value):
    return img, bbox

class ExternalInputGpuIterator(object):
    def __init__(self, batch_size, data_dir, resize):
        self.images_dir = data_dir
        self.batch_size = batch_size
        f = open(self.images_dir)
        lines = f.readlines()
        self.all_gray_data = []
        self.all_of_data = []
        self.all_bbox = []
        self.all_labels = []
        self.all_frame_indices = []
        print("-----loading data into Ram-----")

        for index in tqdm(range(len(lines))):  # 读取每一个数据到内存
            # s = time.time()
            line = lines[index].strip()
            npy_path = line.split(" ")[0]
            npy = np.load(npy_path).astype(np.uint8)      # uint8
            bbox = np.array(list(map(int,line.split(" ")[1].split(','))))[np.newaxis,:]   # (1,5) xyxy

            resized_gray, resized_box = prior_resize(npy[...,0:3], resize, bbox, 128)         # xywc
            resized_of, _ = prior_resize(npy[..., 3:], resize, None, 128)
            # resized_gray2 = np.zeros((1000,1000,10))
            tmp = lines[index].split(".")[0].split("_")[-1]
            if tmp[-1] == "c":
                tmp = tmp[:-1]
            frame_id = int(tmp)

            # print(lines[index],resized_box)
            self.all_gray_data.append(resized_gray)
            self.all_of_data.append(resized_of)
            self.all_bbox.append(resized_box[0, :-1][np.newaxis,...])                    # (1,4) (xywc)
            self.all_labels.append(resized_box[0, -1][np.newaxis,...][np.newaxis,...])   # (1,1)
            self.all_frame_indices.append(frame_id)

            # e = time.time()
            # print(e-s)
            # del resized_gray, resized_of, npy
            # gc.collect()
            # sleep(100)

    def __iter__(self):
        self.i = 0
        self.n = len(self.all_gray_data)
        return self

    def __next__(self):
        gray_batch = []
        of_batch = []
        label_batch = []
        box_batch = []
        frame_indices_batch = []
        for _ in range(self.batch_size):
            gray_batch.append(cp.asarray(self.all_gray_data[self.i]).astype(cp.uint8))
            of_batch.append(cp.asarray(self.all_of_data[self.i]).astype(cp.uint8))
            label_batch.append(cp.asarray(self.all_labels[self.i]).astype(cp.float32))
            box_batch.append(cp.asarray(self.all_bbox[self.i]).astype(cp.float32))
            frame_indices_batch.append(cp.asarray(self.all_frame_indices[self.i]))
            self.i = (self.i + 1) % self.n
        return (gray_batch, of_batch, label_batch, box_batch, frame_indices_batch)

class ExternalInputGpuIterator_NoRam(object):
    def __init__(self, batch_size, data_dir, resize):
        self.images_dir = data_dir
        self.batch_size = batch_size
        f = open(self.images_dir)
        self.lines = f.readlines()
        self.resize = resize

    def __iter__(self):
        self.i = 0
        self.n = len(self.lines)
        return self

    def __next__(self):
        gray_batch = []
        of_batch = []
        label_batch = []
        box_batch = []
        frame_indices_batch = []
        for _ in range(self.batch_size):
            line = self.lines[self.i].strip()
            npy_path = line.split(" ")[0]
            npy = np.load(npy_path).astype(np.uint8)      # uint8
            bbox = np.array(list(map(int,line.split(" ")[1].split(','))))[np.newaxis,:]   # (1,5) xyxy

            resized_gray, resized_box = prior_resize(npy[...,0:3], self.resize, bbox, 128)         # xywc
            resized_of, _ = prior_resize(npy[..., 3:], self.resize, None, 128)

            tmp = self.lines[self.i].split(".")[0].split("_")[-1]
            if tmp[-1] == "c":
                tmp = tmp[:-1]
            frame_id = int(tmp)


            gray_batch.append(cp.asarray(resized_gray).astype(cp.uint8))
            of_batch.append(cp.asarray(resized_of).astype(cp.uint8))
            label_batch.append(cp.asarray(resized_box[0, -1][np.newaxis,...][np.newaxis,...]).astype(cp.float32))
            box_batch.append(cp.asarray(resized_box[0, :-1][np.newaxis,...]).astype(cp.float32))
            frame_indices_batch.append(cp.asarray(frame_id))
            self.i = (self.i + 1) % self.n
        return (gray_batch, of_batch, label_batch, box_batch, frame_indices_batch)

class ExternalInputGpuIteratorCallback(object):
    def __init__(self, batch_size, data_dir, resize):
        self.images_dir = data_dir
        self.batch_size = batch_size
        f = open(self.images_dir)
        self.lines = f.readlines()
        self.batch_size = batch_size
        self.full_iterations = len(self.lines) // batch_size
        self.resize = resize

    def __call__(self, sample_info):
        sample_idx = sample_info.idx_in_epoch
        if sample_info.iteration >= self.full_iterations:
            # Indicate end of the epoch
            raise StopIteration()

        line = self.lines[sample_idx].strip()
        npy_path = line.split(" ")[0]
        npy = np.load(npy_path).astype(np.uint8)  # uint8
        bbox = np.array(list(map(int, line.split(" ")[1].split(','))))[np.newaxis, :]  # (1,5) xyxy

        resized_gray, resized_box = prior_resize(npy[..., 0:3], self.resize, bbox, 128)  # xywc
        resized_of, _ = prior_resize(npy[..., 3:], self.resize, None, 128)

        tmp = self.lines[sample_idx].split(".")[0].split("_")[-1]
        if tmp[-1] == "c":
            tmp = tmp[:-1]
        frame_id = int(tmp)

        gray = resized_gray.astype(np.uint8)
        of = resized_of.astype(np.uint8)
        label = resized_box[0, -1][np.newaxis,...][np.newaxis,...].astype(np.float32)
        box = resized_box[0, :-1][np.newaxis,...].astype(np.float32)
        frame_indices = np.array([frame_id]).astype(np.float32)
        return gray, of, label, box, frame_indices

external_input_callable_def = """
import cupy as cp
import numpy as np
import cv2
from tqdm import tqdm

def prior_resize(image, resize, bbox, pad_value):
    # image = Image.fromarray(img, mode='RGB')
    iw, ih = image.shape[1],image.shape[0]  # 600,800
    h, w = resize[0], resize[1]  # 640,640
    box = bbox

    scale = min(w / iw, h / ih)
    nw = int(iw * scale)  # 640
    nh = int(ih * scale)  # 640
    dx = (w - nw) // 2
    dy = (h - nh) // 2

    # ---------------------------------#
    #   将图像多余的部分加上灰条
    # ---------------------------------#
    # image = image.resize((nw, nh), Image.BILINEAR)   # BILINEAR  BICUBIC
    image = cv2.resize(image,(nw,nh),interpolation=cv2.INTER_LINEAR)
    # new_image = Image.new('RGB', (w, h), (pad_value, pad_value, pad_value))
    new_image = np.ones((resize[0],resize[1],3),dtype=np.uint8)*pad_value
    # new_image.paste(image, (dx, dy))
    new_image[dy:dy+image.shape[0],dx:dx+image.shape[1]] = image
    # image_data = np.array(new_image, np.float32)

    # del image,new_image
    # gc.collect()

    # ---------------------------------#
    #   对真实框进行调整
    # ---------------------------------#
    if box is not None:
        box[:, [0, 2]] = box[:, [0, 2]] * nw / iw + dx
        box[:, [1, 3]] = box[:, [1, 3]] * nh / ih + dy
        box[:, 0:2][box[:, 0:2] < 0] = 0
        box[:, 2][box[:, 2] > w] = w
        box[:, 3][box[:, 3] > h] = h
        box_w = box[:, 2] - box[:, 0]
        box_h = box[:, 3] - box[:, 1]
        box = box[np.logical_and(box_w > 1, box_h > 1)]  # discard invalid box

        box[:, 2:4] = box[:, 2:4] - box[:, 0:2]      # 0-1 (w, h)
        # box[:, 0:2] = box[:, 0:2] + box[:, 2:4] / 2  # 0-1 (xc,yc)
        box[:, 0:2] = box[:, 0:2]                    # 0-1 (x1,y1)

        return new_image, box / resize[0] # box: [0,1]尺寸下的[left,top,right,bottom,class_id] class_id=0/1
    else:
        return new_image, None

class ExternalInputGpuIteratorCallback(object):
    def __init__(self, batch_size, data_dir, resize):
        self.images_dir = data_dir   
        self.batch_size = batch_size
        f = open(self.images_dir)
        self.lines = f.readlines()
        self.batch_size = batch_size
        self.full_iterations = len(self.lines) // batch_size
        self.resize = resize

    def __call__(self, sample_info):
        sample_idx = sample_info.idx_in_epoch
        if sample_info.iteration >= self.full_iterations:
            # Indicate end of the epoch
            raise StopIteration()

        line = self.lines[sample_idx].strip()
        npy_path = line.split(" ")[0]
        npy = np.load(npy_path).astype(np.uint8)  # uint8
        bbox = np.array(list(map(int, line.split(" ")[1].split(','))))[np.newaxis, :]  # (1,5) xyxy

        resized_gray, resized_box = prior_resize(npy[..., 0:3], self.resize, bbox, 128)  # xywc
        resized_of, _ = prior_resize(npy[..., 3:], self.resize, None, 128)

        tmp = self.lines[sample_idx].split(".")[0].split("/")[-1]                
        if tmp[-1] == "c":
            tmp = tmp[:-1]
        frame_id = int(tmp)

        gray = resized_gray.astype(np.uint8)
        of = resized_of.astype(np.uint8)
        label = resized_box[0, -1][np.newaxis,...][np.newaxis,...].astype(np.float32)
        box = resized_box[0, :-1][np.newaxis,...].astype(np.float32)
        frame_indices = np.array([frame_id]).astype(np.float32)
        return gray, of, label, box, frame_indices
"""

with open("external_input_tmp_module.py", 'w') as f:
    f.write(external_input_callable_def)

import external_input_tmp_module

@pipeline_def()
def Npy_pipeline(eii_gpu, resize=(640,640), fp16=True):

    images_gray, images_of, labels, bboxes, frame_indices = fn.external_source(source=eii_gpu, num_outputs=5, device="gpu")

    #------------------------#
    # hue, brightness, contrast
    # ------------------------#
    saturation = fn.random.uniform(range=[0.9, 1.1])
    contrast = fn.random.uniform(range=[0.9, 1.1])
    brightness = fn.random.uniform(range=[0.9, 1.1])
    hue = fn.random.uniform(range=[-0.1, 0.1])
    images_gray = fn.hsv(images_gray, dtype=types.FLOAT, hue=hue, saturation=saturation, device="gpu")  # use float to avoid clipping and
    images_gray = fn.brightness_contrast(images_gray,
                                    # contrast_center = 80,  # input is in float, but in 0..255 range  256x0.3125  out = contrast_center + contrast * (in - contrast_center)
                                    dtype = types.UINT8,
                                    brightness = brightness,
                                    contrast = contrast,
                                    device="gpu")

    dtype = types.FLOAT16 if fp16 else types.FLOAT

    #------------------------#
    # flip, Normalize
    # ------------------------#
    # flip_coin = fn.random.coin_flip(probability=0.5)
    # bboxes = fn.bb_flip(bboxes, ltrb=False, horizontal=flip_coin, device="gpu")   # [0,1] 输入。注意如果采用xywh，xy须是左上角坐标，而不是框中心坐标
    images_gray = fn.crop_mirror_normalize(images_gray,
                                      #crop=(resize[0], resize[1]),
                                      mean=[0.3125 * 255, 0.3125 * 255, 0.3125 * 255],
                                      std=[0.2594 * 255, 0.2594 * 255, 0.2594 * 255],
                                      # mirror=flip_coin,
                                      dtype=dtype,
                                      output_layout="CHW",
                                      pad_output=False, device="gpu")

    images_of = fn.crop_mirror_normalize(images_of,
                                      #crop=(resize[0], resize[1]),
                                      mean=[0.3125 * 255, 0.3125 * 255, 0.3125 * 255],
                                      std=[0.2594 * 255, 0.2594 * 255, 0.2594 * 255],
                                      # mirror=flip_coin,
                                      dtype=dtype,
                                      output_layout="CHW",
                                      pad_output=False, device="gpu")

    # labels=labels.gpu()
    # bboxes=bboxes.gpu()

    fuse_imgs = fn.cat(images_gray, images_of, axis=0, device="gpu")  # (chw)
    bboxes = fn.cast(bboxes, dtype=dtype)                             # (1,4)
    labels = fn.cast(labels, dtype=dtype)                             # (1,1)
    bboxes = fn.cat(bboxes, labels, axis=1, device="gpu")             # (1,5)
    frame_indices = fn.cast(frame_indices, dtype=dtype)
    # Npy_pipeline.set_outputs(fuse_imgs, bboxes, labels)

    return fuse_imgs, bboxes, frame_indices

@pipeline_def()
def Npy_pipeline_parallel(eii_gpu, fp16=True):

    images_gray, images_of, labels, bboxes, frame_indices = fn.external_source(source=eii_gpu, num_outputs=5, batch=False, parallel=True, dtype=[types.UINT8, types.UINT8, types.FLOAT, types.FLOAT, types.FLOAT], device="cpu")
    # images_gray = fn.decoders.image(images_gray, device="mixed")
    # images_of = fn.decoders.image(images_of, device="mixed")
    #------------------------#
    # hue, brightness, contrast
    # ------------------------#
    saturation = fn.random.uniform(range=[0.9, 1.1])
    contrast = fn.random.uniform(range=[0.9, 1.1])
    brightness = fn.random.uniform(range=[0.9, 1.1])
    hue = fn.random.uniform(range=[-0.1, 0.1])
    images_gray = fn.hsv(images_gray, dtype=types.FLOAT, hue=hue, saturation=saturation, device="cpu")  # use float to avoid clipping and
    images_gray = fn.brightness_contrast(images_gray,
                                    # contrast_center = 80,  # input is in float, but in 0..255 range  256x0.3125  out = contrast_center + contrast * (in - contrast_center)
                                    dtype = types.UINT8,
                                    brightness = brightness,
                                    contrast = contrast,
                                    device="cpu")

    dtype = types.FLOAT16 if fp16 else types.FLOAT

    #------------------------#
    # flip, Normalize
    # ------------------------#
    # flip_coin = fn.random.coin_flip(probability=0.5)
    # bboxes = fn.bb_flip(bboxes, ltrb=False, horizontal=flip_coin, device="cpu")   # [0,1] 输入。注意如果采用xywh，xy须是左上角坐标，而不是框中心坐标
    images_gray = fn.crop_mirror_normalize(images_gray,
                                      #crop=(resize[0], resize[1]),
                                      mean=[0.3125 * 255, 0.3125 * 255, 0.3125 * 255],
                                      std=[0.2594 * 255, 0.2594 * 255, 0.2594 * 255],
                                      # mirror=flip_coin,
                                      dtype=dtype,
                                      output_layout="CHW",
                                      pad_output=False, device="cpu")

    images_of = fn.crop_mirror_normalize(images_of,
                                      #crop=(resize[0], resize[1]),
                                      # mean=[0.99 * 255, 0.99 * 255, 0.99 * 255],
                                      # std=[0.0169 * 255, 0.0169 * 255, 0.0169 * 255],
                                      mean=[0.3125 * 255, 0.3125 * 255, 0.3125 * 255],
                                      std=[0.2594 * 255, 0.2594 * 255, 0.2594 * 255],
                                      # mirror=flip_coin,
                                      dtype=dtype,
                                      output_layout="CHW",
                                      pad_output=False, device="cpu")

    fuse_imgs = fn.cat(images_gray, images_of, axis=0, device="cpu")  # (chw)

    bboxes = fn.cast(bboxes, dtype=dtype)                             # (1,4)
    labels = fn.cast(labels, dtype=dtype)                             # (1,1)
    bboxes = fn.cat(bboxes, labels, axis=1)                           # (1,5)
    frame_indices = fn.cast(frame_indices, dtype=dtype)

    fuse_imgs = fuse_imgs.gpu()
    bboxes = bboxes.gpu()
    frame_indices = frame_indices.gpu()
    # Npy_pipeline.set_outputs(fuse_imgs, bboxes, labels)

    return fuse_imgs, bboxes, frame_indices

def get_train_dali_loader(batch_size,resize,fp16,data_dir,num_threads,device_id,parallel=True):
    if parallel:
        eii_gpu = external_input_tmp_module.ExternalInputGpuIteratorCallback(batch_size=batch_size, data_dir=data_dir, resize=resize)
        train_pipe = Npy_pipeline_parallel(eii_gpu,fp16,
                                  batch_size=batch_size,
                                  num_threads=num_threads,
                                  device_id=device_id,
                                  py_num_workers=num_threads,
                                  py_start_method='spawn')
    else:
        eii_gpu = ExternalInputGpuIterator_NoRam(batch_size=batch_size, data_dir=data_dir, resize=resize)
        train_pipe = Npy_pipeline(eii_gpu,resize,fp16,
                                  batch_size=batch_size,
                                  num_threads=num_threads,
                                  device_id=device_id)

    train_loader = DALIGenericIterator(
        train_pipe,
        ["images", "boxes", "frames_id"],
        #reader_name="Reader",
        last_batch_policy=LastBatchPolicy.PARTIAL,   # https://docs.nvidia.com/deeplearning/dali/user-guide/docs/plugins/pytorch_plugin_api.html
        last_batch_padded = True)

    return train_loader

class YoloDataset(Dataset):
    def __init__(self, annotation_lines, input_shape, num_classes, epoch_length, mosaic, train, mosaic_ratio = 0.7, MultiInputs = False, print_img_names = False):
        super(YoloDataset, self).__init__()
        self.annotation_lines   = annotation_lines
        self.input_shape        = input_shape
        self.num_classes        = num_classes
        self.epoch_length       = epoch_length
        self.mosaic             = mosaic
        self.train              = train
        self.mosaic_ratio       = mosaic_ratio

        self.epoch_now          = -1
        self.length             = len(self.annotation_lines)

        self.MultiInputs = MultiInputs
        self.print_img_names = print_img_names

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        # cv2.setNumThreads(0)           # 关闭cv的多线程
        index = index % self.length
        #---------------------------------------------------#
        #   训练时进行数据的随机增强
        #   验证时不进行数据的随机增强
        #---------------------------------------------------#
        line    = self.annotation_lines[index].split()
        box_raw    = np.array([np.array(list(map(int,box.split(',')))) for box in line[1:]])
        #------------------------------#
        #   读取图像并转换成RGB图像
        #------------------------------#
        fuse_image   = np.load(line[0])
        image = fuse_image[...,0:3]
        of_rgb = fuse_image[...,3:]
        image   = cvtColor(image)
        image = Image.fromarray(image, mode='RGB')
        of_rgb   = cvtColor(of_rgb)
        of_rgb = Image.fromarray(of_rgb, mode='RGB')

        if self.mosaic:
            if self.rand() < 0.5 and self.epoch_now < self.epoch_length * self.mosaic_ratio:
                lines = sample(self.annotation_lines, 3)         # ！！！注意这里有个随机抽样！！！
                lines.append(self.annotation_lines[index])
                if not self.MultiInputs:  # 如果有yolov，则不shuffle
                    shuffle(lines)
                image, box  = self.get_random_data_with_Mosaic(lines, self.input_shape)   # 逐张增强图片
            else:
                image, box  = self.get_random_data(self.annotation_lines[index], self.input_shape, random = self.train)
        else:
            if self.train:
                random_flag = False
                if random_flag:
                    image, box,  new_ar_fix, scale_fix, dx_fix, dy_fix, flip_fix = self.get_random_data(image, box_raw, self.input_shape, random = True)     # 随机增强一张图片
                    of_rgb = self.get_random_data_fix(of_rgb, self.input_shape, new_ar_fix, scale_fix, dx_fix, dy_fix, flip_fix)   # 随机增强一张图片
                else:
                    image, box = self.get_random_data(image, box_raw, self.input_shape, random = False)     # 随机增强一张图片
                    of_rgb, _  = self.get_random_data(of_rgb, box_raw, self.input_shape, random = False)    # 随机增强一张图片
            else:  # 验证时无需增强
                image, box = self.get_random_data(image, box_raw, self.input_shape, random = False)
                of_rgb, _ = self.get_random_data(of_rgb, box_raw, self.input_shape, random = False)
        # 到这里，image的数据增强后的，uint8 array

        image = preprocess_input(np.array(image, dtype=np.float32))       # preprocess_input: 标准化, /255 -mean /std
        of_rgb = preprocess_input_of(np.array(of_rgb, dtype=np.float32))  # preprocess_input: 标准化, /255 -mean /std

        image       = np.transpose(image, (2, 0, 1))     # (3,h,w)
        of_rgb      = np.transpose(of_rgb, (2, 0, 1))
        image = np.concatenate([image,of_rgb], 0)        # (6,h,w)
        box         = np.array(box, dtype=np.float32)
        if len(box) != 0:
            box[:, 2:4] = box[:, 2:4] - box[:, 0:2]       # (w, h)
            box[:, 0:2] = box[:, 0:2] + box[:, 2:4] / 2   # (xc,yc)

        if self.print_img_names:
            tmp = self.annotation_lines[index].split(".")[0].split("/")[-1]
            frame_id = int(tmp)
            return image, box, frame_id
        else:
            return image, box             # (6,h,w)

    def rand(self, a=0, b=1):
        return np.random.rand()*(b-a) + a

    def get_random_data(self, image, box_raw, input_shape, jitter=.1, hue=.1, sat=0.1, val=0.1, random=True):
        # line    = annotation_line.split()
        # #------------------------------#
        # #   读取图像并转换成RGB图像
        # #------------------------------#
        # image   = Image.open(line[0])
        # image   = cvtColor(image)
        #------------------------------#
        #   获得图像的高宽与目标高宽
        #------------------------------#
        iw, ih  = image.size   # 600,800
        h, w    = input_shape  # 640,640
        #------------------------------#
        #   获得预测框
        #------------------------------#
        box     = box_raw

        if not random:
            scale = min(w/iw, h/ih)
            nw = int(iw*scale)   # 640
            nh = int(ih*scale)   # 640
            dx = (w-nw)//2
            dy = (h-nh)//2

            #---------------------------------#
            #   将图像多余的部分加上灰条
            #---------------------------------#
            image       = image.resize((nw,nh), Image.BICUBIC)
            new_image   = Image.new('RGB', (w,h), (128,128,128))
            new_image.paste(image, (dx, dy))
            image_data  = np.array(new_image, np.float32)

            #---------------------------------#
            #   对真实框进行调整
            #---------------------------------#
            if len(box)>0:
                np.random.shuffle(box)
                box[:, [0,2]] = box[:, [0,2]]*nw/iw + dx
                box[:, [1,3]] = box[:, [1,3]]*nh/ih + dy
                box[:, 0:2][box[:, 0:2]<0] = 0
                box[:, 2][box[:, 2]>w] = w
                box[:, 3][box[:, 3]>h] = h
                box_w = box[:, 2] - box[:, 0]
                box_h = box[:, 3] - box[:, 1]
                box = box[np.logical_and(box_w>1, box_h>1)] # discard invalid box

            return image_data, box        # box: 640尺寸下的[left,top,right,bottom,class_id] class_id=0/1
                
        #------------------------------------------#
        #   对图像进行缩放并且进行长和宽的扭曲
        #------------------------------------------#
        new_ar = iw/ih * self.rand(1-jitter,1+jitter) / self.rand(1-jitter,1+jitter)  # 扭曲长宽比
        scale = self.rand(.25, 2)
        new_ar_fix = new_ar
        scale_fix = scale

        if new_ar < 1:           # h>w
            nh = int(scale*h)
            nw = int(nh*new_ar)
        else:
            nw = int(scale*w)
            nh = int(nw/new_ar)
        image = image.resize((nw,nh), Image.BICUBIC)

        #------------------------------------------#
        #   将图像多余的部分加上灰条
        #------------------------------------------#
        dx = int(self.rand(0, w-nw))
        dy = int(self.rand(0, h-nh))
        dx_fix = dx
        dy_fix = dy
        new_image = Image.new('RGB', (w,h), (128,128,128))
        new_image.paste(image, (dx, dy))
        image = new_image

        #------------------------------------------#
        #   翻转图像
        #------------------------------------------#
        flip = self.rand()<.5
        flip_fix = flip
        if flip: image = image.transpose(Image.FLIP_LEFT_RIGHT)

        image_data      = np.array(image, np.uint8)
        #---------------------------------#
        #   对图像进行色域变换
        #   计算色域变换的参数
        #---------------------------------#
        r               = np.random.uniform(-1, 1, 3) * [hue, sat, val] + 1
        #---------------------------------#
        #   将图像转到HSV上
        #---------------------------------#
        hue, sat, val   = cv2.split(cv2.cvtColor(image_data, cv2.COLOR_RGB2HSV))
        dtype           = image_data.dtype
        #---------------------------------#
        #   应用变换
        #---------------------------------#
        x       = np.arange(0, 256, dtype=r.dtype)
        lut_hue = ((x * r[0]) % 180).astype(dtype)
        lut_sat = np.clip(x * r[1], 0, 255).astype(dtype)
        lut_val = np.clip(x * r[2], 0, 255).astype(dtype)

        image_data = cv2.merge((cv2.LUT(hue, lut_hue), cv2.LUT(sat, lut_sat), cv2.LUT(val, lut_val)))
        image_data = cv2.cvtColor(image_data, cv2.COLOR_HSV2RGB)

        #---------------------------------#
        #   对真实框进行调整
        #---------------------------------#
        if len(box)>0:
            np.random.shuffle(box)  # (x1,y1,x2,y2)
            box[:, [0,2]] = box[:, [0,2]]*nw/iw + dx
            box[:, [1,3]] = box[:, [1,3]]*nh/ih + dy
            if flip: box[:, [0,2]] = w - box[:, [2,0]]
            box[:, 0:2][box[:, 0:2]<0] = 0
            box[:, 2][box[:, 2]>w] = w
            box[:, 3][box[:, 3]>h] = h
            box_w = box[:, 2] - box[:, 0]
            box_h = box[:, 3] - box[:, 1]
            box = box[np.logical_and(box_w>1, box_h>1)] 
        
        return image_data, box, new_ar_fix, scale_fix, dx_fix, dy_fix, flip_fix

    def get_random_data_fix(self, image, input_shape, new_ar_fix, scale_fix, dx_fix, dy_fix, flip_fix):
        iw, ih = image.size  # 600,800
        h, w = input_shape  # 640,640

        # ------------------------------------------#
        #   对图像进行缩放并且进行长和宽的扭曲
        # ------------------------------------------#
        new_ar = new_ar_fix
        scale = scale_fix
        if new_ar < 1:
            nh = int(scale * h)
            nw = int(nh * new_ar)
        else:
            nw = int(scale * w)
            nh = int(nw / new_ar)
        image = image.resize((nw, nh), Image.BICUBIC)

        # ------------------------------------------#
        #   将图像多余的部分加上灰条
        # ------------------------------------------#
        dx = dx_fix
        dy = dy_fix
        new_image = Image.new('RGB', (w, h), (128, 128, 128))
        new_image.paste(image, (dx, dy))
        image = new_image
        # ------------------------------------------#
        #   翻转图像
        # ------------------------------------------#
        flip = flip_fix
        if flip: image = image.transpose(Image.FLIP_LEFT_RIGHT)

        image_data = np.array(image, np.uint8)

        return image_data

    def merge_bboxes(self, bboxes, cutx, cuty):
        merge_bbox = []
        for i in range(len(bboxes)):
            for box in bboxes[i]:
                tmp_box = []
                x1, y1, x2, y2 = box[0], box[1], box[2], box[3]

                if i == 0:
                    if y1 > cuty or x1 > cutx:
                        continue
                    if y2 >= cuty and y1 <= cuty:
                        y2 = cuty
                    if x2 >= cutx and x1 <= cutx:
                        x2 = cutx

                if i == 1:
                    if y2 < cuty or x1 > cutx:
                        continue
                    if y2 >= cuty and y1 <= cuty:
                        y1 = cuty
                    if x2 >= cutx and x1 <= cutx:
                        x2 = cutx

                if i == 2:
                    if y2 < cuty or x2 < cutx:
                        continue
                    if y2 >= cuty and y1 <= cuty:
                        y1 = cuty
                    if x2 >= cutx and x1 <= cutx:
                        x1 = cutx

                if i == 3:
                    if y1 > cuty or x2 < cutx:
                        continue
                    if y2 >= cuty and y1 <= cuty:
                        y2 = cuty
                    if x2 >= cutx and x1 <= cutx:
                        x1 = cutx
                tmp_box.append(x1)
                tmp_box.append(y1)
                tmp_box.append(x2)
                tmp_box.append(y2)
                tmp_box.append(box[-1])
                merge_bbox.append(tmp_box)
        return merge_bbox

    def get_random_data_with_Mosaic(self, annotation_line, input_shape, jitter=0.3, hue=.1, sat=0.7, val=0.4):
        h, w = input_shape
        min_offset_x = self.rand(0.3, 0.7)
        min_offset_y = self.rand(0.3, 0.7)

        image_datas = [] 
        box_datas   = []
        index       = 0
        for line in annotation_line:
            #---------------------------------#
            #   每一行进行分割
            #---------------------------------#
            line_content = line.split()
            #---------------------------------#
            #   打开图片
            #---------------------------------#
            image = Image.open(line_content[0])
            image = cvtColor(image)
            
            #---------------------------------#
            #   图片的大小
            #---------------------------------#
            iw, ih = image.size
            #---------------------------------#
            #   保存框的位置
            #---------------------------------#
            box = np.array([np.array(list(map(int,box.split(',')))) for box in line_content[1:]])
            
            #---------------------------------#
            #   是否翻转图片
            #---------------------------------#
            flip = self.rand()<.5
            if flip and len(box)>0:
                image = image.transpose(Image.FLIP_LEFT_RIGHT)
                box[:, [0,2]] = iw - box[:, [2,0]]

            #------------------------------------------#
            #   对图像进行缩放并且进行长和宽的扭曲
            #------------------------------------------#
            new_ar = iw/ih * self.rand(1-jitter,1+jitter) / self.rand(1-jitter,1+jitter)
            scale = self.rand(.4, 1)
            if new_ar < 1:
                nh = int(scale*h)
                nw = int(nh*new_ar)
            else:
                nw = int(scale*w)
                nh = int(nw/new_ar)
            image = image.resize((nw, nh), Image.BICUBIC)

            #-----------------------------------------------#
            #   将图片进行放置，分别对应四张分割图片的位置
            #-----------------------------------------------#
            if index == 0:
                dx = int(w*min_offset_x) - nw
                dy = int(h*min_offset_y) - nh
            elif index == 1:
                dx = int(w*min_offset_x) - nw
                dy = int(h*min_offset_y)
            elif index == 2:
                dx = int(w*min_offset_x)
                dy = int(h*min_offset_y)
            elif index == 3:
                dx = int(w*min_offset_x)
                dy = int(h*min_offset_y) - nh
            
            new_image = Image.new('RGB', (w,h), (128,128,128))
            new_image.paste(image, (dx, dy))
            image_data = np.array(new_image)

            index = index + 1
            box_data = []
            #---------------------------------#
            #   对box进行重新处理
            #---------------------------------#
            if len(box)>0:
                np.random.shuffle(box)
                box[:, [0,2]] = box[:, [0,2]]*nw/iw + dx
                box[:, [1,3]] = box[:, [1,3]]*nh/ih + dy
                box[:, 0:2][box[:, 0:2]<0] = 0
                box[:, 2][box[:, 2]>w] = w
                box[:, 3][box[:, 3]>h] = h
                box_w = box[:, 2] - box[:, 0]
                box_h = box[:, 3] - box[:, 1]
                box = box[np.logical_and(box_w>1, box_h>1)]
                box_data = np.zeros((len(box),5))
                box_data[:len(box)] = box
            
            image_datas.append(image_data)
            box_datas.append(box_data)

        #---------------------------------#
        #   将图片分割，放在一起
        #---------------------------------#
        cutx = int(w * min_offset_x)
        cuty = int(h * min_offset_y)

        new_image = np.zeros([h, w, 3])
        new_image[:cuty, :cutx, :] = image_datas[0][:cuty, :cutx, :]
        new_image[cuty:, :cutx, :] = image_datas[1][cuty:, :cutx, :]
        new_image[cuty:, cutx:, :] = image_datas[2][cuty:, cutx:, :]
        new_image[:cuty, cutx:, :] = image_datas[3][:cuty, cutx:, :]

        new_image       = np.array(new_image, np.uint8)
        #---------------------------------#
        #   对图像进行色域变换
        #   计算色域变换的参数
        #---------------------------------#
        r               = np.random.uniform(-1, 1, 3) * [hue, sat, val] + 1
        #---------------------------------#
        #   将图像转到HSV上
        #---------------------------------#
        hue, sat, val   = cv2.split(cv2.cvtColor(new_image, cv2.COLOR_RGB2HSV))
        dtype           = new_image.dtype
        #---------------------------------#
        #   应用变换
        #---------------------------------#
        x       = np.arange(0, 256, dtype=r.dtype)
        lut_hue = ((x * r[0]) % 180).astype(dtype)
        lut_sat = np.clip(x * r[1], 0, 255).astype(dtype)
        lut_val = np.clip(x * r[2], 0, 255).astype(dtype)

        new_image = cv2.merge((cv2.LUT(hue, lut_hue), cv2.LUT(sat, lut_sat), cv2.LUT(val, lut_val)))
        new_image = cv2.cvtColor(new_image, cv2.COLOR_HSV2RGB)

        #---------------------------------#
        #   对框进行进一步的处理
        #---------------------------------#
        new_boxes = self.merge_bboxes(box_datas, cutx, cuty)

        return new_image, new_boxes

# DataLoader中collate_fn使用
def yolo_dataset_collate(batch):
    images = []
    bboxes = []
    names = []
    for img, box, name in batch:
        images.append(img)
        bboxes.append(box)
        names.append(name)
    images = torch.from_numpy(np.array(images)).type(torch.FloatTensor)
    bboxes = [torch.from_numpy(ann).type(torch.FloatTensor) for ann in bboxes]
    names = torch.from_numpy(np.array(names)).type(torch.FloatTensor)

    return images, bboxes, names
