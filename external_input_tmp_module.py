
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
        self.images_dir = data_dir   # "/AI/videoDetection/algorithm/optical_flow/yolox-pytorch-main-TwoStreamYolov-singleClass-cls/2007_train_clinic_and_animal_singleClass_TwoStream.txt"
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
