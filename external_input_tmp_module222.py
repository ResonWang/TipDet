
import cupy as cp
import numpy as np
import cv2
from tqdm import tqdm

def prior_resize(image, resize, pad_value):
    # image = Image.fromarray(img, mode='RGB')
    iw, ih = image.shape[1],image.shape[0]  # 600,800
    h, w = resize[0], resize[1]  # 640,640

    scale = min(w / iw, h / ih)
    nw = int(iw * scale)  # 640
    nh = int(ih * scale)  # 640
    dx = (w - nw) // 2
    dy = (h - nh) // 2

    image = cv2.resize(image,(nw,nh),interpolation=cv2.INTER_LINEAR)
    new_image = np.ones((resize[0],resize[1],3),dtype=np.uint8)*pad_value
    new_image[dy:dy+image.shape[0],dx:dx+image.shape[1]] = image

    return new_image

class ExternalInputGpuIteratorCallback(object):
    def __init__(self, batch_size, data_dir, resize):
        self.images_dir = data_dir             
        self.batch_size = batch_size
        self.lines = data_dir       # imgs
        self.batch_size = batch_size
        self.full_iterations = len(self.lines) // batch_size
        self.resize = resize

    def __call__(self, sample_info):
        sample_idx = sample_info.idx_in_epoch
        if sample_info.iteration >= self.full_iterations:
            # Indicate end of the epoch
            raise StopIteration()

        line = self.lines[sample_idx]
        npy_path = line
        npy = np.load(npy_path).astype(np.uint8)  # uint8

        resized_gray = prior_resize(npy[..., 0:3], self.resize, 128)  # xywc
        resized_of = prior_resize(npy[..., 3:], self.resize, 128)
        tmp = self.lines[sample_idx].split(".")[0].split("/")[-1]
        if tmp[-1] == "c":
            tmp = tmp[:-1]
        frame_id = int(tmp)

        gray = resized_gray.astype(np.uint8)
        of = resized_of.astype(np.uint8)
        frame_indices = np.array([frame_id]).astype(np.float32)

        return gray, of, frame_indices
