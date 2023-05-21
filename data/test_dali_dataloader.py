from nvidia.dali.pipeline import pipeline_def
import nvidia.dali.types as types
import nvidia.dali.fn as fn
from nvidia.dali.plugin.pytorch import DALIGenericIterator, LastBatchPolicy

external_input_callable_def = """
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
"""

with open("external_input_tmp_module222.py", 'w') as f:
    f.write(external_input_callable_def)

import external_input_tmp_module222

@pipeline_def()
def Npy_pipeline_parallel(eii_gpu, fp16=True):

    images_gray, images_of, frame_indices = fn.external_source(source=eii_gpu, num_outputs=3, batch=False, parallel=True, dtype=[types.UINT8, types.UINT8, types.FLOAT], device="cpu")
    # images_gray = fn.decoders.image(images_gray, device="mixed")
    # images_of = fn.decoders.image(images_of, device="mixed")

    dtype = types.FLOAT16 if fp16 else types.FLOAT

    images_gray = fn.crop_mirror_normalize(images_gray,
                                      mean=[0.3125 * 255, 0.3125 * 255, 0.3125 * 255],
                                      std=[0.2594 * 255, 0.2594 * 255, 0.2594 * 255],
                                      dtype=dtype,
                                      output_layout="CHW",
                                      pad_output=False, device="cpu")

    images_of = fn.crop_mirror_normalize(images_of,
                                      mean=[0.3125 * 255, 0.3125 * 255, 0.3125 * 255],
                                      std=[0.2594 * 255, 0.2594 * 255, 0.2594 * 255],
                                      dtype=dtype,
                                      output_layout="CHW",
                                      pad_output=False, device="cpu")

    fuse_imgs = fn.cat(images_gray, images_of, axis=0, device="cpu")  # (chw)

    fuse_imgs = fuse_imgs.gpu()

    frame_indices = frame_indices.gpu()

    return fuse_imgs,frame_indices

def get_test_dali_loader(batch_size,resize,fp16,data_dir,num_threads,device_id):
    eii_gpu = external_input_tmp_module222.ExternalInputGpuIteratorCallback(batch_size=batch_size, data_dir=data_dir, resize=resize)
    test_pipe = Npy_pipeline_parallel(eii_gpu,fp16,
                              batch_size=batch_size,
                              num_threads=num_threads,
                              device_id=device_id,
                              py_num_workers=num_threads,
                              py_start_method='spawn')

    test_loader = DALIGenericIterator(
        test_pipe,
        ["images","frames_id"],
        #reader_name="Reader",
        last_batch_policy = LastBatchPolicy.PARTIAL,   # 最后所剩不足一个batch的话，就用剩下的那些。 https://docs.nvidia.com/deeplearning/dali/user-guide/docs/plugins/pytorch_plugin_api.html
        last_batch_padded = True)

    return test_loader