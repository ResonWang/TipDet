import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models


norm_mean = [0.485, 0.456, 0.406]
norm_std = [0.229, 0.224, 0.225]

test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(norm_mean, norm_std),
])

def denseRGB_filter(img_rgb):
    """
    根据相对运动原理，将光流BGR图转为hsv,然后对s通道进行减均值过滤，因为s通道表示了运动幅度的大小，减去均值则保留了相对比较大的运动
    :param img_bgr:
    :return:
    """
    img_hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
    img_s = img_hsv[..., 1].copy()
    img_s_norm = img_s - 2.5*np.mean(img_s)
    img_s_norm[img_s_norm<0] = 0
    img_hsv[..., 1] = img_s_norm
    rgb_norm = cv2.cvtColor(img_hsv,cv2.COLOR_HSV2RGB)
    return rgb_norm

def get_keyframe_model(path_pretrained_model):
    resnet18_ft = models.resnet18()
    num_ftrs = resnet18_ft.fc.in_features
    resnet18_ft.fc = nn.Linear(num_ftrs, 2)
    state_dict_load = torch.load(path_pretrained_model)
    resnet18_ft.load_state_dict(state_dict_load)
    resnet18_ft.eval().cuda()
    return resnet18_ft

def ifKeyframe(rgb_f, cls_model, of_filter=True):
    if of_filter:
        rgb_f = denseRGB_filter(rgb_f)

    rgb_f224 = cv2.resize(rgb_f, (224, 224), interpolation=cv2.INTER_LINEAR)
    rgb_f_tensor = test_transform(rgb_f224).unsqueeze(0).to("cuda")  # (c,h,w)
    logits = torch.softmax(cls_model(rgb_f_tensor),dim=1)  # (1,2)
    # _, keyframe = torch.max(logits.data, 1)
    keyframe = logits[0,1]
    return keyframe