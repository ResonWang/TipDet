import numpy as np
from PIL import Image
from loguru import logger

#---------------------------------------------------------#
#   将图像转换成RGB图像，防止灰度图在预测时报错。
#   代码仅仅支持RGB图像的预测，所有其它类型的图像都会转化成RGB
#---------------------------------------------------------#
def cvtColor(image):
    if len(np.shape(image)) == 3 and np.shape(image)[2] == 3:
        return image 
    else:
        image = image.convert('RGB')
        return image 

#---------------------------------------------------#
#   对输入图像进行resize
#---------------------------------------------------#
def resize_image(image, size, letterbox_image):
    iw, ih  = image.size
    w, h    = size
    if letterbox_image:
        scale   = min(w/iw, h/ih)
        nw      = int(iw*scale)
        nh      = int(ih*scale)

        image   = image.resize((nw,nh), Image.BICUBIC)
        new_image = Image.new('RGB', size, (128,128,128))
        new_image.paste(image, ((w-nw)//2, (h-nh)//2))
    else:
        new_image = image.resize((w, h), Image.BICUBIC)
    return new_image

#---------------------------------------------------#
#   获得类
#---------------------------------------------------#
def get_classes(classes_path):
    with open(classes_path, encoding='utf-8') as f:
        class_names = f.readlines()
    class_names = [c.strip() for c in class_names]
    return class_names, len(class_names)

def preprocess_input(image):
    image /= 255.0
    # image -= np.array([0.485, 0.456, 0.406])   # coco的
    # image /= np.array([0.229, 0.224, 0.225])

    image -= np.array([0.3125, 0.3125, 0.3125])   # 穿刺的  0.3125032233411558 0.2593676581699875
    image /= np.array([0.2594, 0.2594, 0.2594])

    return image

def preprocess_input_of(image):
    image /= 255.0
    # image -= np.array([0.485, 0.456, 0.406])   # coco的
    # image /= np.array([0.229, 0.224, 0.225])

    # mean = [0.99, 0.99, 0.99],
    # std = [0.0169, 0.0169, 0.0169],

    mean = [0.3125, 0.3125, 0.3125],
    std = [0.2594, 0.2594, 0.2594],

    image -= np.array(mean)   # 穿刺的  0.3125032233411558 0.2593676581699875
    image /= np.array(std)

    return image

#---------------------------------------------------#
#   获得学习率
#---------------------------------------------------#
def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

def show_config(**kwargs):
    print('Default Configurations:')
    print('-' * 70)
    print('|%25s | %40s|' % ('keys', 'values'))
    print('-' * 70)
    for key, value in kwargs.items():
        # print('|%25s | %40s|' % (str(key), str(value)))
        logger.info("args: {}:{}".format(str(key), str(value)))
    print('-' * 70)
    