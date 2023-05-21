import os
from tqdm import tqdm
import cv2
import numpy as np
import random

f_w = open("../dataset/SequenceInput_label.txt",mode='w',newline='')

keyframe_len = 4

keyframes = {}

Base_Dir_OF = "../dataset/OF"
Base_Dir = "../dataset/ImgsLabels"
TS_Dir = "../dataset/Ts_Npy"              # two stream npy file dir
if not os.path.exists(TS_Dir):
    os.mkdir(TS_Dir)

imgs_dir = os.listdir(Base_Dir_OF)
for dir in tqdm(imgs_dir):
    print(dir)
    OF_dir = Base_Dir_OF + "/" + dir
    imgs = os.listdir(OF_dir)
    imgs_id = [int(x.strip(".jpg")) for x in imgs]
    imgs_id.sort(reverse=False)
    imgs = [str(x).zfill(5)+".jpg" for x in imgs_id]

    keyframes[dir[:-3]] = [imgs[0]]*(keyframe_len-1) + imgs

    if not os.path.exists(TS_Dir + "/" + dir[:-3]):
        os.mkdir(TS_Dir + "/" + dir[:-3])


f = open("../dataset/label.txt")
lines = f.readlines()
f.close()
for line in tqdm(lines):
    # print(line)
    img_path = line.split(" ")[0]
    img_name = img_path.split("/")[-1]
    v_id = img_path.split("/")[-2]
    frame_id = img_name.strip(".jpg")

    keyframes_this_video = keyframes[v_id]
    try:
        index = keyframes_this_video.index(img_name, keyframe_len-1)    # in case a label is at the beginning of a video
    except:
        continue

    keyframesNames_this_frame = keyframes_this_video[index-keyframe_len+1:index+1]    # select keyframe_len for each label

    for kf in keyframesNames_this_frame:
        TS_save_path = TS_Dir + "/" + v_id + "/" + kf.replace("jpg","npy")
        if os.path.exists(TS_save_path):
            continue
        raw_img_path = img_path.replace(img_name, kf)
        raw_img = cv2.imread(raw_img_path)
        OF_img_path = Base_Dir_OF + "/" + v_id + "_OF" + "/" + kf
        OF_img = cv2.imread(OF_img_path)
        TS = np.concatenate((raw_img, OF_img), axis=-1)

        np.save(TS_save_path, TS)
        f_w.write(line.replace("ImgsLabels","Ts_Npy").replace(img_name, kf.replace("jpg","npy")))

f_w.close()

# yolo to voc
f = open("../dataset/SequenceInput_label.txt")
f_w = open("../dataset/SequenceInput_label_voc.txt",mode='w',newline='')

lines = f.readlines()
f.close()

for line in lines:
    label = line.strip()
    npy_path = label.split(" ")[0]
    category = label.split(" ")[1]
    img_shape = np.load(npy_path).shape
    H, W = img_shape[0], img_shape[1]
    xc = float(label.split(" ")[2]) * W
    yc = float(label.split(" ")[3]) * H
    w = float(label.split(" ")[4]) * W
    h = float(label.split(" ")[5]) * H
    x1 = int(xc - w/2)
    y1 = int(yc - h/2)
    x2 = int(xc + w/2)
    y2 = int(yc + h/2)
    new_line = npy_path[3:] + " " + str(x1) + "," + str(y1) + "," + str(x2) + "," + str(y2) + "," + category + "\n"
    f_w.write(new_line)

f_w.close()

# group shuffle: shuffle the data with keyframe_len frames as the base unit
txt = "../dataset/SequenceInput_label_voc.txt"
f = open(txt)
lines = f.readlines()
f.close()
group = []
group_num = keyframe_len
total_shuffled = []
for line in lines:
    group.append(line)
    if len(group) == group_num:
        total_shuffled.append(group)
        group = []
random.shuffle(total_shuffled)

txt = "../dataset/SequenceInput_label_groupShuffle.txt"
f = open(txt,"w",newline='')
for item_group in total_shuffled:
    for item in item_group:
        f.write(item)
f.close()