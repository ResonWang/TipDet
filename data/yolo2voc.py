import cv2
import numpy as np

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