import os
from tqdm import tqdm
import cv2
from tools.Dense_optical_flow import flow_to_image_cp, make_color_wheel, flow_to_image
from tools.keyframe import ifKeyframe, get_keyframe_model

keyframe_threshold = 0.5

colorwheel = make_color_wheel()

Base_Dir = "../dataset/ImgsLabels"
Base_Dir_OF = "../dataset/OF"
imgs_dir = os.listdir(Base_Dir)

keyframe_model = "../weights/Res18_keyframe.pth"
keyframe_model = get_keyframe_model(keyframe_model)

f = open("../dataset/label.txt")
lines = f.readlines()
f.close()
label_paths = []
for line in lines:
    img_path = line.split(" ")[0]
    label_paths.append(img_path)

OF_interval = 3

# f = open("../dataset/label.txt")
# labels = f.readlines()
for dir in tqdm(imgs_dir):
    OF_dir = Base_Dir_OF + "/" + dir + "_OF"
    if not os.path.exists(OF_dir):
        os.mkdir(OF_dir)
    imgs = os.listdir(Base_Dir + "/" + dir)
    # computer OF rgb image of each image
    for img in tqdm(imgs):
        if img.endswith("jpg"):
            current_img_path = Base_Dir + "/" + dir + "/" + img
            frame_id = img.strip(".jpg")
            if int(frame_id) <= OF_interval:
                continue

            cur_frame = cv2.imread(current_img_path, cv2.IMREAD_GRAYSCALE)

            pre_frame_id = str(int(frame_id) - OF_interval).zfill(5)
            pre_frame_path = current_img_path.replace(frame_id, pre_frame_id)
            pre_frame = cv2.imread(pre_frame_path, cv2.IMREAD_GRAYSCALE)

            gpu_frame_old = cv2.cuda_GpuMat()
            gpu_frame_old.upload(pre_frame)

            gpu_frame_next = cv2.cuda_GpuMat()
            gpu_frame_next.upload(cur_frame)
            optical_flow = cv2.cuda_FarnebackOpticalFlow.create(pyrScale=0.5, numLevels=3, fastPyramids = False, winSize=9, numIters=3, polyN=5, polySigma=1.2, flags=0)
            gpu_flow = optical_flow.calc(gpu_frame_old, gpu_frame_next, None)
            flow = gpu_flow.download()
            flow_rgb = flow_to_image_cp(flow, colorwheel)

            # judge if belongs to a keyframe
            keyframe = ifKeyframe(flow_rgb, keyframe_model)

            if keyframe > keyframe_threshold:
                cv2.imwrite(OF_dir+'/'+img, flow_rgb)

            for path in label_paths:
                if current_img_path == path:
                    cv2.imwrite(OF_dir + '/' + img, flow_rgb)

