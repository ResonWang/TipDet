import shutil
import warnings
warnings.filterwarnings("ignore")
import os
from data.test_dali_dataloader import get_test_dali_loader
from tqdm import tqdm
import numpy as np
from utils.utils import get_classes
from utils.utils_map import get_coco_map
from nets.model import YOLO_server
from loguru import logger

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)

if __name__ == "__main__":
    map_mode        = 0
    MultiInputs     = True
    dali_batch      = 1      # nvidia Dali
    group_num       = 4
    classes_path    = 'dataset/classes.txt'
    confidence      = 0.001
    nms_iou         = 0.5
    data_path       = 'dataset/ImgsLabels'
    map_out_path    = 'exp/test/exp1'
    test_label      = 'dataset/SequenceInput_label_groupShuffle.txt'

    f = open(test_label)
    labels = f.readlines()
    f.close()
    image_ids = [x.strip().split(" ")[0].split("/")[-2] + "_" + x.strip().split(" ")[0].split("/")[-1].split(".")[0] for x in labels]

    if os.path.exists(map_out_path):
        shutil.rmtree(map_out_path)

    if not os.path.exists(map_out_path):
        os.makedirs(map_out_path)
    if not os.path.exists(os.path.join(map_out_path, 'ground-truth')):
        os.makedirs(os.path.join(map_out_path, 'ground-truth'))
    if not os.path.exists(os.path.join(map_out_path, 'detection-results')):
        os.makedirs(os.path.join(map_out_path, 'detection-results'))
    if not os.path.exists(os.path.join(map_out_path, 'images-optional')):
        os.makedirs(os.path.join(map_out_path, 'images-optional'))

    class_names, _ = get_classes(classes_path)

    if map_mode == 0 or map_mode == 1:
        print("Load model.")
        yolo = YOLO_server(confidence = confidence, nms_iou = nms_iou, phi = 'nano',\
                           model_path = "weights/TipDet_YOLOX_nano.pth",\
                           classes_path = classes_path, input_shape = [640, 640], cuda = True, MultiInputs = MultiInputs,\
                           features = "34", PAFPN_use=False, group_num=group_num, max_boxes = 100)
        print("Load model done.")
        print("Get predict result.")

        image_id_group = []
        image_group = []
        group_num = group_num

        if dali_batch:
            test_bs = 20
            full_imgs_name = []

            logger.info("loading data")
            for i, image_path in enumerate(tqdm(labels)):
                image_path = image_path.strip().split(" ")[0]
                image_id = image_ids[i]
                image       = np.load(image_path)
                image_id_group.append(image_id)
                image_group.append(image)
                full_imgs_name.append(image_path)

            print("len(full_imgs_name):", len(full_imgs_name))
            test_gen = get_test_dali_loader(batch_size=test_bs, resize=(640, 640), fp16=False, data_dir=full_imgs_name, num_threads=4, device_id=0)
            print("len(test_gen):", len(test_gen))
            logger.info("start test")
            yolo.get_map_txt_group_batch2(image_id_group, image_group, yolo.class_names, map_out_path, group_num, test_gen)
            logger.info("finish test")
        else:
            for i, image_path in enumerate(tqdm(labels)):
                image_path = image_path.strip().split(" ")[0]
                image_id = image_ids[i]
                image = np.load(image_path)
                image_id_group.append(image_id)
                image_group.append(image)
                if (i + 1) % group_num == 0:  #
                    # ------------------------------#
                    #   按组预测txt
                    # ------------------------------#
                    yolo.get_map_txt_group(image_id_group, image_group, yolo.class_names, map_out_path)
                    image_id_group = []
                    image_group = []

        print("Get predict result done.")

    if map_mode == 0 or map_mode == 2:
        print("Get ground truth result.")
        for i, image_id in tqdm(enumerate(image_ids)):
            if (i + 1) % group_num == 0:
                with open(os.path.join(map_out_path, "ground-truth/"+image_id+".txt"), "w") as new_f:
                    label = labels[i]
                    left = label.strip().split(" ")[1].split(",")[0]
                    top = label.strip().split(" ")[1].split(",")[1]
                    right = label.strip().split(" ")[1].split(",")[2]
                    bottom = label.strip().split(" ")[1].split(",")[3]
                    new_f.write("%s %s %s %s %s\n" % ("tip", left, top, right, bottom))
        print("Get ground truth result done.")

    if map_mode == 0 or map_mode == 3:
        print("Get map.")
        map = get_coco_map(class_names=class_names, path=map_out_path)[1]
        print("Get map done.")
        print(map)

    if map_mode == 4:
        print("Get map.")
        get_coco_map(class_names = class_names, path = map_out_path)
        print("Get map done.")
