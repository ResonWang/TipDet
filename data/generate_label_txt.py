import os

Base_Dir = "../dataset/ImgsLabels"
imgs_dir = os.listdir(Base_Dir)
f_w = open("../dataset/label.txt",mode='w',newline='')

for img_dir in imgs_dir:
    imgs_or_labels_path = Base_Dir + "/" + img_dir
    imgs_or_labels = os.listdir(imgs_or_labels_path)
    for file in imgs_or_labels:
        if (file.endswith("txt") and not file.startswith("classes")):
            f = open(imgs_or_labels_path + '/' + file)
            label = f.read()
            label = imgs_or_labels_path + '/' + file.replace("txt", "jpg") + " 0" + label[2:]
            f.close()
            f_w.write(label)
f_w.close()