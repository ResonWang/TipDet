import os

Base_Dir = "../dataset/ImgsLabels"
imgs_dir = os.listdir(Base_Dir)

for img_dir in imgs_dir:
    imgs_or_labels_path = Base_Dir + "/" + img_dir
    imgs_or_labels = os.listdir(imgs_or_labels_path)
    for file in imgs_or_labels:
        if (file.endswith("txt") and not file.startswith("classes")):
            f = open(imgs_or_labels_path + '/' + file)
            label = f.read()
            f.close()
            if len(label)==0:
                os.remove(imgs_or_labels_path + '/' + file)
                print("{} is removed".format(imgs_or_labels_path + '/' + file))
