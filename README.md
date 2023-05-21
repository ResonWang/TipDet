# TipDet
## 1. data pipeline
### step1: run **generate_label_txt.py** to get label.txt
### step2: run **generate_key_OF_imgs.py** to get Optical flow RGB images of keyframes for each video
### step3: run **generate_sequence_inputs_label.py** to get the SequenceInput_label_groupShuffle.txt 
### Data of porcine insertion can be downloaded from [here](http://baidu.com) (password). Unzip and put them in the Dataset folder
## 2. training
### run train_sequence_inputs.py
## 3. test
### run test.py
## 4. References
### 1. https://github.com/Megvii-BaseDetection/YOLOX
### 2. https://github.com/bubbliiiing/yolox-pytorch
### 3. https://github.com/YuHengsss/YOLOV
### 4. https://github.com/tomrunia/OpticalFlow_Visualization
### 5. https://github.com/NVIDIA/DALI
### 6. https://github.com/opencv/opencv_contrib
### 7. https://github.com/lucidrains/vit-pytorch
