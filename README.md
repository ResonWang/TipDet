# TipDet
TipDet's performance in the process of thyroid insertion  
https://github.com/ResonWang/TipDet/assets/67107572/9ffe73e6-1e69-413f-80cc-040146325ea8
## 1. data pipeline
+ step1: run ___generate_label_txt.py___ to get label.txt 
+ step2: run ___generate_key_OF_imgs.py___ to get Optical flow RGB images of keyframes for each video
+ step3: run ___generate_sequence_inputs_label.py___ to get the SequenceInput_label_groupShuffle.txt     
Data of porcine insertion can be downloaded from [here](https://pan.baidu.com/s/1UCgN-KRxefVNQ0TqX1LGWA) (x3n5). Unzip and put them in the Dataset folder  
Data of human puncture data will be released after the paper is accepted. 
## 2. training
run train_sequence_inputs.py
## 3. test
run test.py
## 4. References
1. https://github.com/Megvii-BaseDetection/YOLOX
2. https://github.com/bubbliiiing/yolox-pytorch
3. https://github.com/YuHengsss/YOLOV
4. https://github.com/tomrunia/OpticalFlow_Visualization
5. https://github.com/NVIDIA/DALI
6. https://github.com/opencv/opencv_contrib
7. https://github.com/lucidrains/vit-pytorch

