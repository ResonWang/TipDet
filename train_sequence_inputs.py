"""
To train the TipDet-YOLOX-Nano from scratch, one should do it step by step:
step 1: train the naive YOLOX-Nano with public pretrained weight
step 2: construct and train Two-stream YOLOX-Nano with the weight from step 1 to initialize the network
step 3: construct the TipDet-YOLOX-Nano (multiple frames for the detection of the current frame) and train it with the the weight from step 2
"""

import datetime
import os
import numpy as np
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from logger import setup_logger
from nets.yolo import YoloBody
from nets.yolo_training import (ModelEMA, YOLOLoss, get_lr_scheduler,
                                set_optimizer_lr, weights_init)
from utils.callbacks import LossHistory, EvalCallback
from utils.dataloader import YoloDataset, yolo_dataset_collate, get_train_dali_loader
from utils.utils import get_classes, show_config
from utils.utils_fit import fit_one_epoch
from loguru import logger

import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
gpu_id = 0
os.environ['CUDA_VISIBLE_DEVICES'] = "1"

if __name__ == "__main__":

    Cuda            = True
    distributed     = False
    sync_bn         = False
    fp16            = True
    Dali            = True
    classes_path    = 'dataset/classes.txt'
    model_path        = "weights/TipDet_YOLOX_nano.pth"
    input_shape     = [640, 640]
    phi             = 'nano'                # yolox version
    mosaic              = False
    MultiInputs = True                      # if use multiple keyframes to detect
    group_num = 4                           # keyframe num for one frame
    print_batch_imgs_names = True           # must open
    #------------------------------------------------------#
    #  focal loss.
    #------------------------------------------------------#
    focal_loss = False
    alpha = 0.9
    gamma = 5
    #------------------------------------------------------#
    #  Backbone output locations
    #------------------------------------------------------#
    Features = "34"   # "234" "345", "23"
    PAFPN_use = False
    #----------------------------------------------------------------------------------------------------------------------------#
    #  if freeze head
    #------------------------------------------------------------------#
    Init_Epoch          = 0
    Freeze_Epoch        = 30
    Freeze_batch_size   = 256
    if MultiInputs:
        batch_size_val = group_num
    else:
        batch_size_val = 32

    UnFreeze_Epoch      = 300
    Unfreeze_batch_size = 32
    Freeze_Train        = True
    #------------------------------------------------------------------#
    #   Init_lr         maximum lr
    #   Min_lr          minimum lr
    #------------------------------------------------------------------#
    Init_lr             = 1e-4
    Min_lr              = Init_lr * 0.01
    optimizer_type      = "adam"
    momentum            = 0.9
    weight_decay        = 1e-3
    lr_decay_type       = "cos"
    #------------------------------------------------------------------#
    #   save and log
    #------------------------------------------------------------------#
    save_period         = 1
    save_dir            = 'exp/train/exp1'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    setup_logger(save_dir, filename="train_log.txt", mode="a")
    now = datetime.datetime.now()
    logger.info("--------starting training--------")
    logger.info(now)
    logger.info("--------------start--------------")
    #------------------------------------------------------------------#
    #  evaluate
    #------------------------------------------------------------------#
    eval_flag           = False
    eval_period         = 5
    #------------------------------------------------------------------#
    num_workers         = 4     # = gpu x 4
    #----------------------------------------------------#
    #  npy label file
    #----------------------------------------------------#
    train_annotation_path   = "dataset/SequenceInput_label_groupShuffle.txt"
    val_annotation_path     = "dataset/SequenceInput_label_groupShuffle.txt"
    #------------------------------------------------------#
    #  gpu settings
    #------------------------------------------------------#
    ngpus_per_node  = torch.cuda.device_count()
    if distributed:
        dist.init_process_group(backend="nccl")
        local_rank  = int(os.environ["LOCAL_RANK"])
        rank        = int(os.environ["RANK"])
        device      = torch.device("cuda", local_rank)
        if local_rank == 0:
            print(f"[{os.getpid()}] (rank = {rank}, local_rank = {local_rank}) training...")
            print("Gpu Device Count : ", ngpus_per_node)
    else:
        device          = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        local_rank      = 0
        rank            = 0
    #----------------------------------------------------#
    #  classes
    #----------------------------------------------------#
    class_names, num_classes = get_classes(classes_path)
    print("class_names, num_classes",class_names, num_classes)
    #------------------------------------------------------#
    #   freeze bn ?
    #------------------------------------------------------#
    def slow_bn(M):
        for m in M.modules():
            if isinstance(m, nn.BatchNorm2d):
                # m.eps = 1e-5
                m.momentum = 0
    #------------------------------------------------------#
    #   construct model
    #------------------------------------------------------#
    model = YoloBody(num_classes, phi, MultiInputs=MultiInputs, group_num=group_num, features = Features, PAFPN_use=PAFPN_use)
    #------------------------------------------------------#
    #   initialize weight
    #------------------------------------------------------#
    weights_init(model)
    if model_path != '':
        if local_rank == 0:
            print('Load weights {}.'.format(model_path))

        model_dict      = model.state_dict()
        pretrained_dict = torch.load(model_path, map_location = device)
        load_key, no_load_key, temp_dict = [], [], {}
        for k, v in pretrained_dict.items():
            if k in model_dict.keys() and np.shape(model_dict[k]) == np.shape(v):
                temp_dict[k] = v
                load_key.append(k)
            else:
                no_load_key.append(k)

        model_dict.update(temp_dict)
        model.load_state_dict(model_dict)

        if local_rank == 0:
            print("\nSuccessful Load Key:", str(load_key)[:500], "……\nSuccessful Load Key Num:", len(load_key))
            print("\nFail To Load Key:", str(no_load_key)[:500], "……\nFail To Load Key num:", len(no_load_key))

    #----------------------#
    #   loss
    #----------------------#
    yolo_loss    = YOLOLoss(num_classes, fp16, features=Features, MultiInputs=MultiInputs, focal_loss=focal_loss, alpha=alpha, gamma=gamma, group_num = group_num, Dali=Dali)   # 选择focal loss还是BCEloss
    #----------------------#
    #   log
    #----------------------#
    if local_rank == 0:
        time_str        = datetime.datetime.strftime(datetime.datetime.now(),'%Y_%m_%d_%H_%M_%S')
        log_dir         = os.path.join(save_dir, "loss_" + str(time_str))
        loss_history    = LossHistory(log_dir, model, input_shape=input_shape)   # inference will happen here
    else:
        loss_history    = None
    #----------------------#
    #   fp16
    #----------------------#
    if fp16:
        from torch.cuda.amp import GradScaler as GradScaler
        scaler = GradScaler()
    else:
        scaler = None
    #----------------------#
    model_train     = model.train()
    #----------------------#
    #   sync
    #----------------------#
    if sync_bn and ngpus_per_node > 1 and distributed:
        model_train = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model_train)
    elif sync_bn:
        print("Sync_bn is not support in one gpu or not distributed.")
    #----------------------#
    #   cuda
    #----------------------#
    if Cuda:
        if distributed:
            model_train = model_train.cuda(local_rank)
            model_train = torch.nn.parallel.DistributedDataParallel(model_train, device_ids=[local_rank], find_unused_parameters=True)
        else:
            model_train = torch.nn.DataParallel(model)
            cudnn.benchmark = True
            model_train = model_train.cuda()


    model.apply(slow_bn)
    # ema = ModelEMA(model_train)
    ema = False
    
    #---------------------------#
    #   dataset txt
    #---------------------------#
    with open(train_annotation_path, encoding='utf-8') as f:
        train_lines = f.readlines()
    with open(val_annotation_path, encoding='utf-8') as f:
        val_lines   = f.readlines()
    num_train   = len(train_lines)
    num_val     = len(val_lines)
     
    if local_rank == 0:
        show_config(
            train_and_val_size = num_train+num_val, train_size = num_train,\
            classes_path = classes_path, model_path = model_path, input_shape = input_shape, \
            Init_Epoch = Init_Epoch, Freeze_Epoch = Freeze_Epoch, UnFreeze_Epoch = UnFreeze_Epoch, Freeze_batch_size = Freeze_batch_size, \
            Unfreeze_batch_size = Unfreeze_batch_size, Freeze_Train = Freeze_Train, \
            Init_lr = Init_lr, Min_lr = Min_lr, optimizer_type = optimizer_type, momentum = momentum, weight_decay = weight_decay,\
            lr_decay_type = lr_decay_type, \
            save_period = save_period, save_dir = save_dir, num_workers = num_workers, num_train = num_train, num_val = num_val, \
            mosaic = mosaic, FP16 = fp16, batch_size_val=batch_size_val,  MultiInputs= MultiInputs, group_num = group_num, \
            focal_loss = focal_loss, features = Features, PAFPN_use = PAFPN_use, \
            train_annotation_path = train_annotation_path, val_annotation_path = val_annotation_path, Dali = Dali, \
        )

        wanted_step = 5e4 if optimizer_type == "sgd" else 1.5e4
        total_step  = num_train // Unfreeze_batch_size * UnFreeze_Epoch
        if total_step <= wanted_step:
            wanted_epoch = wanted_step // (num_train // Unfreeze_batch_size) + 1

    if True:
        UnFreeze_flag = False

        if Freeze_Train:
            freeze_modules = [model.backbone.parameters(),
                              model.head.cls_convs.parameters(),
                              model.head.stems.parameters(),
                              model.head.reg_convs.parameters(),
                              model.head.cls_preds.parameters(),
                              model.head.cls_convs2.parameters(),
                              model.head.reg_preds.parameters(),
                              model.head.obj_preds.parameters(),
                              model.head.gray_128to64.parameters(),
                              ]

            for moudle in freeze_modules:
                for param in moudle:                   # model.backbone.backbone.parameters():
                    param.requires_grad = False

        batch_size = Freeze_batch_size if Freeze_Train else Unfreeze_batch_size
        nbs             = 64
        lr_limit_max    = 1e-3 if optimizer_type == 'adam' else 5e-2  # 0.05
        lr_limit_min    = 3e-4 if optimizer_type == 'adam' else 5e-4  # 0.0005
        Init_lr_fit     = min(max(batch_size / nbs * Init_lr, lr_limit_min), lr_limit_max)  # 0.02
        Min_lr_fit      = min(max(batch_size / nbs * Min_lr, lr_limit_min * 1e-2), lr_limit_max * 1e-2) # 0.0002

        pg0, pg1, pg2 = [], [], []  
        for k, v in model.named_modules():
            if hasattr(v, "bias") and isinstance(v.bias, nn.Parameter):
                pg2.append(v.bias)    
            if isinstance(v, nn.BatchNorm2d) or "bn" in k:
                pg0.append(v.weight)    
            elif hasattr(v, "weight") and isinstance(v.weight, nn.Parameter):
                pg1.append(v.weight)   
        optimizer = {
            'adam'  : optim.Adam(pg0, Init_lr_fit, betas = (momentum, 0.999)),
            'sgd'   : optim.SGD(pg0, Init_lr_fit, momentum = momentum, nesterov=True)
        }[optimizer_type]
        optimizer.add_param_group({"params": pg1, "weight_decay": weight_decay})
        optimizer.add_param_group({"params": pg2})

        lr_scheduler_func = get_lr_scheduler(lr_decay_type, Init_lr_fit, Min_lr_fit, UnFreeze_Epoch)
        #
        epoch_step      = num_train // batch_size
        epoch_step_val  = num_val // batch_size_val
        
        if epoch_step == 0 or epoch_step_val == 0:
            raise ValueError("Dataset is too small, enlarge it!")
        
        if ema:
            ema.updates     = epoch_step * Init_Epoch

        if Dali:
            project_dir = "./"
            data_dir = "dataset/SequenceInput_label_groupShuffle.txt"
            # fp16: let the input to be fp16
            train_gen  = get_train_dali_loader(batch_size=batch_size, resize=input_shape, fp16=False, data_dir=project_dir+data_dir, num_threads=num_workers, device_id=gpu_id)
        else:
            train_dataset   = YoloDataset(train_lines, input_shape, num_classes, epoch_length = UnFreeze_Epoch, mosaic=mosaic, train = True, MultiInputs=MultiInputs, print_img_names=print_batch_imgs_names)

        val_dataset     = YoloDataset(val_lines, input_shape, num_classes, epoch_length = UnFreeze_Epoch, mosaic=False, train = False, MultiInputs=MultiInputs, print_img_names=print_batch_imgs_names)
        
        if distributed:
            train_sampler   = torch.utils.data.distributed.DistributedSampler(train_dataset, shuffle=True,)
            val_sampler     = torch.utils.data.distributed.DistributedSampler(val_dataset, shuffle=False,)
            batch_size      = batch_size // ngpus_per_node
            shuffle         = False
        else:
            train_sampler   = None
            val_sampler     = None
            if MultiInputs:
                shuffle     = False
            else:
                shuffle     = True

        if Dali:
            gen = train_gen
        else:
            gen             = DataLoader(train_dataset, shuffle = shuffle, batch_size = batch_size, num_workers = num_workers, pin_memory=True,
                                    drop_last=True, collate_fn=yolo_dataset_collate, sampler=train_sampler)   # 每迭代一次，生成一个batch的数据
        gen_val         = DataLoader(val_dataset, shuffle = shuffle, batch_size = batch_size_val, num_workers = num_workers, pin_memory=True,
                                    drop_last=True, collate_fn=yolo_dataset_collate, sampler=val_sampler)


        if local_rank == 0:
            eval_callback   = EvalCallback(model, input_shape, class_names, num_classes, val_lines, log_dir, Cuda, \
                                            eval_flag=eval_flag, period=eval_period, MINOVERLAP=0.2, MultiInputs = MultiInputs, group_num=group_num)
        else:
            eval_callback   = None
        
        #---------------------------------------#
        #   start training
        #---------------------------------------#
        for epoch in range(Init_Epoch, UnFreeze_Epoch):

            if epoch >= Freeze_Epoch and not UnFreeze_flag and Freeze_Train:
                batch_size = Unfreeze_batch_size
                nbs             = 64
                lr_limit_max    = 1e-3 if optimizer_type == 'adam' else 5e-2
                lr_limit_min    = 3e-4 if optimizer_type == 'adam' else 5e-4
                Init_lr_fit     = min(max(batch_size / nbs * Init_lr, lr_limit_min), lr_limit_max)
                Min_lr_fit      = min(max(batch_size / nbs * Min_lr, lr_limit_min * 1e-2), lr_limit_max * 1e-2)

                lr_scheduler_func = get_lr_scheduler(lr_decay_type, Init_lr_fit, Min_lr_fit, UnFreeze_Epoch)
                
                for param in model.backbone.parameters():
                    param.requires_grad = True

                epoch_step      = num_train // batch_size
                epoch_step_val  = num_val // batch_size_val

                if epoch_step == 0 or epoch_step_val == 0:
                    raise ValueError("Dataset is too small, enlarge it!")

                if distributed:
                    batch_size = batch_size // ngpus_per_node
                    
                if ema:
                    ema.updates     = epoch_step * epoch

                if Dali:
                    gen = train_gen
                else:
                    gen             = DataLoader(train_dataset, shuffle = shuffle, batch_size = batch_size, num_workers = num_workers, pin_memory=True,
                                                drop_last=True, collate_fn=yolo_dataset_collate, sampler=train_sampler)

                gen_val         = DataLoader(val_dataset, shuffle = shuffle, batch_size = batch_size_val, num_workers = num_workers, pin_memory=True,
                                            drop_last=True, collate_fn=yolo_dataset_collate, sampler=val_sampler)

                UnFreeze_flag = True

            if not Dali:
                gen.dataset.epoch_now       = epoch
                gen_val.dataset.epoch_now   = epoch

            if distributed:
                train_sampler.set_epoch(epoch)

            set_optimizer_lr(optimizer, lr_scheduler_func, epoch)
            # train, evaluate
            fit_one_epoch(model_train, model, ema, yolo_loss, loss_history, eval_callback, optimizer, epoch, epoch_step, epoch_step_val, gen, gen_val, UnFreeze_Epoch, Cuda, fp16, scaler, save_period, save_dir, MultiInputs, group_num, print_batch_imgs_names, Dali, local_rank)
                        
            if distributed:
                dist.barrier()

        if local_rank == 0:
            loss_history.writer.close()
