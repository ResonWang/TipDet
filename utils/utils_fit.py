import os
from loguru import logger
import torch
from tqdm import tqdm

from utils.utils import get_lr


def fit_one_epoch(model_train, model, ema, yolo_loss, loss_history, eval_callback, optimizer, epoch, epoch_step, epoch_step_val, gen, gen_val, Epoch, cuda, fp16, scaler, save_period, save_dir, MultiInputs, group_num, print_batch_imgs_names, Dali, local_rank=0):
    loss        = 0
    val_loss    = 0
    iou_loss = 0
    obj_loss = 0
    cls_loss = 0
    val_iou_loss = 0
    val_obj_loss = 0
    val_cls_loss = 0
    if MultiInputs:
        cls_ref_loss = 0
        val_cls_ref_loss = 0

    if local_rank == 0:
        # print('Start Train')
        logger.info('Start Train')
        pbar = tqdm(total=epoch_step,desc=f'Epoch {epoch + 1}/{Epoch}',postfix=dict,mininterval=0.3,ncols=200)

    model_train.train()
    for iteration, batch in enumerate(gen):
        if iteration >= epoch_step:
            break

        if not Dali:
            images, targets = batch[0], batch[1]      # targets: xywc 640下的值

            if print_batch_imgs_names:
                # print("-----------------------imgs-----------------------------")
                images_ids = batch[2]
                # print(images_ids)
                # print("--------------------------------------------------------")

            with torch.no_grad():
                if cuda:
                    images  = images.cuda(local_rank)
                    targets = [ann.cuda(local_rank) for ann in targets]
                    images_ids = images_ids.cuda(local_rank)
        else:
            images = batch[0]["images"]    # torch.tensor device=cuda, (bs, c, h, w)
            targets = batch[0]["boxes"]    # torch.tensor device=cuda, (bs, 1, 5) bbox + cls label bbox: [0,1], xywh  abs=bbox*640
            # targets = [target for target in targets]  #
            if print_batch_imgs_names:
                # print("-----------------------imgs-----------------------------")
                images_ids = batch[0]["frames_id"]   # torch.tensor device=cuda, (bs,)
        #----------------------#
        #   清零梯度
        #----------------------#
        optimizer.zero_grad()
        if not fp16:
            #----------------------#
            #   前向传播
            #----------------------#
            if MultiInputs:
                outputs, VID_FC, pred_result, pred_idx = model_train([images,images_ids])   # (160,)
                outputs = [outputs, VID_FC, pred_result, pred_idx]
            else:
                outputs         = model_train(images)
            # print("outputs:",outputs)
            # print("targets",targets)
            #----------------------#
            #   计算损失
            #----------------------#
            loss_value = yolo_loss(outputs, targets)['total_loss']
            #----------------------#
            #   反向传播
            #----------------------#
            loss_value.backward()
            optimizer.step()
        else:
            from torch.cuda.amp import autocast
            with autocast():
                if MultiInputs:
                    outputs, VID_FC, pred_result, pred_idx = model_train([images,images_ids])
                    outputs = [outputs, VID_FC, pred_result, pred_idx]
                else:
                    outputs = model_train(images)
                #----------------------#
                #   计算损失
                #----------------------#
                # loss_all = yolo_loss(outputs, targets, iteration=iteration)
                loss_all = yolo_loss(outputs, targets)   # list 160x(1,5)
                loss_value = loss_all['total_loss']
            #----------------------#
            #   反向传播
            #----------------------#
            scaler.scale(loss_value).backward()
            scaler.step(optimizer)
            scaler.update()
        if ema:
            ema.update(model_train)

        # print(model_train.module.backbone.backbone.stem.conv.bn.running_mean)
        # print(model_train.module.backbone.backbone.stem.conv.bn.weight)
        # print(model.head.reg_preds[0].weight[0, :, 0, 0])

        loss += loss_value.item()
        iou_loss += loss_all['iou_loss'].item()
        obj_loss += loss_all['obj_loss'].item()
        cls_loss += loss_all['cls_loss'].item()
        if MultiInputs:
            cls_ref_loss += loss_all['cls_ref_loss'].item()

        if local_rank == 0:
            if MultiInputs:
                pbar.set_postfix(**{'total_loss'  : loss / (iteration + 1),
                                    'iou_loss': iou_loss / (iteration + 1),
                                    'obj_loss': obj_loss / (iteration + 1),
                                    'cls_loss': cls_loss / (iteration + 1),
                                    'cls_ref_loss': cls_ref_loss / (iteration + 1),
                                    'lr'    : get_lr(optimizer)})
            else:
                pbar.set_postfix(**{'total_loss'  : loss / (iteration + 1),
                                    'iou_loss': iou_loss / (iteration + 1),
                                    'obj_loss': obj_loss / (iteration + 1),
                                    'cls_loss': cls_loss / (iteration + 1),
                                    'lr'    : get_lr(optimizer)})
            pbar.update(1)
    if hasattr(torch.cuda, 'empty_cache'):  # 释放缓存分配器当前所管理的所有未使用的缓存
        torch.cuda.empty_cache()

    if local_rank == 0:
        pbar.close()
        # print('Finish Train')
        # print('Start Validation')
        logger.info('Finish Train')
        logger.info('Start Validation')
        pbar = tqdm(total=epoch_step_val, desc=f'Epoch {epoch + 1}/{Epoch}',postfix=dict,mininterval=0.3,ncols=200)

    if ema:
        model_train_eval = ema.ema
    else:
        model_train_eval = model_train.eval()

    # 验证
    for iteration, batch in enumerate(gen_val):
        if iteration >= epoch_step_val:
            break
        images, targets = batch[0], batch[1]

        if print_batch_imgs_names:
            # print("-----------------------imgs-----------------------------")
            images_ids = batch[2]
            # print(images_ids)
            # print("--------------------------------------------------------")

        with torch.no_grad():
            if cuda:
                images  = images.cuda(local_rank)
                targets = [ann.cuda(local_rank) for ann in targets]
                images_ids = images_ids.cuda(local_rank)
            #----------------------#
            #   清零梯度
            #----------------------#
            optimizer.zero_grad()
            #----------------------#
            #   前向传播
            #----------------------#
            if MultiInputs:
                outputs, VID_FC, pred_result, pred_idx = model_train_eval([images,images_ids])
                outputs = [outputs, VID_FC, pred_result, pred_idx]
            else:
                outputs         = model_train_eval(images)

            #----------------------#
            #   计算损失
            #----------------------#
            loss_all = yolo_loss(outputs, targets)
            loss_value = loss_all['total_loss']

        val_loss += loss_value.item()
        val_iou_loss += loss_all['iou_loss'].item()
        val_obj_loss += loss_all['obj_loss'].item()
        val_cls_loss += loss_all['cls_loss'].item()
        if MultiInputs:
            val_cls_ref_loss += loss_all['cls_ref_loss'].item()
        if local_rank == 0:
            if MultiInputs:
                pbar.set_postfix(**{'val_loss'  : val_loss / (iteration + 1),
                                    'val_iou_loss': val_iou_loss / (iteration + 1),
                                    'val_obj_loss': val_obj_loss / (iteration + 1),
                                    'val_cls_loss': val_cls_loss / (iteration + 1),
                                    'val_cls_ref_loss': val_cls_ref_loss / (iteration + 1)})
            else:
                pbar.set_postfix(**{'val_loss'  : val_loss / (iteration + 1),
                                    'val_iou_loss': val_iou_loss / (iteration + 1),
                                    'val_obj_loss': val_obj_loss / (iteration + 1),
                                    'val_cls_ref_loss': val_cls_loss / (iteration + 1)})
            pbar.update(1)


    if local_rank == 0:
        pbar.close()
        # print('Finish Validation')
        logger.info('Finish Validation')
        loss_history.append_loss(epoch + 1, loss / epoch_step, val_loss / epoch_step_val)
        if MultiInputs:
            eval_callback.on_epoch_end_yolov(epoch + 1, model_train_eval, group_num=group_num)
        else:
            eval_callback.on_epoch_end(epoch + 1, model_train_eval)         # 统计MAP
        # print('Epoch:'+ str(epoch + 1) + '/' + str(Epoch))
        # print('Total Loss: %.3f || Val Loss: %.3f ' % (loss / epoch_step, val_loss / epoch_step_val))
        logger.info('Epoch:'+ str(epoch + 1) + '/' + str(Epoch))
        logger.info('Total Loss: %.3f || Val Loss: %.3f ' % (loss / epoch_step, val_loss / epoch_step_val))
        logger.info('Epoch{} train_loss:{},iou_loss:{},obj_loss:{},cls_loss:{},cls_ref_loss:{},lr:{}'.format(epoch + 1,
                                                                                             loss / epoch_step,
                                                                                             iou_loss / epoch_step,
                                                                                             obj_loss / epoch_step,
                                                                                             cls_loss / epoch_step,
                                                                                             cls_ref_loss / epoch_step,
                                                                                             get_lr(optimizer)))
        logger.info('Epoch{} val_loss:{},iou_loss:{},obj_loss:{},cls_loss:{},cls_ref_loss:{}'.format(epoch + 1,
                                                                                     val_loss / epoch_step_val,
                                                                                     val_iou_loss / epoch_step_val,
                                                                                     val_obj_loss / epoch_step_val,
                                                                                     val_cls_loss / epoch_step_val,
                                                                                     val_cls_ref_loss / epoch_step_val))

        #-----------------------------------------------#
        #   保存权值
        #-----------------------------------------------#
        if ema:
            save_state_dict = ema.ema.state_dict()
        else:
            save_state_dict = model.state_dict()

        if (epoch + 1) % save_period == 0 or epoch + 1 == Epoch:
            torch.save(save_state_dict, os.path.join(save_dir, "ep%03d-loss%.3f-val_loss%.3f.pth" % (epoch + 1, loss / epoch_step, val_loss / epoch_step_val)))

        if len(loss_history.val_loss) <= 1 or (val_loss / epoch_step_val) <= min(loss_history.val_loss):
            # print('Save best model to best_epoch_weights.pth')
            logger.info('Save best model to best_epoch_weights.pth')
            torch.save(save_state_dict, os.path.join(save_dir, "best_epoch_weights.pth"))
            
        torch.save(save_state_dict, os.path.join(save_dir, "last_epoch_weights.pth"))