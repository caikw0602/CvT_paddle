import math
import sys
import argparse
import os
import time


import paddle
import paddle.nn.functional as F

from cvt import build_cvt as build_model
from datasets import get_dataset
from datasets import get_data_loader
from utils import AverageMeter
from utils import write_log
from utils import get_logger
from utils import all_reduce_mean
from utils import skip_weight_decay_fn
from mixup import Mixup
from model_ema import ModelEma
from losses import SoftTargetCrossEntropyLoss, LabelSmoothingCrossEntropyLoss
from config import get_config
from config import update_config


def get_arguments():
    """return argumeents, this will overwrite the config by (1) yaml file (2) argument values"""
    parser = argparse.ArgumentParser('CvT')
    parser.add_argument('-cfg', type=str, default=None)
    parser.add_argument('-dataset', type=str, default=None)
    parser.add_argument('-data_path', type=str, default=None)
    parser.add_argument('-output', type=str, default=None)
    parser.add_argument('-batch_size', type=int, default=None)
    parser.add_argument('-batch_size_eval', type=int, default=None)
    parser.add_argument('-image_size', type=int, default=None)
    parser.add_argument('-accum_iter', type=int, default=None)
    parser.add_argument('-pretrained', type=str, default=None)
    parser.add_argument('-resume', type=str, default=None)
    parser.add_argument('-last_epoch', type=int, default=None)
    parser.add_argument('-eval', action='store_true')
    parser.add_argument('-amp', action='store_true')
    arguments = parser.parse_args()
    return arguments

def train(dataloader,
          model,
          optimizer,
          criterion,
          epoch,
          total_epochs,
          total_batches,
          debug_steps,
          accum_iter,
          model_ema,
          mixup_fn,
          amp_grad_scaler,
          local_logger,
          master_logger
          ):
    """Training for one epoch
    Args:
        dataloader: paddle.io.DataLoader, dataloader instance
        model: nn.Layer, a ViT model
        optimizer: nn.optimizer
        criterion: nn.XXLoss
        epoch: int, current epoch
        total_epochs: int, total num of epochs
        total_batches: int, total num of batches for one epoch
        debug_steps: int, num of iters to log info, default: 100
        accum_iter: int, num of iters for accumulating gradients, default: 1
        model_ema: ModelEma, model moving average instance
        mixup_fn: Mixup, mixup instance, default: None
        amp_grad_scaler: GradScaler, if not None pass the GradScaler and enable AMP, default: None
        local_logger: logger for local process/gpu, default: None
        master_logger: logger for main process, default: None
    Returns:
        train_loss_meter.avg: float, average loss on current process/gpu
        train_acc_meter.avg: float, average acc@1 on current process/gpu
        master_loss_meter.avg: float, average loss on all processes/gpus
        master_acc_meter.avg: float, average acc@1 on all processes/gpus
        train_time: float, training time
    """
    time_st = time.time()
    train_loss_meter = AverageMeter()
    train_acc_meter = AverageMeter()

    model.train()
    optimizer.clear_grad()

    for batch_id, data in enumerate(dataloader):
        # get data
        imgs = data[0]
        label = data[1]
        label_ori = label.clone()
        batch_size = imgs.shape[0]

        if mixup_fn is not None:
            imgs, label = mixup_fn(imgs, label_ori)

        # forward
        # 混合精度train
        with paddle.amp.auto_cast(amp_grad_scaler is not None):
            output = model(imgs)
            loss = criterion(output, label)

        loss_value = loss.item()
        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        loss = loss / accum_iter

        # backward and step
        if amp_grad_scaler is None: # fp32 training
            loss.backward()
            if ((batch_id + 1) % accum_iter == 0 or (batch_id + 1) == len(dataloader)):
                optimizer.step()
                optimizer.clear_grad()
        else:  # amp 混合精度train
            scaled_loss = amp_grad_scaler.scale(loss)
            scaled_loss.backward()
            if ((batch_id + 1) % accum_iter == 0) or (batch_id + 1 == len(dataloader)):
                # amp for param group reference: https://github.com/PaddlePaddle/Paddle/issues/37188
                amp_grad_scaler.step(optimizer)
                amp_grad_scaler.update()
                optimizer.clear_grad()

        # EMA
        if model_ema is not None:
            model_ema.update(model)

        # average of output and kd_output, same as eval mode
        pred = F.softmax(output)
        acc = paddle.metric.accuracy(pred, label_ori if mixup_fn else label_ori.unsqueeze(1)).item()

        # sync from other gpus for overall loss and acc
        # master_loss = all_reduce_mean(loss_value)
        # master_acc = all_reduce_mean(acc)
        # master_batch = all_reduce_mean(batch_size)

        # master_loss_meter.update(master_loss, master_batch)
        # master_acc_meter.update(master_acc, master_batch)
        train_loss_meter.update(loss_value, batch_size)
        train_acc_meter.update(acc, batch_size)

        if batch_id % debug_steps == 0 or batch_id + 1 == len(dataloader):
            general_message = (f"Epoch[{epoch:03d}/{total_epochs:03d}], "
                               f"Step[{batch_id:04d}/{total_batches:04d}], "
                               f"Lr: {optimizer.get_lr():04f}, ")
            local_message = (general_message +
                             f"Loss: {loss_value:.4f} ({train_loss_meter.avg:.4f}), "
                             f"Avg Acc: {train_acc_meter.avg:.4f}")
            # master_message = (general_message +
            #                   f"Loss: {master_loss:.4f} ({master_loss_meter.avg:.4f}), "
            #                   f"Avg Acc: {master_acc_meter.avg:.4f}")
            write_log(local_logger, master_logger, local_message)

    # paddle.distributed.barrier()
    time_end = time.time()
    train_time = time_end - time_st

    return (train_loss_meter.avg,
            train_acc_meter.avg,
            train_time)

@paddle.no_grad()
def validate(dataloader,
             model,
             criterion,
             total_batches,
             debug_steps=100,
             local_logger=None,
             master_logger=None):
    """Validation for the whole dataset
    Args:
        dataloader: paddle.io.DataLoader, dataloader instance
        model: nn.Layer, a ViT model
        total_batches: int, total num of batches for one epoch
        debug_steps: int, num of iters to log info, default: 100
        local_logger: logger for local process/gpu, default: None
        master_logger: logger for main process, default: None
    Returns:
        val_loss_meter.avg: float, average loss on current process/gpu
        val_acc1_meter.avg: float, average top1 accuracy on current processes/gpus
        val_acc5_meter.avg: float, average top5 accuracy on current processes/gpus
        master_loss_meter.avg: float, average loss on all processes/gpus
        master_acc1_meter.avg: float, average top1 accuracy on all processes/gpus
        master_acc5_meter.avg: float, average top5 accuracy on all processes/gpus
        val_time: float, validation time
    """
    model.eval()
    val_loss_meter = AverageMeter()
    val_acc1_meter = AverageMeter()
    val_acc5_meter = AverageMeter()

    time_st = time.time()

    for batch_id, data in enumerate(dataloader):
        # get data
        imgs = data[0]
        label = data[1]
        batch_size = imgs.shape[0]

        output = model(imgs)
        loss = criterion(output, label)
        loss_value = loss.item()

        pred = F.softmax(output)
        acc1 = paddle.metric.accuracy(pred, label.unsqueeze(1), k=1).item()
        acc5 = paddle.metric.accuracy(pred, label.unsqueeze(1), k=5).item()

        # sync from other gpus for overall loss and acc
        val_loss_meter.update(loss_value, batch_size)
        val_acc1_meter.update(acc1, batch_size)
        val_acc5_meter.update(acc5, batch_size)

        if batch_id % debug_steps == 0:
            local_message = (f"Step[{batch_id:04d}/{total_batches:04d}], "
                             f"Avg Loss: {val_loss_meter.avg:.4f}, "
                             f"Avg Acc@1: {val_acc1_meter.avg:.4f}, "
                             f"Avg Acc@5: {val_acc5_meter.avg:.4f}")
            write_log(local_logger, master_logger, local_message)

    paddle.distributed.barrier()
    time_end = time.time()
    time_val = time_end - time_st

    return (val_loss_meter.avg,
            val_acc1_meter.avg,
            val_acc5_meter.avg,
            time_val)

def main_worker(*args):
    """main method for each process"""
    # STEP 0: Preparation
    paddle.device.set_device('gpu')
    # paddle.distributed.init_parallel_env()
    # world_size = paddle.distributed.get_world_size()
    # local_rank = paddle.distributed.get_rank()
    config = args[0]
    last_epoch = config.TRAIN.LAST_EPOCH
    # seed = config.SEED + local_rank
    seed = config.SEED
    paddle.seed(seed)
    local_logger, master_logger = get_logger(config.SAVE)
    message = (f'----- {config}')
    write_log(local_logger, master_logger, message)

    # STEP 1: Create model
    model = build_model(config)
    # define model ema
    model_ema = None
    if not config.EVAL and config.TRAIN.MODEL_EMA:
        model_ema = ModelEma(model, decay=config.TRAIN.MODEL_EMA_DECAY)
        if config.TRAIN.MODEL_EMA_FORCE_CPU:
            model_ema.to('cpu')

    # STEP 2: Create train and val dataloader
    if not config.EVAL:
        dataset_train = args[1]
        dataloader_train = get_data_loader(config, dataset_train, True, False)
        total_batch_train = len(dataloader_train)
        message = f'----- Total # of train batch (single gpu): {total_batch_train}'
        write_log(local_logger, master_logger, message)

    dataset_val = args[2]
    dataloader_val = get_data_loader(config, dataset_val, False, False)
    total_batch_val = len(dataloader_val)
    message = f'----- Total # of val batch (single gpu): {total_batch_val}'
    write_log(local_logger, master_logger, message)

    # STEP 3: (Optional) Define Mixup function
    mixup_fn = None
    # if (config.TRAIN.MIXUP_PROB > 0 or config.TRAIN.CUTMIX_ALPHA > 0 or
    #     config.TRAIN.CUTMIX_MINMAX is not None):
    #     mixup_fn = Mixup(mixup_alpha=config.TRAIN.MIXUP_ALPHA,
    #                      cutmix_alpha=config.TRAIN.CUTMIX_ALPHA,
    #                      cutmix_minmax=config.TRAIN.CUTMIX_MINMAX,
    #                      prob=config.TRAIN.MIXUP_PROB,
    #                      switch_prob=config.TRAIN.MIXUP_SWITCH_PROB,
    #                      mode=config.TRAIN.MIXUP_MODE,
    #                      label_smoothing=config.TRAIN.SMOOTHING)#

    # STEP 4: Define loss/criterion
    if mixup_fn is not None:
        criterion = SoftTargetCrossEntropyLoss()
    elif config.TRAIN.SMOOTHING:
        criterion = LabelSmoothingCrossEntropyLoss()
    else:
        criterion = paddle.nn.CrossEntropyLoss()
    # Use CrossEntropyLoss for val
    criterion_val = paddle.nn.CrossEntropyLoss()

    # STEP 5: Define optimizer and lr_scheduler
    if not config.EVAL:
        # set lr according to batch size and world size
        if config.TRAIN.LINEAR_SCALED_LR is not None:
            effective_batch_size = config.DATA.BATCH_SIZE * config.TRAIN.ACCUM_ITER
            config.TRAIN.BASE_LR = (
                    config.TRAIN.BASE_LR * effective_batch_size / config.TRAIN.LINEAR_SCALED_LR
            )
            config.TRAIN.WARMUP_START_LR = (
                    config.TRAIN.WARMUP_START_LR * effective_batch_size / config.TRAIN.LINEAR_SCALED_LR
            )
            config.TRAIN.END_LR = (
                    config.TRAIN.END_LR * effective_batch_size / config.TRAIN.LINEAR_SCALED_LR
            )
            message = (f'Base lr is scaled to: {config.TRAIN.BASE_LR}, '
                       f'warmup start lr is scaled to: {config.TRAIN.WARMUP_START_LR}, '
                       f'end lr is scaled to: {config.TRAIN.BASE_LR}')
            write_log(local_logger, master_logger, message)
        # define scaler for amp training
        amp_grad_scaler = paddle.amp.GradScaler() if config.AMP else None
        # warmup + cosine lr scheduler
        if config.TRAIN.WARMUP_EPOCHS > 0:
            cosine_lr_scheduler = paddle.optimizer.lr.CosineAnnealingDecay(
                learning_rate=config.TRAIN.BASE_LR,
                T_max=config.TRAIN.NUM_EPOCHS - config.TRAIN.WARMUP_EPOCHS,
                eta_min=config.TRAIN.END_LR,
                last_epoch=-1
            )
            lr_scheduler = paddle.optimizer.lr.LinearWarmup(
                learning_rate=cosine_lr_scheduler,
                warmup_steps=config.TRAIN.WARMUP_EPOCHS,
                start_lr=config.TRAIN.WARMUP_START_LR,
                end_lr=config.TRAIN.BASE_LR,
                last_epoch=config.TRAIN.LAST_EPOCH
            )
        else:
            lr_scheduler = paddle.optimizer.lr.CosineAnnealingDecay(
                learning_rate=config.TRAIN.BASE_LR,
                T_max=config.TRAIN.NUM_EPOCHS,
                eta_min=config.TRAIN.END_LR,
                last_epoch=config.TRAIN.LAST_EPOCH
            )

        # set gradient clip
        if config.TRAIN.GRAD_CLIP:
            clip = paddle.nn.ClipGradByGlobalNorm(config.TRAIN.GRAD_CLIP)
        else:
            clip = None
        # set optimizer
        optimizer = paddle.optimizer.AdamW(
            parameters=model.parameters(),
            learning_rate=lr_scheduler,
            beta1=config.TRAIN.OPTIMIZER.BETAS[0],
            beta2=config.TRAIN.OPTIMIZER.BETAS[1],
            weight_decay=config.TRAIN.WEIGHT_DECAY,
            epsilon=config.TRAIN.OPTIMIZER.EPS,
            grad_clip=clip,
            apply_decay_param_fun=skip_weight_decay_fn(
                model,  # skip bn and bias
                ['pos_embed', 'cls_token', 'dist_token']),  # skip custom ops
        )
    # STEP 6: (Optional) Load pretrained model weights for evaluation or finetuning
    if config.MODEL.PRETRAINED:
        assert os.path.isfile(config.MODEL.PRETRAINED) is True
        model_state = paddle.load(config.MODEL.PRETRAINED)
        if 'model' in model_state: # load state_dict with multi items: model, optimier, and epoch
            # pretrain only load model weight, opt and epoch are ignored
            if 'model_ema' in model_state:
                model_state = model_state['model_ema']
            else:
                model_state = model_state['model']
        model.set_state_dict(model_state)
        message = f"----- Pretrained: Load model state from {config.MODEL.PRETRAINED}"
        write_log(local_logger, master_logger, message)

    # STEP 7: (Optional) Load model weights and status for resume training
    if config.MODEL.RESUME:
        assert os.path.isfile(config.MODEL.RESUME) is True
        model_state = paddle.load(config.MODEL.RESUME)
        if 'model' in model_state: # load state_dict with multi items: model, optimier, and epoch
            model.set_state_dict(model_state['model'])

            if 'optimizer' in model_state:
                optimizer.set_state_dict(model_state['optimizer'])
            if 'epoch' in model_state:
                config.TRAIN.LAST_EPOCH = model_state['epoch']
                last_epoch = model_state['epoch']
            if 'lr_scheduler' in model_state:
                lr_scheduler.set_state_dict(model_state['lr_scheduler'])
            if 'amp_grad_scaler' in model_state and amp_grad_scaler is not None:
                amp_grad_scaler.load_state_dict(model_state['amp_grad_scaler'])
            if config.TRAIN.MODEL_EMA:
                model_ema.module.set_state_dict(model_state['model_ema'])

            lr_scheduler.step(last_epoch + 1)

            message = (f"----- Resume Training: Load model from {config.MODEL.RESUME}, w/t "
                       f"opt = [{'optimizer' in model_state}], "
                       f"lr_scheduler = [{'lr_scheduler' in model_state}], "
                       f"model_ema = [{'model_ema' in model_state}], "
                       f"epoch = [{model_state.get('epoch', -1)}], "
                       f"amp_grad_scaler = [{'amp_grad_scaler' in model_state}]")
            write_log(local_logger, master_logger, message)
        else: # direct load pdparams without other items
            message = f"----- Resume Training: Load {config.MODEL.RESUME}, w/o opt/epoch/scaler"
            write_log(local_logger, master_logger, message, 'warning')
            model.set_state_dict(model_state)
            lr_scheduler.step(last_epoch + 1)

    # STEP 8: Enable model data parallelism on multi processes
    # model = paddle.DataParallel(model)

    # STEP 9: (Optional) Run evaluation and return
    if config.EVAL:
        write_log(local_logger, master_logger, '----- Start Validation')
        val_loss, val_acc1, val_acc5, val_time = validate(
            dataloader=dataloader_val,
            model=model,
            criterion=criterion_val,
            total_batches=total_batch_val,
            debug_steps=config.REPORT_FREQ,
            local_logger=local_logger,
            master_logger=master_logger
        )
        local_message = ("----- Validation: " +
                         f"Validation Loss: {val_loss:.4f}, " +
                         f"Validation Acc@1: {val_acc1:.4f}, " +
                         f"Validation Acc@5: {val_acc5:.4f}, " +
                         f"time: {val_time:.2f}")
        write_log(local_logger, master_logger, local_message)
        return

    # STEP 10: Run training
    write_log(local_logger, master_logger, f"----- Start training from epoch {last_epoch+1}.")
    for epoch in range(last_epoch + 1, config.TRAIN.NUM_EPOCHS + 1):
        # Train one epoch
        write_log(local_logger, master_logger, f'Train epoch {epoch}. lr={optimizer.get_lr():.6e}')
        train_loss, train_acc, train_time = train(
            dataloader=dataloader_train,
            model=model,
            optimizer=optimizer,
            criterion=criterion,
            epoch=epoch,
            total_epochs=config.TRAIN.NUM_EPOCHS,
            total_batches=total_batch_train,
            debug_steps=config.REPORT_FREQ,
            accum_iter=config.TRAIN.ACCUM_ITER,
            model_ema=model_ema,
            mixup_fn=mixup_fn,
            amp_grad_scaler=amp_grad_scaler,
            local_logger=local_logger,
            master_logger=master_logger
        )

        # update lr
        lr_scheduler.step()

        general_message = (f"----- Epoch[{epoch:03d}/{config.TRAIN.NUM_EPOCHS:03d}], "
                           f"Lr: {optimizer.get_lr():.4f}, "
                           f"time: {train_time:.2f}, ")
        local_message = (general_message +
                         f"Train Loss: {train_loss:.4f}, "
                         f"Train Acc: {train_acc:.4f}")
        write_log(local_logger, master_logger, local_message)

        # Evaluation (optional)
        if epoch % config.REPORT_FREQ or epoch == config.TRAIN.NUM_EPOCHS:
            write_log(local_logger, master_logger, f'----- Validation after Epoch: {epoch}')
            val_loss, val_acc1, val_acc5, val_time = validate(
                dataloader=dataloader_val,
                model=model,
                criterion=criterion_val,
                total_batches=total_batch_val,
                debug_steps=config.REPORT_FREQ,
                local_logger=local_logger,
                master_logger=master_logger)
            local_message = (f"----- Epoch[{epoch:03d}/{config.TRAIN.NUM_EPOCHS:03d}], " +
                             f"Validation Loss: {val_loss:.4f}, " +
                             f"Validation Acc@1: {val_acc1:.4f}, " +
                             f"Validation Acc@5: {val_acc5:.4f}, " +
                             f"time: {val_time:.2f}")
            write_log(local_logger, master_logger, local_message)

        # Save model weights and training status
        if True:
            if epoch % config.SAVE_FREQ == 0 or epoch == config.TRAIN.NUM_EPOCHS:
                model_path = os.path.join(
                    config.SAVE, f"Epoch-{epoch}-Loss-{val_loss}.pdparams")
                state_dict = dict()
                state_dict['model'] = model.state_dict()
                if model_ema is not None:
                    state_dict['model_ema'] = model_ema.state_dict()
                state_dict['optimizer'] = optimizer.state_dict()
                state_dict['epoch'] = epoch
                if lr_scheduler is not None:
                    state_dict['lr_scheduler'] = lr_scheduler.state_dict()
                if amp_grad_scaler is not None:
                    state_dict['amp_grad_scaler'] = amp_grad_scaler.state_dict()
                paddle.save(state_dict, model_path)
                message = (f"----- Save model: {model_path}")
                write_log(local_logger, master_logger, message)


def main():
    # config updated order: (1) default in config.py, (2) yaml file, (3) arguments
    config = update_config(get_config(), get_arguments())

    # set output folder
    config.SAVE = os.path.join(config.SAVE,
        f"{'eval' if config.EVAL else 'train'}-{time.strftime('%Y%m%d-%H-%M')}")
    if not os.path.exists(config.SAVE):
        os.makedirs(config.SAVE, exist_ok=True)

    # print("==========================CvT Model======================")
    # model = build_model(config)
    # paddle.summary(model, input_size=(2, 3, 224, 224))
    #
    # out = model(paddle.randn(shape=(2, 3, 224, 224)))
    # print('Shape of out :', out.shape, "[PaddlePaddle Version]")

    # get train dataset if in train mode and val dataset
    dataset_train = get_dataset(config, is_train=True) if not config.EVAL else None
    dataset_val = get_dataset(config, is_train=False)

    # dist spawn lunch: use CUDA_VISIBLE_DEVICES to set available gpus
    # paddle.distributed.spawn(main_worker, args=(config, dataset_train, dataset_val))
    main_worker(config, dataset_train, dataset_val)


if __name__ == '__main__':
    main()