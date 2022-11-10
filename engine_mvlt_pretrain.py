# Copyright (c)  Westone Information Industry Inc.
# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# BEiT: https://github.com/microsoft/unilm/tree/master/beit
# MAE: https://github.com/facebookresearch/mae
# --------------------------------------------------------
import math
import sys
from typing import Iterable

import torch

import util.misc as misc
import util.lr_sched as lr_sched
import util.token_converter as token_converter

import numpy as np
from PIL import Image


def train(model: torch.nn.Module,
          data_loader, optimizer: torch.optim.Optimizer,
          device: torch.device, num_iter: int, loss_scaler,
          log_writer=None,
          args=None):
    model.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(
        window_size=1, fmt='{value:.6f}'))
    print_freq = 2

    converter = token_converter.TokenLabelConverter(args)

    accum_iter = args.accum_iter

    optimizer.zero_grad()

    if log_writer is not None:
        print('log_dir: {}'.format(log_writer.log_dir))

    for data_iter_step, (images, _words, w_mask) in enumerate(metric_logger.log_every_iter(data_loader, num_iter, args.start_iter, print_freq), args.start_iter):

        # we use a per iteration (instead of per epoch) lr scheduler
        if data_iter_step % accum_iter == 0:
            lr_sched.adjust_learning_rate_iter(optimizer, data_iter_step, args)

        images = images.to(device, non_blocking=True)
        words = converter.encode(_words)
        w_mask = w_mask.to(device, non_blocking=True).flatten(1).to(torch.bool)

        with torch.cuda.amp.autocast():
            loss_dict, res_dict, mask = model(
                images, words, w_mask, mask_ratio=args.mask_ratio)

        loss = loss_dict["loss"]
        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        loss /= accum_iter
        loss_scaler(loss, optimizer, parameters=model.parameters(),
                    update_grad=(data_iter_step + 1) % accum_iter == 0)
        if (data_iter_step + 1) % accum_iter == 0:
            optimizer.zero_grad()

        torch.cuda.synchronize()

        metric_logger.update(loss=loss_value)

        if args.exp_semantic:
            exp_v_loss = loss_dict["exp_v_loss"].item()
            metric_logger.update(exp_v_loss=exp_v_loss)
            exp_t_loss = loss_dict["exp_t_loss"].item()
            metric_logger.update(exp_t_loss=exp_t_loss)

        if args.imp_semantic:
            imp_v_loss = loss_dict["imp_v_loss"].item()
            metric_logger.update(imp_v_loss=imp_v_loss)
            imp_t_loss = loss_dict["imp_t_loss"].item()
            metric_logger.update(imp_t_loss=imp_t_loss)

        lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(lr=lr)

        loss_value_reduce = misc.all_reduce_mean(loss_value)
        if log_writer is not None and (data_iter_step + 1) % accum_iter == 0:
            log_writer.add_scalar('train_loss', loss_value_reduce, data_iter_step)
            log_writer.add_scalar('lr', lr, data_iter_step)

        if args.output_dir and ((data_iter_step + 1) % (args.save_model_freq * accum_iter) == 0 or data_iter_step + 1 == args.num_iter):
            misc.save_model_iter(
                args=args, model=model, model_without_ddp=model.module, optimizer=optimizer,
                loss_scaler=loss_scaler, iteration=data_iter_step + 1)
       
        if misc.is_main_process() and data_iter_step % (1000 * accum_iter) == 0:

            log_writer_add_images(data_iter_step, model, images, mask, res_dict, log_writer, args)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

def log_writer_add_images(data_iter_step, model, images, mask, res_dict, log_writer, args):

        im_list = []
        images = images.detach().to('cpu')
        im = images[0].numpy()
        im = im * 0.5+0.5
        im_list.append(im)

        mask = mask.detach().to('cpu')
        mask = mask.unsqueeze(-1).repeat(1, 1,
                                         model.module.patch_embed.patch_size[0]**2 * 3)
        mask = model.module.unpatchify(mask)
        im_masked = images[0]*(1-mask[0])
        im = im_masked.numpy()
        im = im * 0.5+0.5
        im_list.append(im)

        str_list = ["gt","mask"]
        if args.exp_semantic:
            exp_v = res_dict["exp_v"]

            pred_imgs = model.module.unpatchify(exp_v.detach().to('cpu'))
            im_paste = images[0] * (1 - mask[0]) + pred_imgs[0] * mask[0]
            im = im_paste.numpy()
            im = im * 0.5+0.5
            im_list.append(im)
            str_list.append("exp-predict")

        if args.imp_semantic:
            imp_v = res_dict["imp_v"]
            pred_imgs = model.module.unpatchify(imp_v.detach().to('cpu'))
            im_paste = images[0] * (1 - mask[0]) + pred_imgs[0] * mask[0]
            im = im_paste.to('cpu').numpy()
            im = im * 0.5+0.5
            im_list.append(im)
            str_list.append("imp-predict")
                
        im_list = np.concatenate(im_list, 1)
        log_writer.add_image(" | ".join(str_list), im_list, data_iter_step)
 
