# Copyright (c)  Westone Information Industry Inc.
# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# BEiT: https://github.com/microsoft/unilm/tree/master/beit
# MAE: https://github.com/facebookresearch/mae
# --------------------------------------------------------

import math
import sys
import os
import re
from typing import Iterable, Optional

import torch
import torch.nn.functional as F

from timm.utils import accuracy
import util.token_converter as token_converter

import util.misc as misc
import util.lr_sched as lr_sched

from dataset import hierarchical_dataset, AlignCollate


def train(model: torch.nn.Module, criterion: torch.nn.Module,
          data_loader, data_loader_val, optimizer: torch.optim.Optimizer,
          device: torch.device, num_iter: int, loss_scaler, max_norm: float = 0,
          log_writer=None,
          args=None,
          is_main_process=False):
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
        
    if is_main_process:
        log = open(f'{args.output_dir}/log.txt', 'a')
 
    max_accuracy = 0.0
    for data_iter_step, (images, _words,) in enumerate(metric_logger.log_every_iter(data_loader, num_iter, args.start_iter, print_freq), args.start_iter):
        # we use a per iteration (instead of per epoch) lr scheduler
        if data_iter_step % accum_iter == 0:
            lr_sched.adjust_learning_rate_iter(optimizer, data_iter_step, args)

        images = images.to(device, non_blocking=True)
        words = converter.encode(_words)

        with torch.cuda.amp.autocast():
            outputs, outputs_iter = model(images, words)
            loss = criterion(
                outputs.view(-1, outputs.shape[-1]), words.contiguous().view(-1))
            if len(outputs_iter) > 0:
                loss_iter = 0
                for res in outputs_iter:
                    loss_iter += criterion(res.view(-1,
                                           outputs.shape[-1]), words.contiguous().view(-1))
                loss_iter /= len(outputs_iter)
                loss = loss * 0.5 + loss_iter * 0.5

        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        loss /= accum_iter
        loss_scaler(loss, optimizer, clip_grad=max_norm,
                    parameters=model.parameters(), create_graph=False,
                    update_grad=(data_iter_step + 1) % accum_iter == 0)
        if (data_iter_step + 1) % accum_iter == 0:
            optimizer.zero_grad()

        torch.cuda.synchronize()

        metric_logger.update(loss=loss_value)
        min_lr = 10.
        max_lr = 0.
        for group in optimizer.param_groups:
            min_lr = min(min_lr, group["lr"])
            max_lr = max(max_lr, group["lr"])

        metric_logger.update(lr=max_lr)

        loss_value_reduce = misc.all_reduce_mean(loss_value)
        if log_writer is not None and (data_iter_step + 1) % accum_iter == 0:
            log_writer.add_scalar('loss', loss_value_reduce, data_iter_step)
            log_writer.add_scalar('lr', max_lr, data_iter_step)

        if args.output_dir and data_iter_step + 1 == args.num_iter:
            misc.save_model_iter(
                args=args, model=model, model_without_ddp=model.module, optimizer=optimizer,
                loss_scaler=loss_scaler, iteration=data_iter_step)

        if data_iter_step % (args.val_interval * args.accum_iter) == 0 or data_iter_step + 1 == args.num_iter:
            test_stats = evaluate(data_loader_val, model,
                                  device, converter, args)
            if is_main_process:
                print(
                    f"Accuracy of the network on the test images: {test_stats['acc']:.5f}")
                log.write(f"[{data_iter_step}/{num_iter}] iterAccuracy of the network on the test images: {test_stats['acc']:.5f}\n")
            if max_accuracy < test_stats["acc"]:
                max_accuracy = test_stats["acc"]
                if args.output_dir:
                    misc.save_model_iter(
                        args=args, model=model, model_without_ddp=model.module, optimizer=optimizer,
                        loss_scaler=loss_scaler, iteration=data_iter_step, is_best=True)
            if is_main_process:
                print(f'Max accuracy: {max_accuracy:.5f}')
                log.write(f"Max accuracy: {max_accuracy:.5f}\n")
                log.flush()
            model.train(True)

            if log_writer is not None:
                log_writer.add_scalar('perf/test_acc', test_stats['acc'], data_iter_step)
                log_writer.add_scalar('perf/test_loss', test_stats['loss'], data_iter_step)

    if is_main_process:
        log.close()
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def test(data_loader, model, device, opt, global_rank):
    """ evaluation with 6 benchmark evaluation datasets """

    eval_data_list = ['IIIT5k_3000', 'SVT',
                      'IC13_857', 'IC15_1811', 'SVTP', 'CUTE80']
    converter = token_converter.TokenLabelConverter(opt)

    evaluate(data_loader, model, device, converter, opt, name="")
    evaluation_batch_size = opt.batch_size

    list_accuracy = []
    total_evaluation_data_number = 0
    total_correct_number = 0

    if global_rank == 0:
        log = open(f'{opt.output_dir}/log_all_evaluation.txt', 'a')
        dashed_line = '-' * 80
        print(dashed_line)
        log.write(dashed_line + '\n')
    for eval_data in eval_data_list:
        eval_data_path = os.path.join(opt.data_path_val, eval_data)
        name = eval_data
        AlignCollate_evaluation = AlignCollate(
            imgH=opt.imgH, imgW=opt.imgW, opt=opt, is_eval=True)
        eval_data, eval_data_log = hierarchical_dataset(
            root=eval_data_path, opt=opt, global_rank=global_rank)
        sampler_val = torch.utils.data.SequentialSampler(eval_data)
        evaluation_loader = torch.utils.data.DataLoader(
            eval_data, batch_size=evaluation_batch_size,
            shuffle=False,
            sampler=sampler_val,
            num_workers=int(opt.num_workers),
            collate_fn=AlignCollate_evaluation,
            pin_memory=True,
            drop_last=False,
        )

        test_state = evaluate(
            evaluation_loader, model, device, converter, opt, name)
        accuracy_by_best_model = test_state['acc']
        list_accuracy.append(accuracy_by_best_model)
        total_evaluation_data_number += len(eval_data)
        total_correct_number += accuracy_by_best_model * len(eval_data)

        if global_rank == 0:
            log.write(eval_data_log)
            print(f'Acc {accuracy_by_best_model:0.5f}')
            log.write(f'Acc {accuracy_by_best_model:0.5f}\n')
            print(dashed_line)
            log.write(dashed_line + '\n')

    if global_rank == 0:
        total_accuracy = total_correct_number / total_evaluation_data_number
        evaluation_log = 'accuracy: '
        for name, accuracy in zip(eval_data_list, list_accuracy):
            evaluation_log += f'{name}: {accuracy:0.5f}\t'
        evaluation_log += f'total_accuracy: {total_accuracy:0.5f}\t'
        print(evaluation_log)
        log.write(evaluation_log + '\n')
        log.close()

    return None


@torch.no_grad()
def evaluate(data_loader, model, device, converter, opt, name=""):
    """ validation or evaluation """
    criterion = torch.nn.CrossEntropyLoss()

    metric_logger = misc.MetricLogger(delimiter="  ")
    header = 'Test: ' + name
    print(header)

    model.eval()
    for i, (image_tensors, labels, _) in enumerate(data_loader):
        batch_size = image_tensors.size(0)

        image = image_tensors.to(device)

        target = converter.encode(labels)

        with torch.cuda.amp.autocast():
            preds, preds_iter_list = model(image, target)

        loss = criterion(preds.contiguous().view(-1,
                         preds.shape[-1]), target.contiguous().view(-1))
        metric_logger.update(loss=loss.item())

        if opt.iter_correct > 0:
            acc_v = accrucy(preds, labels, converter, device, batch_size, opt)
            metric_logger.meters['acc_v'].update(acc_v, n=batch_size)
            acc_iter = accrucy(
                preds_iter_list[-1], labels, converter, device, batch_size, opt)
            metric_logger.meters['acc'].update(acc_iter, n=batch_size)
        else:
            acc = accrucy(preds, labels, converter, device, batch_size, opt)
            metric_logger.meters['acc'].update(acc, n=batch_size)

    metric_logger.synchronize_between_processes()

    if opt.iter_correct > 0:
        print('* Acc {acc.global_avg:.5f} Acc_iter0 {acc_v.global_avg:.4f} loss {losses.global_avg:.3f}'
              .format(acc=metric_logger.acc, acc_v=metric_logger.acc_v, losses=metric_logger.loss))
    else:
        print('* Acc {acc.global_avg:.5f} loss {losses.global_avg:.3f}'
              .format(acc=metric_logger.acc, losses=metric_logger.loss))

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


def accrucy(preds, labels, converter, device, batch_size, opt):
    _, preds_index = preds.topk(1, dim=-1, largest=True, sorted=True)
    preds_index = preds_index.view(-1, converter.batch_max_length)

    length_for_pred = torch.IntTensor(
        [converter.batch_max_length - 1] * batch_size).to(device)
    preds_str = converter.decode(preds_index[:, 1:], length_for_pred)

    n_correct = 0
    for gt, pred in zip(labels, preds_str):
        pred_EOS = pred.find('[s]')
        pred = pred[:pred_EOS]  # prune after "end of sentence" token ([s])

        # To evaluate 'case sensitive model' with alphanumeric and case insensitve setting.
        # if opt.sensitive and opt.data_filtering_off:
        if opt.sensitive:
            pred = pred.lower()
            gt = gt.lower()
            alphanumeric_case_insensitve = '0123456789abcdefghijklmnopqrstuvwxyz'
            out_of_alphanumeric_case_insensitve = f'[^{alphanumeric_case_insensitve}]'
            pred = re.sub(out_of_alphanumeric_case_insensitve, '', pred)
            gt = re.sub(out_of_alphanumeric_case_insensitve, '', gt)

        if pred == gt:
            n_correct += 1

    acc = n_correct / float(batch_size)

    return acc
