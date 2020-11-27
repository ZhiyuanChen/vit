import os
import time

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms

import numpy as np
import subprocess

try:
    from apex.parallel import DistributedDataParallel as DDP
    from apex.fp16_utils import *
    from apex import amp, optimizers
    from apex.multi_tensor_apply import multi_tensor_applier
except ImportError:
    raise ImportError("Please install apex from https://www.github.com/nvidia/apex to run this example.")

from tqdm import tqdm

import models
import data
from utils import *
from run import parse


def main(args):
    global best_acc1
    best_acc1 = 0
    init(args)
    memory_format = torch.channels_last if args.channels_last else torch.contiguous_format

    log("creating model '{}'".format(args.arch))
    model = models.__dict__[args.arch](pretrained=args.pretrained)

    checkpoint = torch.load(args.checkpoint)
    model.load_state_dict(checkpoint)

    if args.resume:
        resume(model, args.resume)

    if args.sync_bn:
        import apex
        log("using apex synced BN")
        model = apex.parallel.convert_syncbn_model(model)

    model = model.cuda().to(memory_format=memory_format)

    args.lr = args.lr * float(args.batch_size*args.world_size) / 256.
    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    model, optimizer = amp.initialize(
        model, optimizer, opt_level=args.opt_level,
        keep_batchnorm_fp32=args.keep_batchnorm_fp32,
        loss_scale=args.loss_scale
    )

    if args.distributed:
        model = DDP(model)

    criterion = nn.CrossEntropyLoss().cuda()

    traindir = os.path.join(args.data, 'train')
    valdir = os.path.join(args.data, 'val')

    crop_size = 384
    val_size = 384

    log("loading dataset '{}'".format(args.data))
    val_transform = transforms.Compose([
        transforms.Resize(val_size),
        transforms.CenterCrop(crop_size),
    ])
    val_dataset, val_sampler, val_loader = \
        data.load_data(valdir, val_transform, args.batch_size, args.workers,
                       memory_format, shuffle=False, profile=args.profile)

    log("length of validation dataset '{}'".format(len(val_loader)))

    validate(val_loader, model, criterion, args)


def validate(val_loader, model, criterion, args):
    log('evaluating')
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    model.eval()
    end = time.time()

    for iteration, (input, target) in enumerate(val_loader):
        with torch.no_grad():
            output = model(input)
            loss = criterion(output, target)

        # measure accuracy
        acc1, acc5 = accuracy(output.data, target, topk=(1, 5))
        reduced_loss = loss.data

        # average loss and accuracy across processes for logging
        if args.distributed:
            reduced_loss, acc1, acc5 = reduce_tensors(
                reduced_loss, acc1, acc5, world_size=args.world_size)

        # to_python_float incurs a host<->device sync
        losses.update(to_python_float(reduced_loss), input.size(0))
        top1.update(to_python_float(acc1), input.size(0))
        top5.update(to_python_float(acc5), input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # TODO:  Change timings to mirror train().
        if iteration % args.print_freq == 0:
            log('Test: [{0}/{1}]\t'
                'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                'Speed {2:.3f} ({3:.3f})\t'
                'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                'Acc@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                  iteration, len(val_loader),
                  args.world_size * args.batch_size / batch_time.val,
                  args.world_size * args.batch_size / batch_time.avg,
                  batch_time=batch_time, loss=losses,
                  top1=top1, top5=top5))

    log(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
          .format(top1=top1, top5=top5))

    return top1.avg


if __name__ == '__main__':
    global args    
    args = parse()
    log('\nArguments:')
    log('\n'.join([f'{k}\t{v}' for k, v in vars(args).items()]))
    log('\n')

    main(args)
