import os
import time

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms

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
from validate import validate
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
    del checkpoint['encoder.pos_embed']
    del checkpoint['head.weight']
    del checkpoint['head.bias']
    model.load_state_dict(checkpoint, strict=False)

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
    # scheduler = LRScheduler(optimizer)

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
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(crop_size),
        transforms.RandomHorizontalFlip(),
    ])
    val_transform = transforms.Compose([
        transforms.Resize(val_size),
        transforms.CenterCrop(crop_size),
    ])
    train_dataset, train_sampler, train_loader = \
        data.load_data(traindir, train_transform, args.batch_size,
                       args.workers, memory_format, profile=args.profile)
    val_dataset, val_sampler, val_loader = \
        data.load_data(valdir, val_transform, args.batch_size, args.workers,
                       memory_format, shuffle=False, profile=args.profile)

    log("length of traning dataset '{}'".format(len(train_loader)))
    log("length of validation dataset '{}'".format(len(val_loader)))

    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)

        train(train_loader, model, criterion, optimizer, epoch, args.warmup_epoch, args)

        acc1 = validate(val_loader, model, criterion, args)

        if args.local_rank == 0:
            is_best = acc1 > best_acc1
            best_acc1 = max(acc1, best_acc1)
            save_checkpoint({
                'epoch': epoch + 1,
                'arch': args.arch,
                'state_dict': model.state_dict(),
                'best_acc1': best_acc1,
                'optimizer' : optimizer.state_dict(),
            }, is_best)


def train(train_loader, model, criterion, optimizer, epoch, warmup_epoch, args):
    log('training {}'.format(epoch))
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    model.train()
    end = time.time()

    for iteration, (input, target) in enumerate(train_loader):
        adjust_learning_rate(args.lr, optimizer, epoch, warmup_epoch, iteration, len(train_loader))

        output = model(input)

        loss = criterion(output, target)
        optimizer.zero_grad()
        with amp.scale_loss(loss, optimizer) as scaled_loss:
            scaled_loss.backward()
        optimizer.step()

        if iteration % args.print_freq == 0:
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
            torch.cuda.synchronize()
            batch_time.update((time.time() - end) / args.print_freq)
            end = time.time()

            log('Epoch: [{0}][{1}/{2}]\t'
                'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                'Speed {3:.3f} ({4:.3f})\t'
                'Loss {loss.val:.10f} ({loss.avg:.4f})\t'
                'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                'Acc@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                    epoch, iteration, len(train_loader),
                    args.world_size*args.batch_size/batch_time.val,
                    args.world_size*args.batch_size/batch_time.avg,
                    batch_time=batch_time,
                    loss=losses, top1=top1, top5=top5))


if __name__ == '__main__':
    global args    
    args = parse()
    log('\nArguments:')
    log('\n'.join([f'{k}\t{v}' for k, v in vars(args).items()]))

    main(args)
