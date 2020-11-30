import os
import time

import torch
import torch.nn as nn

from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter

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
    global best_acc1, writer, save_dir
    best_acc1, writer, save_dir = 0, None, None
    _experiment, _writer, _save_dir = init(args)
    if _writer is not None:
        writer = _writer
    if _save_dir is not None:
        save_dir = _save_dir
    memory_format = torch.channels_last if args.channels_last else torch.contiguous_format

    log("creating model '{}'".format(args.arch))
    model = models.__dict__[args.arch](**vars(args))

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

    args.learning_rate = args.learning_rate * float(args.batch_size*args.world_size) / 4096.
    optimizer = torch.optim.SGD(model.parameters(), 0,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)
    scheduler = LRScheduler(
        optimizer, epochs=args.epochs, strategy=args.strategy,
        lr=args.learning_rate, param=args.param,
        warmup_steps=args.warmup_steps, warmup_begin_lr=args.warmup_lr)

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
                       args.workers, memory_format)
    val_dataset, val_sampler, val_loader = \
        data.load_data(valdir, val_transform, args.batch_size, args.workers,
                       memory_format, shuffle=False)

    log("length of traning dataset '{}'".format(len(train_loader)))
    log("length of validation dataset '{}'".format(len(val_loader)))

    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)

        train(train_loader, model, criterion, optimizer, scheduler, writer, epoch, args)

        acc1 = validate(val_loader, model, criterion, writer, args)

        if args.local_rank == 0:
            is_best = acc1 > best_acc1
            best_acc1 = max(acc1, best_acc1)
            if is_best:
                torch.save({
                    'epoch': epoch,
                    'arch': args.arch,
                    'state_dict': model.state_dict(),
                    'acc1': acc1,
                    'optimizer' : optimizer.state_dict(),
                    'scheduler' : scheduler.state_dict(),
                }, os.path.join(save_dir, f'{acc1}.pth'))
            elif epoch % args.save_freq == 0:
                torch.save({
                    'epoch': epoch,
                    'arch': args.arch,
                    'state_dict': model.state_dict(),
                    'acc1': acc1,
                    'optimizer' : optimizer.state_dict(),
                    'scheduler' : scheduler.state_dict(),
                }, os.path.join(save_dir, f'epoch_{epoch}.pth'))

def train(loader, model, criterion, optimizer, scheduler, writer, epoch, args):
    log('training {}'.format(epoch))
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    model.train()
    end = time.time()

    iteration = 0
    fetcher = data.DataFetcher(loader)
    images, target = next(fetcher)

    while images is not None:
        iteration += 1
        if args.profile >= 0 and iteration == args.profile:
            log("Profiling begun at iteration {}".format(iteration))
            torch.cuda.cudart().cudaProfilerStart()
        if args.profile >= 0: torch.cuda.nvtx.range_push("Body of iteration {}".format(iteration))

        if args.profile >= 0: torch.cuda.nvtx.range_push("forward")
        output = model(images)
        if args.profile >= 0: torch.cuda.nvtx.range_pop()

        loss = criterion(output, target)
        optimizer.zero_grad()

        if args.profile >= 0: torch.cuda.nvtx.range_push("backward")
        with amp.scale_loss(loss, optimizer) as scaled_loss:
            scaled_loss.backward()
        if args.profile >= 0: torch.cuda.nvtx.range_pop()

        if args.gradient_clip:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.gradient_clip)

        if args.profile >= 0: torch.cuda.nvtx.range_push("optimizer.step()")
        optimizer.step()
        if args.profile >= 0: torch.cuda.nvtx.range_pop()

        if args.profile >= 0: torch.cuda.nvtx.range_push("optimizer.step()")
        scheduler.step()
        if args.profile >= 0: torch.cuda.nvtx.range_pop()

        if iteration % args.print_freq == 0:
            # measure accuracy
            acc1, acc5 = accuracy(output.data, target, topk=(1, 5))
            reduced_loss = loss.data

            # average loss and accuracy across processes for logging
            if args.distributed:
                reduced_loss, acc1, acc5 = reduce_tensors(
                    reduced_loss, acc1, acc5, world_size=args.world_size)

            # to_python_float incurs a host<->device sync
            losses.update(data.to_python_float(reduced_loss), images.size(0))
            top1.update(data.to_python_float(acc1), images.size(0))
            top5.update(data.to_python_float(acc5), images.size(0))
            lr = optimizer.param_groups[0]['lr'] * 1000

            # measure elapsed time
            torch.cuda.synchronize()
            batch_time.update((time.time() - end) / args.print_freq)
            end = time.time()

            if args.tensorboard and args.local_rank == 0:
                total_iter = iteration + iteration * epoch
                writer.add_scalar('train/loss', losses.val, total_iter)
                writer.add_scalar('train/acc1', top1.val, total_iter)
                writer.add_scalar('train/acc5', top5.val, total_iter)

            log('Epoch: [{0}][{1}/{2}]\t'
                'LR {lr:.4f}\t'
                'Time {batch_time.val:.2f} ({batch_time.avg:.2f})\t'
                # 'Speed {3:.3f} ({4:.3f})\t'
                'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                'Acc@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                    epoch, iteration, len(loader),
                    # args.world_size*args.batch_size/batch_time.val,
                    # args.world_size*args.batch_size/batch_time.avg,
                    lr=lr, batch_time=batch_time,
                    loss=losses, top1=top1, top5=top5))

        if args.profile >= 0: torch.cuda.nvtx.range_push("next(fetcher)")
        images, target = next(fetcher)
        if args.profile >= 0: torch.cuda.nvtx.range_pop()

        if args.profile >= 0: torch.cuda.nvtx.range_pop()

        if args.profile >= 0 and iteration == args.profile + 10:
            log("Profiling ended at iteration {}".format(iteration))
            torch.cuda.cudart().cudaProfilerStop()
            quit()


if __name__ == '__main__':
    global args    
    args = parse()
    log('\nArguments:')
    log('\n'.join([f'{k}\t{v}' for k, v in vars(args).items()]))

    main(args)
