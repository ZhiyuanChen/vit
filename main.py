import os
import time

import torch
import torch.nn as nn

from torchvision import transforms

from tqdm import tqdm

import models
import data
from utils import *
from run import parse
from timm.data import create_transform, Mixup
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy

try:
    import apex
    from apex.parallel import DistributedDataParallel as DDP
    from apex import amp, optimizers

    sync_bn = apex.parallel.convert_syncbn_model
    APEX_AVAILABLE = True
except ImportError:
    from torch.nn.parallel import DistributedDataParallel as DDP

    sync_bn = nn.SyncBatchNorm.convert_sync_batchnorm
    APEX_AVAILABLE = False


def main(args):
    global best_acc1, experiment, logger, writer, save_dir
    best_acc1, experiment, logger, writer, save_dir = init(args)
    args.scale_factor = float(
        args.batch_size * args.world_size) * args.accum_steps / args.lr_factor
    args.lr = args.lr * args.scale_factor
    args.final_lr = args.final_lr * args.scale_factor
    args.warmup_steps = args.warmup_steps // args.scale_factor

    print('\nArguments:' + '\n'.join([f'{k}\t{v}' for k, v in vars(args).items()]))

    mixup_fn = None
    mixup_active = args.mixup > 0 or args.cutmix > 0. or args.cutmix_minmax is not None
    if mixup_active:
        mixup_fn = Mixup(
            mixup_alpha=args.mixup, cutmix_alpha=args.cutmix, cutmix_minmax=args.cutmix_minmax,
            prob=args.mixup_prob, switch_prob=args.mixup_switch_prob, mode=args.mixup_mode,
            label_smoothing=args.smoothing, num_classes=args.num_classes)

    memory_format = torch.channels_last if args.channels_last else torch.contiguous_format

    print("creating model '{}'".format(args.arch))
    model = getattr(models, args.arch)(**vars(args))

    if args.sync_bn:
        print('Convert model with Sync BatchNormal')
        model = sync_bn(model)

    model = model.cuda().to(memory_format=memory_format)
    optimizer = None
    scheduler = None

    if args.train:
        if args.optimizer in ('SGD', 'RMSprop'):
            optimizer = getattr(torch.optim, args.optimizer)(
                model.parameters(), args.lr, momentum=args.momentum,
                weight_decay=args.weight_decay)
        else:
            optimizer = getattr(torch.optim, args.optimizer)(
                model.parameters(), args.lr,
                weight_decay=args.weight_decay)
        print("loading training set from '{}'".format(args.train_data))
        # color_jitter = (float(args.color_jitter),) * 3
        # train_transform = transforms.Compose([
        #     transforms.RandomResizedCrop(args.img_size),
        #     transforms.RandomHorizontalFlip(),
        #     transforms.ColorJitter(*color_jitter),
        #     getattr(autoaugment, 'ImageNet')
        # ])
        train_transform = create_transform(
            input_size=args.img_size,
            is_training=True,
            use_prefetcher=True,
            color_jitter=args.color_jitter,
            auto_augment=args.auto_augment,
            interpolation=args.train_interpolation,
            re_prob=args.random_erase_prob,
            re_mode=args.random_erase_mode,
            re_count=args.random_erase_count,
        )
        train_transform.transforms = train_transform.transforms[:-1]

        train_dataset, train_sampler, train_loader = \
            data.load_data(args.train_data, train_transform, memory_format,
                           shuffle=False, **vars(args))

        print("length of traning dataset '{}'".format(len(train_loader)))
        scheduler = LRScheduler(optimizer,
                                steps=args.epochs * len(train_loader),
                                final_lr=args.final_lr,
                                strategy=args.strategy,
                                warmup_steps=args.warmup_steps,
                                accum_steps=args.accum_steps)
    if args.validate:
        print("loading validation set from '{}'".format(args.val_data))
        # val_transform = create_transform(
        #    input_size=args.img_size,
        #    is_training=False,
        #    use_prefetcher=True,
        #    interpolation=args.train_interpolation,
        # )
        val_transform = transforms.Compose([
            transforms.Resize(args.img_size),
            transforms.CenterCrop(args.img_size),
        ])
        val_dataset, val_sampler, val_loader = \
            data.load_data(args.val_data, val_transform, memory_format,
                           shuffle=False, **vars(args))
        print("length of validation dataset '{}'".format(len(val_loader)))

    if args.checkpoint:
        load_checkpoint(model, args, optimizer, scheduler, best_acc1)

    if APEX_AVAILABLE:
        model, optimizer = amp.initialize(
            model,
            optimizer,
            opt_level=args.opt_level,
            keep_batchnorm_fp32=args.keep_batchnorm_fp32,
            loss_scale=args.loss_scale)

    if args.distributed:
        model = DDP(model)

    train_criterion = nn.CrossEntropyLoss()
    val_criterion = nn.CrossEntropyLoss()
    if args.mixup > 0.:
        train_criterion = SoftTargetCrossEntropy()
    elif args.smoothing:
        train_criterion = LabelSmoothingCrossEntropy(smoothing=args.smoothing)
    train_criterion.cuda()

    for epoch in range(args.start_epoch, args.epochs):

        if args.train:
            if args.distributed:
                train_sampler.set_epoch(epoch)
            acc1, acc5, loss = train(train_loader, model, train_criterion,
                                      optimizer, scheduler, epoch, args,
                                      logger, writer, mixup_fn)
        if args.validate:
            val_writer = writer if not args.train else None
            acc1, acc5, loss = validate(val_loader, model, val_criterion, args, 
                                        logger, val_writer)

        # This impliies args.tensorboard and proc_id == 0:
        if writer:
            writer.add_scalar('validate/loss', loss, epoch)
            writer.add_scalar('validate/acc1', acc1, epoch)
            writer.add_scalar('validate/acc5', acc5, epoch)

        if args.proc_id == 0:
            is_best = acc1 > best_acc1
            best_acc1 = max(acc1, best_acc1)
            net = model.module if args.distributed else model
            if (epoch + 1) % args.save_freq == 0 or is_best:
                state_dict = {
                    'epoch': epoch,
                    'arch': args.arch,
                    'args': vars(args),
                    'state_dict': net.state_dict(),
                    'acc1': acc1,
                    'acc5': acc5,
                    'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict(),
                }
                save_checkpoint(state_dict, is_best, save_dir, f'epoch-{epoch}.pth')


def train(loader, model, criterion, optimizer, scheduler, epoch, args, logger=None, writer=None, mixup_fn=None):
    print('training {}'.format(epoch))
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    model.train()
    end = time.time()

    iteration = 0
    fetcher = data.DataFetcher(loader)
    images, labels = next(fetcher)

    while images is not None:
        iteration += 1
        if 0 <= args.profile == iteration:
            print("Profiling begun at iteration {}".format(iteration))
            torch.cuda.cudart().cudaProfilerStart()
        if args.profile >= 0:
            torch.cuda.nvtx.range_push(
                "Body of iteration {}".format(iteration))

        targets = labels
        if mixup_fn is not None:
            images, targets = mixup_fn(images, labels)

        if args.profile >= 0: torch.cuda.nvtx.range_push("forward")
        output = model(images)
        if args.profile >= 0: torch.cuda.nvtx.range_pop()

        loss = criterion(output, targets)
        loss = loss / args.accum_steps

        if args.profile >= 0: torch.cuda.nvtx.range_push("backward")
        if APEX_AVAILABLE:
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()
        if args.profile >= 0: torch.cuda.nvtx.range_pop()

        if args.gradient_clip:
            torch.nn.utils.clip_grad_norm_(model.parameters(),
                                           args.gradient_clip)

        if (iteration + 1) % args.accum_steps == 0:
            if args.profile >= 0:
                torch.cuda.nvtx.range_push("optimizer.step()")
            optimizer.step()
            optimizer.zero_grad()
            if args.profile >= 0: torch.cuda.nvtx.range_pop()

            if args.profile >= 0:
                torch.cuda.nvtx.range_push("scheduler.step()")
            scheduler.step()
            if args.profile >= 0: torch.cuda.nvtx.range_pop()

        if iteration % args.print_freq == 0:
            # measure accuracy
            acc1, acc5 = accuracy(output.data, labels, topk=(1, 5))
            reduced_loss = loss.data

            # average loss and accuracy across processes for logging
            if args.distributed:
                reduced_loss, acc1, acc5 = reduce_tensors(
                    reduced_loss, acc1, acc5, world_size=args.world_size)

            # to_python_float incurs a host<->device sync
            losses.update(to_python_float(reduced_loss), images.size(0))
            top1.update(to_python_float(acc1), images.size(0))
            top5.update(to_python_float(acc5), images.size(0))
            lr = optimizer.param_groups[0]['lr']

            # measure elapsed time
            torch.cuda.synchronize()
            batch_time.update((time.time() - end) / args.print_freq)
            end = time.time()

            # This implies args.tensorboard and args.rank == 0:
            if writer:
                total_iter = iteration + len(loader) * epoch
                writer.add_scalar('train/loss', losses.val, total_iter)
                writer.add_scalar('train/acc1', top1.val, total_iter)
                writer.add_scalar('train/acc5', top5.val, total_iter)
                writer.add_scalar('train/lr', lr, total_iter)

            print(f'Epoch: [{epoch}][{iteration}/{len(loader)}]\t'
                  f'LR {lr:.6f}\t'
                  f'Time {batch_time.val:.2f} ({batch_time.avg:.2f})\t'
                  # f'Speed {args.world_size*args.batch_size/batch_time.val:.2f} '
                  # f'({args.world_size*args.batch_size/batch_time.avg:.2f})\t'
                  f'Loss {losses.val:.4f} ({losses.avg:.4f})\t'
                  f'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  f'Acc@5 {top5.val:.3f} ({top5.avg:.3f})')

        if args.profile >= 0: torch.cuda.nvtx.range_push("next(fetcher)")
        images, labels = next(fetcher)
        if args.profile >= 0: torch.cuda.nvtx.range_pop()

        if args.profile >= 0: torch.cuda.nvtx.range_pop()

        if args.profile >= 0 and iteration == args.profile + 100:
            print("Profiling ended at iteration {}".format(iteration))
            torch.cuda.cudart().cudaProfilerStop()
            quit()

    optimizer.step()
    optimizer.zero_grad()

    return top1.avg, top5.avg, losses.avg


@torch.no_grad()
def validate(loader, model, criterion, args, logger=None, writer=None):
    print('evaluating')
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    model.eval()
    end = time.time()

    iteration = 0
    # It is crucial to build a new fetcher at each epoch
    fetcher = data.DataFetcher(loader)
    images, targets = next(fetcher)

    while images is not None:
        iteration += 1

        output = model(images)
        loss = criterion(output, targets)

        # measure accuracy
        acc1, acc5 = accuracy(output.data, targets, topk=(1, 5))
        reduced_loss = loss.data

        # average loss and accuracy across processes for logging
        if args.distributed:
            reduced_loss, acc1, acc5 = reduce_tensors(
                reduced_loss, acc1, acc5, world_size=args.world_size)

        # to_python_float incurs a host<->device sync
        losses.update(to_python_float(reduced_loss), images.size(0))
        top1.update(to_python_float(acc1), images.size(0))
        top5.update(to_python_float(acc5), images.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # This impliies args.tensorboard and args.rank == 0:
        if writer:
            writer.add_scalar('validate/loss', losses.val, iteration)
            writer.add_scalar('validate/acc1', top1.val, iteration)
            writer.add_scalar('validate/acc5', top5.val, iteration)

        # TODO:  Change timings to mirror train().
        if iteration % args.print_freq == 0:
            print(f'Test: [{iteration}/{len(loader)}]\t'
                  f'Time {batch_time.val:.2f} ({batch_time.avg:.2f})\t'
                  # f'Speed {args.world_size * args.batch_size / batch_time.val:.3f} '
                  # f'({args.world_size * args.batch_size / batch_time.avg:.3f})\t'
                  f'Loss {losses.val:.4f} ({losses.avg:.4f})\t'
                  f'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  f'Acc@5 {top5.val:.3f} ({top5.avg:.3f})')

        images, targets = next(fetcher)

    print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'.format(top1=top1, top5=top5))

    return top1.avg, top5.avg, losses.avg


if __name__ == '__main__':
    global args
    args = parse()

    main(args)
