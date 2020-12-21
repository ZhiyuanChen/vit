import os
import shutil
import logging
import logging.config
import math

import numpy as np

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.distributed as dist

from torch.utils.tensorboard import SummaryWriter

import subprocess


def log(string, proc_id=0):
    if int(os.environ['SLURM_PROCID']) == proc_id:
        print(string)


def init(args):
    proc_id = 0
    if args.slurm:
        proc_id = int(os.environ['SLURM_PROCID'])
        ntasks = int(os.environ['SLURM_NTASKS'])
        node_list = os.environ['SLURM_NODELIST']
        num_gpus = torch.cuda.device_count()
        addr = subprocess.getoutput(
            'scontrol show hostname {} | head -n1'.format(node_list))
        local_rank = proc_id % num_gpus
        os.environ['MASTER_PORT'] = args.port
        os.environ['MASTER_ADDR'] = addr
        os.environ['WORLD_SIZE'] = str(ntasks)
        os.environ['RANK'] = str(proc_id)
        os.environ['LOCAL_RANK'] = str(local_rank)
        args.local_rank = local_rank

    cudnn.benchmark = True
    if args.deterministic:
        cudnn.benchmark = False
        cudnn.deterministic = True
        torch.manual_seed(args.local_rank)
        torch.set_printoptions(precision=10)

    args.distributed = False
    if 'WORLD_SIZE' in os.environ:
        args.distributed = int(os.environ['WORLD_SIZE']) >= 1

    args.gpu = 0
    args.world_size = 1

    if args.distributed:
        args.gpu = args.local_rank
        torch.cuda.set_device(args.gpu)
        torch.distributed.init_process_group(backend='nccl',
                                             init_method='env://')
        args.world_size = torch.distributed.get_world_size()

    assert torch.backends.cudnn.enabled, "Amp requires cudnn backend to be enabled."

    experiment, logger, writer, save_dir = None, None, None, None
    if proc_id == 0:
        name = f'{args.arch}-g{args.gpus}-b{args.batch_size}-e{args.epochs}' \
               f'-d{args.dropout}-gc{args.gradient_clip}-lr{args.lr}' \
               f'-m{args.momentum}-wd{args.weight_decay}-{args.strategy}' \
               f'-ws{args.warmup_steps}-as{args.accum_steps}-{args.opt_level}'
        experiment = os.path.join(args.experiments, name.strip('/'))
        if args.tensorboard:
            os.makedirs(experiment, exist_ok=True)
            writer = SummaryWriter(log_dir=experiment)
        if args.log_dir:
            logger = setup_logger(args.logdir)
        if args.train:
            save_dir = os.path.join(experiment, args.save_dir)
            os.makedirs(save_dir, exist_ok=True)
    return experiment, logger, writer, save_dir


def setup_logger(log_dir):
    """Creates and returns a fancy logger."""
    # Why is setting up proper logging so !@?#! ugly?
    os.makedirs(log_dir, exist_ok=True)
    logging.config.dictConfig({
        'version': 1,
        'disable_existing_loggers': False,
        'formatters': {
            'standard': {
                'format': '%(asctime)s [%(levelname)s] %(name)s: %(message)s'
            },
        },
        'handlers': {
            'stderr': {
                'level': 'INFO',
                'formatter': 'standard',
                'class': 'logging.StreamHandler',
                'stream': 'ext://sys.stderr',
            },
            'logfile': {
                'level': 'DEBUG',
                'formatter': 'standard',
                'class': 'logging.FileHandler',
                'filename': os.path.join(log_dir, 'train.log'),
                'mode': 'a',
            }
        },
        'loggers': {
            '': {
                'handlers': ['stderr', 'logfile'],
                'level': 'DEBUG',
                'propagate': True
            },
        }
    })
    logger = logging.getLogger(__name__)
    logger.flush = lambda: [h.flush() for h in logger.handlers]
    return logger


def resume(model, optimizer, args):
    if os.path.isfile(args.checkpoint):
        log("=> loading checkpoint '{}'".format(args.checkpoint))
        checkpoint = torch.load(args.checkpoint, map_location=lambda storage, loc: storage.cuda(args.gpu))
        args.start_epoch = checkpoint['epoch']
        best_acc1 = checkpoint['best_acc1']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        log("=> loaded checkpoint '{}' (epoch {})"
                .format(checkpoint, checkpoint['epoch']))
    else:
        log("=> no checkpoint found at '{}'".format(args.checkpoint))


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def save_checkpoint(state, is_best, save_dir, save_name='checkpoint.pth', best_name=None):
    path = os.path.join(save_dir, save_name)
    torch.save(state, path)
    if is_best:
        best = os.path.join(save_dir, best_name or f'{state["acc1"]}.pth')
        shutil.copyfile(path, best)


def reduce_tensor(tensor, world_size):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.reduce_op.SUM)
    rt /= world_size
    return rt


def reduce_tensors(*tensors, world_size):
    return [reduce_tensor(tensor, world_size) for tensor in tensors]


class LRScheduler(torch.optim.lr_scheduler._LRScheduler):
    def __init__(
        self,
        optimizer,
        steps,
        final_lr,
        min_lr=1e-6,
        strategy="cosine",
        warmup_steps=10_000,
        accum_steps=1,
        last_epoch=-1,
    ):
        if strategy not in ("constant", "cosine", "linear"):
            raise ValueError(
                "Only 'constant' or 'linear' warmup_method accepted"
                "got {}".format(strategy)
            )
        self.steps = math.ceil(steps / accum_steps)
        self.final_lr = final_lr
        self.min_lr = min_lr
        self.strategy = strategy
        self.warmup_steps = math.ceil(warmup_steps / accum_steps)
        super(LRScheduler, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        progress = (self._step_count - self.warmup_steps) / float(self.steps - self.warmup_steps)
        progress = np.clip(progress, 0.0, 1.0)
        ratio = getattr(self, self.strategy)(progress)
        if self.warmup_steps:
            ratio = ratio * np.minimum(1., self._step_count / self.warmup_steps)
        return [max(self.min_lr, lr * ratio) for lr in self.base_lrs]

    def linear(self, progress):
        return self.final_lr + (1 - self.final_lr) * (1.0 - progress)

    def cosine(self, progress):
        return 0.5 * (1. + np.cos(np.pi * progress))
