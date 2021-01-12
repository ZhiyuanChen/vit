import os
import shutil
import logging
import logging.config
import math

import numpy as np
from scipy.ndimage import zoom

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.distributed as dist

import subprocess

from functools import wraps


def catch(func, error=Exception):
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            result = func(*args, **kwargs)
            return result
        except error as e:
            print(error)

    return wrapper


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
        torch.distributed.init_process_group(backend='nccl')
        args.world_size = torch.distributed.get_world_size()

    if args.apex and not torch.backends.cudnn.enabled:
        raise RuntimeError('Amp requires cudnn backend to be enabled.')

    # proc_id is default to be 0 in case of not distributed
    global best_acc1, experiment, logger, writer, save_dir
    best_acc1, experiment, logger, writer, save_dir = 0, None, None, None, None
    name = f'{args.arch}-g{args.gpus}-b{args.batch_size}-e{args.epochs}' \
           f'-d{args.dropout}-gc{args.gradient_clip}-o{args.optimizer}' \
           f'-lr{args.lr}-m{args.momentum}-wd{args.weight_decay}' \
           f'-{args.strategy}-ws{args.warmup_steps}-as{args.accum_steps}' \
           f'-{args.opt_level}'
    experiment = os.path.join(args.experiment_dir, name.strip('/'))
    save_dir = os.path.join(experiment, args.save_dir)
    if proc_id == 0:
        os.makedirs(experiment, exist_ok=True)
        if args.tensorboard:
            from torch.utils.tensorboard import SummaryWriter
            writer = SummaryWriter(experiment)
        if args.log:
            logger = setup_logger(experiment)
        if args.train:
            os.makedirs(save_dir, exist_ok=True)

    setup_print(proc_id)
    return best_acc1, experiment, logger, writer, save_dir


def setup_print(proc_id):
    global logger
    import builtins as __builtin__
    builtin_print = __builtin__.print

    def print(*args, file=None, **kwargs):
        force = kwargs.pop('force', False)
        if proc_id == 0 or force:
            logger.info(*args, **kwargs) if logger else builtin_print(*args, file=file, **kwargs)

    __builtin__.print = print


def setup_logger(experiment):
    """Creates and returns a fancy logger."""
    # Why is setting up proper logging so !@?#! ugly?
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
                'filename': os.path.join(experiment, 'train.log'),
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


def load_checkpoint(model, optimizer, scheduler, args):
    if not os.path.isfile(args.checkpoint):
        raise FileNotFoundError('checkpoint ')
    print(f'=> loading checkpoint "{args.checkpoint}"')
    checkpoint = torch.load(args.checkpoint, map_location=lambda storage, loc: storage.cuda(args.gpu))
    args.start_epoch = checkpoint['epoch']
    best_acc1 = checkpoint['best_acc1']
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    scheduler.load_state_dict(checkpoint['scheduler'])
    print(f'=> loaded checkpoint "{args.checkpoint}" (epoch {checkpoint["epoch"]}')


def reduce_tensor(tensor, world_size):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.reduce_op.SUM)
    rt /= world_size
    return rt


def reduce_tensors(*tensors, world_size):
    return [reduce_tensor(tensor, world_size) for tensor in tensors]


def pos_embed_scale(pos_embed, img_size, patch_size, mode='constant', order=1):
    pos_embed_length_ckpt = pos_embed.shape[1]
    pos_embed_length_model = np.square(img_size) // np.square(patch_size) + 1
    if pos_embed_length_ckpt == pos_embed_length_model:
        return pos_embed
    print('The length of position embeding in checkpoint is: '
          f'{pos_embed_length_ckpt}, while in current model, it should be: '
          f'{pos_embed_length_model}. Performing {mode} interpolation')
    pos_embed_tok, pos_embed_grid = pos_embed[:, :1], pos_embed[0, 1:]
    grid_size_ckpt = int(np.sqrt(len(pos_embed_grid)))
    grid_size_model = int(np.sqrt(np.square(img_size) // np.square(patch_size)))
    zoom_factor = (grid_size_model/ grid_size_ckpt, grid_size_model/ grid_size_ckpt, 1)
    pos_embed_grid = pos_embed_grid.reshape(grid_size_ckpt, grid_size_ckpt, -1)
    # TODO use torch.interpolate for zoom
    pos_embed_grid = torch.from_numpy(zoom(pos_embed_grid, zoom_factor, mode=mode, order=order))
    pos_embed_grid = pos_embed_grid.reshape(1, grid_size_model * grid_size_model, -1)
    pos_embed = torch.cat((pos_embed_tok, pos_embed_grid), axis=1)
    return pos_embed


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
