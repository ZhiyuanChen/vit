import os
import math
import shutil

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

    log("opt_level = {}".format(args.opt_level))
    log("keep_batchnorm_fp32 = {} {}".format(args.keep_batchnorm_fp32, type(args.keep_batchnorm_fp32)))
    log("loss_scale = {} {}".format(args.loss_scale, type(args.loss_scale)))
    log("\nCUDNN VERSION: {}\n".format(torch.backends.cudnn.version()))

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

    experiment, writer, save_dir = None, None, None
    if proc_id == 0:
        experiment = os.path.join(
            args.experiments, 
            f'{args.arch}-g{args.gpus}-b{args.batch_size}-d{args.dropout}' + \
            f'-gc{args.gradient_clip}-lr{args.learning_rate}-m{args.momentum}' + \
            f'-wd{args.weight_decay}-{args.strategy}{args.param}' + \
            f'-wu{args.warmup_steps}'
            )
        if args.tensorboard:
            tensorboard_dir = os.path.join(experiment, args.tensorboard_dir)
            os.makedirs(tensorboard_dir, exist_ok=True)
            writer = SummaryWriter(log_dir=tensorboard_dir)
        if args.train:
            save_dir = os.path.join(experiment, args.save_dir)
            os.makedirs(save_dir, exist_ok=True)
    return experiment, writer, save_dir


def resume(model, checkpoint):
    if os.path.isfile(checkpoint):
        log("=> loading checkpoint '{}'".format(checkpoint))
        checkpoint = torch.load(checkpoint, map_location = lambda storage, loc: storage.cuda(args.gpu))
        args.start_epoch = checkpoint['epoch']
        best_prec1 = checkpoint['best_prec1']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        log("=> loaded checkpoint '{}' (epoch {})"
                .format(checkpoint, checkpoint['epoch']))
    else:
        log("=> no checkpoint found at '{}'".format(checkpoint))


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


def save_checkpoint(state, is_best, folder, filename='checkpoint.pth', bestname=None):
    path = os.path.join(folder, filename)
    torch.save(state, path)
    if is_best:
        best = os.path.join(folder, bestname or f'{state["acc1"]}.pth')
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
        epochs,
        lr,
        strategy="cosine",
        param=295,
        warmup_steps=10_000,
        warmup_begin_lr=0.0,
        last_epoch=-1,
        min_lr=0
    ):
        if strategy not in ("constant", "cosine", "linear"):
            raise ValueError(
                "Only 'constant' or 'linear' warmup_method accepted"
                "got {}".format(strategy)
            )
        self.last_epoch = last_epoch
        self.epochs = epochs
        self.strategy = strategy
        self.param = param
        self.warmup_steps = warmup_steps
        self.warmup_begin_lr = warmup_begin_lr
        self.warmup_gamma = (lr - warmup_begin_lr) / warmup_steps if warmup_steps > 0 else 0
        self.min_lr = min_lr
        super(LRScheduler, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch == 0:
            return self.base_lrs
        elif self._step_count < self.warmup_steps:
            return self.warmup()
        else:
            return self._get_lr(self.strategy, self.param)

    def _get_lr(self, strategy, param):
        return getattr(self, strategy)(param)

    def warmup(self):
        return [group['lr'] + self.warmup_gamma
                for group in self.optimizer.param_groups]

    def linear(self, gamma):
        return [group['lr'] * gamma
                for group in self.optimizer.param_groups]

    def cosine(self, T_max):
        if (self.last_epoch - 1 - T_max) % (2 * T_max) == 0:
            return [group['lr'] + (base_lr - self.min_lr) *
                    (1 - math.cos(math.pi / T_max)) / 2
                    for base_lr, group in
                    zip(self.base_lrs, self.optimizer.param_groups)]
        return [(1 + math.cos(math.pi * self.last_epoch / T_max)) /
                (1 + math.cos(math.pi * (self.last_epoch - 1) / T_max)) *
                (group['lr'] - self.min_lr) + self.min_lr
                for group in self.optimizer.param_groups]

def adjust_learning_rate(lr, optimizer, epoch, warmup_steps, step, len_epoch):
    """LR schedule that should yield 76% converged accuracy with batch size 256"""
    factor = epoch // 30

    if epoch >= 80:
        factor = factor + 1

    lr = lr * (0.1**factor)

    """Warmup"""
    if epoch < warmup_steps:
        lr = lr * float(1 + step + epoch*len_epoch) / (5. * len_epoch)

    # if(args.local_rank == 0):
    #     print("epoch = {}, step = {}, lr = {}".format(epoch, step, lr))

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
