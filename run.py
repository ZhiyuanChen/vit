import sys
import os
import subprocess
import argparse

# import models

prof = 'nvprof --profile-child-processes --profile-from-start off -fo {}'

comm = 'GLOG_logtostderr=-1 GLOG_vmodule=MemcachedClient=-1 MC_COUNT_DISP=1000000 \
        OMPI_MCA_btl_smcuda_use_cuda_ipc=0 OMPI_MCA_mpi_warn_on_fork=0  \
        srun --mpi=pmi2 --job-name={} --partition={} -n {} --gres=gpu:{} --ntasks-per-node={}'

# model_names = sorted(name for name in models.__dict__
#                      if name.islower() and not name.startswith("__")
#                      and callable(models.__dict__[name]))
model_names = ['s16', 'b16', 'b32', 'l16', 'l32', 'h14', 'v50', 'b50']

backends = dict(
    pytorch=dict(
        pytorch=True,
        parrots=False,
        distributed=True,
        tensorboard=True,
        log=True,
        apex=True,
        slurm=True
    ),
    parrots=dict(
        pytorch=False,
        parrots=True,
        distributed=True,
        tensorboard=False,
        log=True,
        apex=False,
        slurm=True
    )
)

modes = dict(
    i21k=dict(
        train=True,
        tune=False,
        validate=True,
        pre_logits=True,
        arch='b16',
        train_data='/mnt/lustre/share_data/imagenet22k',
        val_data='/mnt/lustre/share_data/ImageNet-Pytorch/val',
        pin_memory=True,
        drop_last=True,
        num_classes=21843,
        img_size=224,
        batch_size=256,
        epochs=90,
        save_freq=10,
        optimizer='AdamW',
        lr=1e-3,
        lr_factor=512.0,
        strategy='linear',
        warmup_steps=5_000,
        weight_decay=0.03
    ),
    train=dict(
        arch='b16',
        train=True,
        tune=False,
        validate=True,
        pre_logits=True,
        train_data='/mnt/lustre/share_data/ImageNet-Pytorch/train',
        val_data='/mnt/lustre/share_data/ImageNet-Pytorch/val',
        num_classes=1000,
        img_size=224,
        batch_size=256,
        epochs=300,
        save_freq=10,
        optimizer='AdamW',
        lr=5e-4,
        weight_decay=0.05,
        lr_factor=512.0,
        strategy='cosine',
        warmup_steps=5_000,
    ),
    tune=dict(
        train=True,
        tune=True,
        validate=True,
        pre_logits=False,
        train_data='/mnt/lustre/share_data/ImageNet-Pytorch/train',
        val_data='/mnt/lustre/share_data/ImageNet-Pytorch/val',
        pin_memory=True,
        drop_last=True,
        num_classes=1000,
        img_size=384,
        batch_size=16,
        epochs=8,
        optimizer='SGD',
        lr=1e-2,
        lr_factor=512.0,
        strategy='cosine',
        warmup_steps=500,
        weight_decay=0.0
    ),
    validate=dict(
        train=False,
        tune=False,
        validate=True,
        pre_logits=True,
        val_data='/mnt/lustre/share_data/ImageNet-Pytorch/val',
        pin_memory=True,
        num_classes=1000,
        img_size=384,
        batch_size=16,
        epochs=1
    )
)


def set_gpu(gpus):
    gres_gpu = min(gpus, 8)
    ntasks_per_node = min(gpus, 8)
    return gpus, gres_gpu, ntasks_per_node


def parse():
    selector = argparse.ArgumentParser(description='Vision Transformer')
    plat = selector.add_mutually_exclusive_group()
    plat.add_argument('-pt', '--pytorch', action='store_true', help='pytorch backend')
    plat.add_argument('-pa', '--parrots', action='store_true', help='parrots backend')
    mode = selector.add_argument_group()
    mode.add_argument('-t', '--train', action='store_true', help='train model')
    mode.add_argument('-u', '--tune', action='store_true', help='tune model')
    mode.add_argument('-v', '--validate', action='store_true', help='validate model')

    parser = argparse.ArgumentParser(description='Vision Transformer')
    parser.add_argument('-id', '--experiment_id', type=str, default='8992',
                        help='id of experiment')

    mode = parser.add_argument_group()
    mode.add_argument('-d', '--distributed', action='store_true', help='distributed data parallel')

    log = parser.add_argument_group()
    log.add_argument('-tb', '--tensorboard', action='store_true',
                     help='use tensorboard')
    log.add_argument('-log', '--log', action='store_true', help='use logger')
    log.add_argument('-ed', '--experiment_dir', type=str, default='experiments',
                     help='directory of results')
    log.add_argument('-sd', '--save_dir', type=str, default='checkpoints',
                     help='directory of saved_checkpoints')
    log.add_argument('-pf', '--print_freq', type=int, default=100, metavar='N',
                     help='print frequency (default: 100)')
    log.add_argument('-sf', '--save_freq', type=int, default=1, metavar='N',
                     help='save frequency (default: 1)')

    profile = parser.add_argument_group()
    profile.add_argument('--profile', type=int, default=-1,
                         help='Run a few iterations for profiling.')
    profile.add_argument('--profile_dir', type=str, default='profile',
                         help='directory of profile files')
    profile.add_argument('--profile_name', type=str, default='%p.nvprof',
                         help='name of profile files')

    data = parser.add_argument_group()
    data.add_argument('-td', '--train_data', type=str, metavar='DIR', default=None,
                      help='path to train dataset')
    data.add_argument('-vd', '--val_data', metavar='DIR', default=None,
                      help='path to validation dataset')
    data.add_argument('-b', '--batch_size', type=int, metavar='N', default=64,
                      help='mini-batch size per process (default: 64)')
    data.add_argument('-dl', '--drop_last', action='store_true',
                      help='drop last element in data loader')
    data.add_argument('-pm', '--pin_memory', action='store_true')
    data.add_argument('-pw', '--persistent_workers', action='store_true')
    data.add_argument('-as', '--accum_steps', type=int, metavar='N', default=1,
                      help='gradient accumulation steps')

    model = parser.add_argument_group()
    model.add_argument('-a', '--arch', default='l16', choices=model_names, metavar='ARCH',
                       help='model architecture: ' + ' | '.join(model_names) + ' (default: l16)')
    model.add_argument('-pl', '--pre_logits', action='store_true',
                       help='Pre_logits')
    model.add_argument('-pt', '--pretrained', action='store_true',
                       help='use pre-trained model')
    model.add_argument('-c', '--checkpoint', type=str, metavar='PATH',
                       default=None, help='checkpoint to validate')
    model.add_argument('-r', '--resume', type=str, metavar='PATH', default=None,
                       help='path to latest checkpoint (default: None)')
    model.add_argument('-n', '--num_classes', type=int, metavar='N',
                       help='number of classes')
    model.add_argument('-s', '--img_size', type=int, metavar='N',
                       help='image size for model')
    model.add_argument('-do', '--dropout', type=float, metavar='M', default=0.0,
                       help='drop out rate')
    model.add_argument('-ado', '--attn_dropout', type=float, metavar='M', default=0.0,
                       help='drop out rate for attention')
    model.add_argument('-dp', '--drop_prob', type=float, default=0.1,
                       help='Stochastic depth (default: 0.1)')
    model.add_argument('--sync_bn', action='store_true',
                        help='enabling apex sync BN.')

    train = parser.add_argument_group()
    train.add_argument('-e', '--epochs', type=int, metavar='N',
                       help='number of total epochs to run')
    train.add_argument('-se', '--start_epoch', type=int, metavar='N', default=0,
                       help='manual epoch number (useful on restarts)')

    optimize = parser.add_argument_group()
    optimize.add_argument('-o', '--optimizer', type=str, metavar='OPTIMIZER',
                          default='AdamW', help='Optimizer (default: "AdamW"')
    optimize.add_argument('-l', '--lr', type=float, metavar='LR', default=1e-3,
                          help='base learning rate, scaled by total batch size / lr_factor')
    optimize.add_argument('-lrf', '--lr_factor', type=float, default=4096.0,
                          help='scale learning rate')
    optimize.add_argument('-flr', '--final_lr', type=float, metavar='LR', default=1e-5,
                          help='final learning rate, scaled by total batch size / lr_factor')
    optimize.add_argument('-mo', '--momentum', type=float, metavar='M', default=0.9,
                          help='momentum')
    optimize.add_argument('-wd', '--weight_decay', type=float, metavar='W', default=0.05,
                          help='weight decay (default: 0.05)')
    optimize.add_argument('-ls', '--strategy', type=str, default='linear',
                          help='learning rate scaling strategy')
    optimize.add_argument('-ws', '--warmup_steps', type=int, metavar='N', default=5000,
                          help='number of warm up steps to run')
    optimize.add_argument('-gc', '--gradient_clip', type=float, default=1.0,
                          help='gradient clip')
    optimize.add_argument('--deterministic', action='store_true')

    aug = parser.add_argument_group()
    aug.add_argument('-cj', '--color_jitter', type=float, default=0.4, metavar='PCT',
                     help='Color jitter factor (default: 0.4)')
    aug.add_argument('-aa', '--auto_augment', type=str, default='rand-m9-mstd0.5-inc1', metavar='NAME',
                     help='Use AutoAugment policy. "v0" or "original". " + " (default: rand-m9-mstd0.5-inc1)'),
    aug.add_argument('-sm', '--smoothing', type=float, default=0.1,
                     help='Label smoothing (default: 0.1)')
    aug.add_argument('-ti', '--train_interpolation', type=str, default='bicubic',
                     help='Training interpolation (random, bilinear, bicubic default: "bicubic")')
    aug.add_argument('-ra', '--repeated_aug', action='store_true',
                     help='use repeated augmentation')

    erase = parser.add_argument_group()
    erase.add_argument('-rep', '--random_erase_prob', type=float, default=0.25, metavar='PCT',
                       help='Random erase prob (default: 0.25)')
    erase.add_argument('-rem', '--random_erase_mode', type=str, default='pixel',
                       help='Random erase mode (default: "pixel")')
    erase.add_argument('-rec', '--random_erase_count', type=int, default=1,
                       help='Random erase count (default: 1)')
    erase.add_argument('-res', '--random_erase_split', action='store_true', default=False,
                       help='Do not random erase first (clean) augmentation split')

    mixup = parser.add_argument_group()
    mixup.add_argument('-mu', '--mixup', type=float, default=0.8,
                       help='mixup alpha, mixup enabled if > 0. (default: 0.8)')
    mixup.add_argument('-mum', '--mixup_mode', type=str, default='batch',
                       help='How to apply mixup/cutmix params. Per "batch", "pair", or "elem"')
    mixup.add_argument('-mup', '--mixup_prob', type=float, default=1.0,
                       help='Probability of performing mixup or cutmix when either/both is enabled')
    mixup.add_argument('-cm', '--cutmix', type=float, default=1.0,
                       help='cutmix alpha, cutmix enabled if > 0. (default: 1.0)')
    mixup.add_argument('--cutmix_minmax', type=float, nargs='+', default=None,
                       help='cutmix min/max ratio, overrides alpha and enables cutmix if set (default: None)')
    mixup.add_argument('-msp', '--mixup_switch_prob', type=float, default=0.5,
                       help='Probability of switching to cutmix when both mixup and cutmix enabled')

    apex = parser.add_argument_group()
    apex.add_argument('--apex', action='store_true')

    fp16 = parser.add_argument_group()
    fp16.add_argument('--opt_level', type=str, default='O1')
    fp16.add_argument('--keep_batchnorm_fp32', type=str, default=None)
    fp16.add_argument('--loss_scale', type=str, default=None)
    fp16.add_argument('--channels_last', action='store_true')

    slurm = parser.add_argument_group()
    slurm.add_argument('--slurm', action='store_true')
    slurm.add_argument('--port', type=str, default='29500')
    slurm.add_argument("--local_rank", type=int, default=0)
    slurm.add_argument('-j', '--job_name', type=str, default='ViT')
    slurm.add_argument('-p', '--partition', type=str, default='pat_prototype')
    slurm.add_argument('-x', '--exclude', nargs='+', default=None)
    slurm.add_argument('-g', '--gpus', type=int, default=32)
    slurm.add_argument('-w', '--num_workers', type=int, metavar='N', default=64,
                       help='number of data loading workers (default: 64)')

    selects, unknown = selector.parse_known_args(sys.argv[:3])
    backend = 'parrots' if selects.parrots else 'pytorch'
    mode = 'tune' if selects.tune else 'train' if selects.train else 'validate'
    platform = backends[backend]
    defaults = modes[mode]

    parser.set_defaults(**platform, **defaults)
    args, unkown = parser.parse_known_args(sys.argv[3:])

    return args


if __name__ == '__main__':
    args = parse()
    kwargs = vars(args)

    arguments = list()

    gpus, gres_gpu, ntasks_per_node = set_gpu(args.gpus)
    command = comm.format(args.job_name + args.experiment_id, args.partition,
                          gpus, gres_gpu, ntasks_per_node)

    arguments.append('--parrots' if args.parrots else '--pytorch')
    arguments.append('--tune' if args.tune else '--train' if args.train else '--validate')
    del kwargs['parrots']
    del kwargs['pytorch']
    del kwargs['train']
    del kwargs['tune']
    del kwargs['validate']

    for k, v in kwargs.items():
        if v is None:
            continue
        elif type(v) is bool:
            if v:
                arguments.append(f'--{k}')
        else:
            arguments.append(f'--{k} {v}')
    if args.profile >= 0:
        os.makedirs(args.profile_dir, exist_ok=True)
        command += ' ' + prof.format(
            os.path.join(args.profile_dir, args.profile_name))
    arguments = ' '.join(arguments)
    command += ' python -u main.py ' + arguments
    command = ' '.join(command.split())
    print(command)
    res = subprocess.run(command, shell=True)
    if res != 0:
        exit(1)
