import argparse
import subprocess
import os

import models

prof = 'nvprof --profile-child-processes --profile-from-start off -fo {}'

comm = 'GLOG_logtostderr=-1 GLOG_vmodule=MemcachedClient=-1 MC_COUNT_DISP=1000000 \
        OMPI_MCA_btl_smcuda_use_cuda_ipc=0 OMPI_MCA_mpi_warn_on_fork=0  \
        srun --mpi=pmi2 --job-name={} --partition={} -n {} --gres=gpu:{} --ntasks-per-node={}'


def set_gpu(gpus):
    gres_gpu = min(gpus, 8)
    ntasks_per_node = min(gpus, 8)
    return gpus, gres_gpu, ntasks_per_node


def parse():
    model_names = sorted(name for name in models.__dict__
                         if name.islower() and not name.startswith("__")
                         and callable(models.__dict__[name]))
    parser = argparse.ArgumentParser(description='Vision Transformer')

    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('-t', '--train', action='store_true')
    mode.add_argument('-v', '--validate', action='store_true', help='validate model')

    # Log
    parser.add_argument('-tb', '--tensorboard', action='store_true',
                        help='use tensorboard')
    parser.add_argument('-log', '--log', action='store_true', help='use logger')
    parser.add_argument('-ed', '--experiment_dir', type=str, default='experiments',
                        help='directory of results')
    parser.add_argument('-pf', '--print_freq', default=100, type=int,
                        metavar='N', help='print frequency (default: 100)')
    parser.add_argument('-sf', '--save_freq', default=10, type=int,
                        metavar='N', help='save frequency (default: 10)')
    parser.add_argument('-sd', '--save_dir', type=str, default='checkpoints',
                        help='directory of saved_checkpoints')

    # Profile
    parser.add_argument('--profile', default=-1, type=int,
                        help='Run a few iterations for profiling.')
    parser.add_argument('--profile_dir', default='profile', type=str,
                        help='directory of profile files')
    parser.add_argument('--profile_name', default='%p.nvprof', type=str,
                        help='name of profile files')

    # Data
    parser.add_argument('-td', '--train_data',
                        default='/mnt/lustre/share_data/imagenet22k',
                        metavar='DIR', help='path to train dataset')
    parser.add_argument('-vd', '--val_data',
                        default='/mnt/lustre/share_data/ImageNet-Pytorch/val',
                        metavar='DIR', help='path to validation dataset')
    parser.add_argument('-a', '--arch', default='l16', choices=model_names,
                        metavar='ARCH', help='model architecture: ' + ' | '.join(model_names) + ' (default: l16)')

    # Model
    parser.add_argument('-pt', '--pretrained', action='store_true', help='use pre-trained model')
    parser.add_argument('-c', '--checkpoint',
                        default='/mnt/lustre/chenzhiyuan1/pyvit/checkpoints/21kl16.pth',
                        type=str,
                        help='checkpoint to validate')
    parser.add_argument('-r', '--resume', default=None, type=str, metavar='PATH',
                        help='path to latest checkpoint (default: None)')
    parser.add_argument('-n', '--num_classes', default=21843, type=int, metavar='N',
                        help='number of classes')
    parser.add_argument('-s', '--img_size', default=224, type=int, metavar='N',
                        help='image size to crop (default: 224)')
    parser.add_argument('-do', '--dropout', default=0.1, type=float, metavar='M',
                        help='drop out rate')
    parser.add_argument('-ado', '--attn_dropout', default=0.1, type=float, metavar='M',
                        help='drop out rate for attention')
    parser.add_argument('-dp', '--drop_path', default=0.1, type=float, metavar='M',
                        help='Drop path rate (default: 0.1)')
    parser.add_argument('-db', '--drop_block', default=None, type=float, metavar='M',
                        help='Drop block rate (default: None)')

    # Train
    parser.add_argument('-b', '--batch_size', default=64, type=int, metavar='N',
                        help='mini-batch size per process (default: 16)')
    parser.add_argument('-as', '--accum_steps', default=1, type=int, metavar='N',
                        help='gradient accumulation steps')
    parser.add_argument('-e', '--epochs', default=90, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('-se', '--start_epoch', default=0, type=int, metavar='N',
                        help='manual epoch number (useful on restarts)')

    # Optimize
    parser.add_argument('-o', '--optimizer', default='AdamW', type=str,
                        metavar='OPTIMIZER', help='Optimizer (default: "AdamW"')
    parser.add_argument('-l', '--lr', default=5e-4, type=float, metavar='LR',
                        help='base learning rate, scaled by total batch size / lr_factor')
    parser.add_argument('-lrf', '--lr_factor', default=4096.0, type=float,
                        help='scale learning rate')
    parser.add_argument('-flr', '--final_lr', default=1e-5, type=float, metavar='LR',
                        help='final learning rate, scaled by total batch size / lr_factor')
    parser.add_argument('-m', '--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('-wd', '--weight_decay', default=0.0001, type=float, metavar='W',
                        help='weight decay (default: 0.0001)')
    parser.add_argument('-ls', '--strategy', default='linear', type=str,
                        help='learning rate scaling strategy')
    parser.add_argument('-ws', '--warmup_steps', default=5000, type=int, metavar='N',
                        help='number of warm up epochs to run')
    parser.add_argument('-gc', '--gradient_clip', default=1.0, type=float, help='gradient clip')
    parser.add_argument('--deterministic', action='store_true')

    # Augmentation
    parser.add_argument('-cj', '--color_jitter', type=float, default=0.4, metavar='PCT',
                        help='Color jitter factor (default: 0.4)')
    parser.add_argument('-aa', '--auto_augment', type=str, default='rand-m9-mstd0.5-inc1', metavar='NAME',
                        help='Use AutoAugment policy. "v0" or "original". " + \
                             "(default: rand-m9-mstd0.5-inc1)'),
    parser.add_argument('-sm', '--smoothing', type=float, default=0.1,
                        help='Label smoothing (default: 0.1)')
    parser.add_argument('-ti', '--train_interpolation', type=str, default='bicubic',
                        help='Training interpolation (random, bilinear, bicubic default: "bicubic")')
    parser.add_argument('-ra', '--repeated_aug', action='store_true',
                        help='use repeated augmentation')

    # Random Erase
    parser.add_argument('-rep', '--random_erase_prob', type=float, default=0.25, metavar='PCT',
                        help='Random erase prob (default: 0.25)')
    parser.add_argument('-rem', '--random_erase_mode', type=str, default='pixel',
                        help='Random erase mode (default: "pixel")')
    parser.add_argument('-rec', '--random_erase_count', type=int, default=1,
                        help='Random erase count (default: 1)')
    parser.add_argument('-res', '--random_erase_split', action='store_true', default=False,
                        help='Do not random erase first (clean) augmentation split')

    # Mixup
    parser.add_argument('-mu', '--mixup', type=float, default=0.8,
                        help='mixup alpha, mixup enabled if > 0. (default: 0.8)')
    parser.add_argument('-mum', '--mixup_mode', type=str, default='batch',
                        help='How to apply mixup/cutmix params. Per "batch", "pair", or "elem"')
    parser.add_argument('-mup', '--mixup_prob', type=float, default=1.0,
                        help='Probability of performing mixup or cutmix when either/both is enabled')
    parser.add_argument('-cm', '--cutmix', type=float, default=1.0,
                        help='cutmix alpha, cutmix enabled if > 0. (default: 1.0)')
    parser.add_argument('--cutmix_minmax', type=float, nargs='+', default=None,
                        help='cutmix min/max ratio, overrides alpha and enables cutmix if set (default: None)')
    parser.add_argument('-msp', '--mixup_switch_prob', type=float, default=0.5,
                        help='Probability of switching to cutmix when both mixup and cutmix enabled')

    parser.add_argument("--local_rank", default=0, type=int)
    parser.add_argument('--sync_bn',
                        action='store_true',
                        help='enabling apex sync BN.')
    parser.add_argument('--opt_level', type=str, default='O1')
    parser.add_argument('--keep_batchnorm_fp32', type=str, default=None)
    parser.add_argument('--loss_scale', type=str, default=None)
    parser.add_argument('--channels_last', action='store_true')

    # Slurm
    parser.add_argument('--slurm', action='store_true')
    parser.add_argument('--port', type=str, default='29500')
    parser.add_argument('-j', '--job_name', default='ViT', type=str)
    parser.add_argument('-p', '--partition', default='pat_prototype', type=str)
    parser.add_argument('-x', '--exclude', nargs='+', default=None)
    parser.add_argument('-g', '--gpus', default=32, type=int)
    parser.add_argument('-w', '--workers', default=64, type=int, metavar='N',
                        help='number of data loading workers (default: 64)')

    parser.set_defaults(train=True, repeated_aug=True, tensorboard=True, logger=True, slurm=True)
    args, unknown = parser.parse_known_args()
    return args


if __name__ == '__main__':
    args = parse()

    if (not args.train) and (not args.validate):
        args.train = True
    mode = 'train' if args.train else 'validate'

    arguments = list()

    gpus, gres_gpu, ntasks_per_node = set_gpu(args.gpus)
    command = comm.format(args.job_name, args.partition, gpus, gres_gpu,
                          ntasks_per_node)

    for k, v in vars(args).items():
        if v is None or k in ('train', 'validate'):
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
    command += f' python -u {mode}.py ' + arguments
    command = ' '.join(command.split())
    print(command)
    res = subprocess.run(command, shell=True)
    if res != 0:
        exit(1)
