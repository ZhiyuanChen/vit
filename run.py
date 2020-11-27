import argparse
import subprocess

import models


comm = 'GLOG_logtostderr=-1 GLOG_vmodule=MemcachedClient=-1 MC_COUNT_DISP=1000000 \
        OMPI_MCA_btl_smcuda_use_cuda_ipc=0 OMPI_MCA_mpi_warn_on_fork=0  \
        srun --mpi=pmi2 --job-name={} --partition={} -n {} --gres=gpu:{} --ntasks-per-node={} \
        python -u {}.py'

def set_gpu(gpus):
    if gpus < 8:
        gres_gpu = 1
        ntasks_per_node = 1
    else:
        gres_gpu = 8
        ntasks_per_node = 8
    return gpus, gres_gpu, ntasks_per_node


def parse():
    model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))
    parser = argparse.ArgumentParser(description='Vision Transformer')

    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('-t', '--train', action='store_true')
    mode.add_argument('-v', '--validate', action='store_true',
                      help='validate model on validation set')
    parser.add_argument('--profile', default=-1, type=int,
                        help='Run a few iterations for profiling.')

    parser.add_argument('-d', '--data', metavar='DIR', help='path to dataset',
                        default='/mnt/lustre/share_data/ImageNet-Pytorch')
    parser.add_argument('-a', '--arch', metavar='ARCH', default='l16',
                        choices=model_names,
                        help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: l16)')
    parser.add_argument('-pt', '--pretrained', action='store_true',
                        help='use pre-trained model')
    parser.add_argument('-c', '--checkpoint',
                        default='checkpoints/21kl16.pth',
                        type=str,
                        help='checkpoint to validate')
    parser.add_argument('-b', '--batch_size', default=12, type=int,
                        metavar='N', help='mini-batch size per process (default: 12)')
    parser.add_argument('-w', '--workers', default=2, type=int, metavar='N',
                        help='number of data loading workers (default: 2)')
    parser.add_argument('-pf', '--print_freq', default=20, type=int,
                        metavar='N', help='print frequency (default: 20)')

    parser.add_argument('-e', '--epochs', default=200, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('-se', '--start_epoch', default=0, type=int, metavar='N',
                        help='manual epoch number (useful on restarts)')
    parser.add_argument('-we', '--warmup_epoch', default=5, type=int, metavar='N',
                        help='number of warm up epochs to run')
    parser.add_argument('--lr', '--learning_rate', default=0.01, type=float,
                        metavar='LR', help='Initial learning rate.  Will be scaled by <global batch size>/256: args.lr = args.lr*float(args.batch_size*args.world_size)/256.  A warmup schedule will also be applied over the first 5 epochs.')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('-wd', '--weight_decay', default=1e-4, type=float,
                        metavar='W', help='weight decay (default: 1e-4)')
    parser.add_argument('-r', '--resume', default=None, type=str, metavar='PATH',
                        help='path to latest checkpoint (default: None)')
    parser.add_argument('--deterministic', action='store_true')

    parser.add_argument("--local_rank", default=0, type=int)
    parser.add_argument('--sync_bn', action='store_true',
                        help='enabling apex sync BN.')
    parser.add_argument('--opt_level', type=str, default='O0')
    parser.add_argument('--keep_batchnorm_fp32', type=str, default=None)
    parser.add_argument('--loss_scale', type=str, default=None)
    parser.add_argument('--channels_last', action='store_true')

    parser.add_argument('--slurm', action='store_true')
    parser.add_argument('--port', type=str, default='29500')
    parser.add_argument('-j', '--job_name', default='ViT', type=str)
    parser.add_argument('-p','--partition', default='pat_prototype', type=str)
    parser.add_argument('-x', '--exclude', nargs='+', default=None)
    parser.add_argument('-g', '--gpus', default=8, type=int)

    parser.set_defaults(train=True, slurm=True)
    args, unknown = parser.parse_known_args()
    return args


if __name__ == '__main__':
    args = parse()

    mode = 'train' if args.train else 'validate'
    arguments = list()

    gpus, gres_gpu, ntasks_per_node = set_gpu(args.gpus)
    command = comm.format(
        args.job_name, args.partition, gpus, gres_gpu, ntasks_per_node, mode)

    for k, v in vars(args).items():
        if v is None or k in ('train', 'validate'):
            continue
        elif type(v) is bool:
            if v:
                arguments.append(f'--{k}')
        else:
            arguments.append(f'--{k} {v}')
    arguments = ' ' + ' '.join(arguments)
    command += arguments
    command = ' '.join(command.split())
    print(command)
    res = subprocess.run(command, shell=True)
    if res != 0:
        exit(1)
