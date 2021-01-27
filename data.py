import sys
import os
import random
import math

from io import BytesIO
from functools import partial

import torch
import torch.nn as nn
import torch.distributed as dist
import torchvision.datasets as datasets

from PIL import Image
import numpy as np

sys.path.append(r'/mnt/lustre/share/pymc/py3')
import mc

from utils import catch

import warnings

warnings.filterwarnings("ignore", "(Possibly )?corrupt EXIF data", UserWarning)


class ImageFolder(datasets.ImageFolder):
    def __init__(self, *args, **kwargs):
        super().__init__(loader=self._loader, *args, **kwargs)

    def _init_memcached(self):
        server_list = '/mnt/lustre/share/memcached_client/server_list.conf'
        client = '/mnt/lustre/share/memcached_client/client.conf'
        self.client = mc.MemcachedClient.GetInstance(server_list, client)

    def _loader(self, path):
        self._init_memcached()
        try:
            value = mc.pyvector()
            self.client.Get(path, value)
            buffer = mc.ConvertBuffer(value)
            im = Image.open(BytesIO(buffer)).convert('RGB')
        except Exception as e:
            folder = '/'.join(path.split('/')[:-1])
            image = random.choice(os.listdir(folder))
            path = os.path.join(folder, image)
            im = self._loader(path)
        return im


class DataFetcher(object):
    def __init__(self, loader):
        self.iter = iter(loader)
        self.stream = torch.cuda.Stream()
        self.mean = torch.tensor([0.485 * 255, 0.456 * 255, 0.406 * 255]).cuda().view(1, 3, 1, 1)
        self.std = torch.tensor([0.229 * 255, 0.224 * 255, 0.225 * 255]).cuda().view(1, 3, 1, 1)
        self.preload()

    def preload(self):
        try:
            self.next_input, self.next_target = next(self.iter)
        except StopIteration:
            self.next_input = None
            self.next_target = None
            return
        with torch.cuda.stream(self.stream):
            self.next_input = self.next_input.cuda(non_blocking=True)
            self.next_target = self.next_target.cuda(non_blocking=True)
            self.next_input = self.next_input.float()
            # normalise
            self.next_input = self.next_input.sub_(127.5).div_(127.5)
            # self.next_input = self.next_input.sub_(self.mean).div_(self.std)

    def __iter__(self):
        return self

    def __next__(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        input = self.next_input
        target = self.next_target
        if input is not None:
            input.record_stream(torch.cuda.current_stream())
        if target is not None:
            target.record_stream(torch.cuda.current_stream())
        self.preload()
        return input, target


class RepeatedAugmentSampler(torch.utils.data.Sampler):
    """Sampler that restricts data loading to a subset of the dataset for distributed,
    with repeated augmentation.
    It ensures that different each augmented version of a sample will be visible to a
    different process (GPU)
    Heavily based on torch.utils.data.DistributedSampler
    """

    def __init__(self, dataset, num_replicas=None, rank=None, shuffle=True):
        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = dist.get_rank()
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.num_samples = int(math.ceil(len(self.dataset) * 3.0 / self.num_replicas))
        self.total_size = self.num_samples * self.num_replicas
        # self.num_selected_samples = int(math.ceil(len(self.dataset) / self.num_replicas))
        self.num_selected_samples = int(math.floor(len(self.dataset) // 256 * 256 / self.num_replicas))
        self.shuffle = shuffle

    def __iter__(self):
        # deterministically shuffle based on epoch
        g = torch.Generator()
        g.manual_seed(self.epoch)
        if self.shuffle:
            indices = torch.randperm(len(self.dataset), generator=g).tolist()
        else:
            indices = list(range(len(self.dataset)))

        # add extra samples to make it evenly divisible
        indices = [ele for ele in indices for i in range(3)]
        indices += indices[:(self.total_size - len(indices))]
        assert len(indices) == self.total_size

        # subsample
        indices = indices[self.rank:self.total_size:self.num_replicas]
        assert len(indices) == self.num_samples

        return iter(indices[:self.num_selected_samples])

    def __len__(self):
        return self.num_selected_samples

    def set_epoch(self, epoch):
        self.epoch = epoch


def fast_collate(batch, memory_format):
    imgs = [img[0] for img in batch]
    targets = torch.tensor([target[1] for target in batch], dtype=torch.int64)
    w, h = imgs[0].size
    tensor = torch.zeros((len(imgs), 3, h, w), dtype=torch.uint8).contiguous(memory_format=memory_format)
    for i, img in enumerate(imgs):
        nump_array = np.array(img, dtype=np.uint8)
        if nump_array.ndim < 3:
            nump_array = np.expand_dims(nump_array, axis=-1)
        nump_array = np.rollaxis(nump_array, 2)
        tensor[i] += torch.from_numpy(nump_array)
    return tensor, targets


def load_data(path, transform, memory_format, batch_size, num_workers,
              shuffle=True, collate_fn=None, worker_init_fn=None,
              repeated_aug=False, distributed=True, pin_memory=True,
              drop_last=False, persistent_workers=False, **kwargs):
    dataset = ImageFolder(path, transform)

    sampler = RepeatedAugmentSampler(dataset) if repeated_aug else \
        torch.utils.data.distributed.DistributedSampler(dataset) if distributed else None
    shuffle = (sampler is None) if shuffle else shuffle
    collate_fn = collate_fn if collate_fn is not None else partial(fast_collate, memory_format=memory_format)

    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler,
        batch_sampler=None,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=pin_memory,
        drop_last=drop_last,
        worker_init_fn=worker_init_fn,
        persistent_workers=persistent_workers
    )

    return dataset, sampler, loader
