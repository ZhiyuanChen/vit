import sys

from io import BytesIO

from PIL import Image

import torch
import torch.nn as nn
import torchvision.datasets as datasets

sys.path.append(r'/mnt/lustre/share/pymc/py3')
import mc

import numpy as np


class ImageFolder(datasets.ImageFolder):
    def __init__(self, *args, **kwargs):
        super().__init__(loader=self._loader, *args, **kwargs)
        self._init_memcached()

    def _init_memcached(self):
        server_list = '/mnt/lustre/share/memcached_client/server_list.conf'
        client = '/mnt/lustre/share/memcached_client/client.conf'
        self.client = mc.MemcachedClient.GetInstance(server_list, client)

    def _loader(self, path):
        value = mc.pyvector()
        self.client.Get(path, value)
        buffer = mc.ConvertBuffer(value)
        im = Image.open(BytesIO(buffer)).convert('RGB')
        return im


def to_python_float(t):
    if hasattr(t, 'item'):
        return t.item()
    else:
        return t[0]


class DataFetcher(object):
    def __init__(self, loader):
        self.iter = iter(loader)
        self.stream = torch.cuda.Stream()
        self.mean = torch.tensor([0.485 * 255, 0.456 * 255, 0.406 * 255]).cuda().view(1,3,1,1)
        self.std = torch.tensor([0.229 * 255, 0.224 * 255, 0.225 * 255]).cuda().view(1,3,1,1)
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

    def next(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        input = self.next_input
        target = self.next_target
        if input is not None:
            input.record_stream(torch.cuda.current_stream())
        if target is not None:
            target.record_stream(torch.cuda.current_stream())
        self.preload()
        return input, target


def fast_collate(batch, memory_format):
    imgs = [img[0] for img in batch]
    targets = torch.tensor([target[1] for target in batch], dtype=torch.int64)
    w = imgs[0].size[0]
    h = imgs[0].size[1]
    tensor = torch.zeros((len(imgs), 3, h, w), dtype=torch.uint8).contiguous(memory_format=memory_format)
    for i, img in enumerate(imgs):
        nump_array = np.array(img, dtype=np.uint8)
        if(nump_array.ndim < 3):
            nump_array = np.expand_dims(nump_array, axis=-1)
        nump_array = np.rollaxis(nump_array, 2)
        tensor[i] += torch.from_numpy(nump_array)
    return tensor, targets

def load_data(path, transform, batch_size, num_workers, memory_format, shuffle=None, distributed=True, profile=-1, collate_fn=None):
    dataset = ImageFolder(path, transform)

    sampler = torch.utils.data.distributed.DistributedSampler(dataset) if distributed else None

    shuffle = (sampler is None) if shuffle is not None else shuffle
    collate_fn = collate_fn if collate_fn is not None else lambda b: fast_collate(b, memory_format)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        sampler=sampler,
        collate_fn=collate_fn
    )

    return dataset, sampler, loader
