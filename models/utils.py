import torch
import torch.nn as nn


try:
    from torch.hub import load_state_dict_from_url
except ImportError:
    from torch.utils.model_zoo import load_url as load_state_dict_from_url


class DropModule(nn.Module):
    def __init__(self, drop_prob=0., epsilon=1e-7):
        self.drop_prob = drop_prob
        self.epsilon = epsilon

    def forward(self, x):
        if not self.training or self.drop_prob < self.epsilon:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        output = x.div(keep_prob) * random_tensor
        return output
