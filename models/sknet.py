import torch
import torch.nn.functional as F
from torch import nn
from torch import Tensor

from typing import Type, Any, Callable, Union, List, Optional

from .resnet import ResNet, Bottleneck


class SKConv(nn.Module):
    def __init__(self, width, branches=2, groups=32, reduce=16, stride=1, len=32):
        super(SKConv, self).__init__()
        len = max(width // reduce, len)
        self.convs = nn.ModuleList([nn.Sequential(
            nn.Conv2d(width, width, kernel_size=3, stride=stride, padding=1+i, dilation=1+i,
                      groups=groups, bias=False),
            nn.BatchNorm2d(width),
            nn.ReLU(inplace=True)
        ) for i in range(branches)])
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Sequential(
            nn.Conv2d(width, len, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(len),
            nn.ReLU(inplace=True)
        )
        self.fcs = nn.ModuleList([])
        for i in range(branches):
            self.fcs.append(
                nn.Conv2d(len, width, kernel_size=1, stride=1)
            )
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = torch.stack([conv(x) for conv in self.convs], dim=1)
        attention = torch.sum(x, dim=1)
        attention = self.gap(attention)
        attention = self.fc(attention)
        attention = [fc(attention) for fc in self.fcs]
        attention = torch.stack(attention, dim=1)
        attention = self.softmax(attention)
        x = torch.sum(x * attention, dim=1)
        return x


class SKUnit(Bottleneck):

    expansion: int = 2

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm: Optional[Callable[..., nn.Module]] = nn.BatchNorm2d
    ) -> None:
        super(SKUnit, self).__init__(inplanes, planes, stride, downsample, groups, base_width, dilation, norm)
        width = int(planes * (base_width / 64.)) * groups
        self.conv2 = SKConv(width, stride=stride, groups=groups)


class SKNet(ResNet):

    def __init__(
        self,
        block: Type[SKUnit],
        layers: List[int],
        num_classes: int = 1000,
        zero_init_residual: bool = False,
        groups: int = 1,
        width_per_group: int = 64,
        replace_stride_with_dilation: Optional[List[bool]] = None,
        norm: Optional[Callable[..., nn.Module]] = nn.BatchNorm2d
    ) -> None:
        super(SKNet, self).__init__(block, layers, num_classes, zero_init_residual, groups, width_per_group,
                              replace_stride_with_dilation, norm)


def sk50(**kwargs):
    return SKNet(SKUnit, [3, 4, 6, 3])
