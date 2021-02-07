import torch
import torch.nn.functional as F
from torch import nn
from torch import Tensor

from typing import Type, Any, Callable, Union, List, Optional, Tuple

from .resnet import ResNet, Bottleneck, conv3x3, conv1x1
from .bottleneck_transformer import MHSA


class SBConv(nn.Module):
    def __init__(self, planes, branches=2, groups=32, reduce=16, stride=1, hidden=32, height=14, width=14):
        super(SBConv, self).__init__()
        hidden = max(planes // reduce, hidden)
        self.conv = nn.Sequential(
            conv3x3(planes, planes, stride=stride, groups=groups),
            nn.BatchNorm2d(planes),
            nn.ReLU(inplace=True))
        mhsa = MHSA(planes, height=height, width=width)
        self.attn = mhsa if stride != 2 else nn.Sequential(mhsa, nn.AvgPool2d(2, 2))
        self.convs = nn.ModuleList([self.conv, self.attn])
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Sequential(
            nn.Conv2d(planes, hidden, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.ReLU(inplace=True)
        )
        self.fcs = nn.ModuleList([
            nn.Conv2d(hidden, planes, kernel_size=1, stride=1)
            for i in range(branches)])
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


class SBUnit(Bottleneck):

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
        norm: Optional[Callable[..., nn.Module]] = nn.BatchNorm2d,
        img_size: Tuple[int, int] = (14, 14)
    ) -> None:
        height, width = img_size
        super(SBUnit, self).__init__(inplanes, planes, stride, downsample, groups, base_width, dilation, norm)
        planes = int(planes * (base_width / 64.)) * groups
        self.conv2 = SBConv(planes, stride=stride, groups=groups, height=height, width=width)


class SKBot(ResNet):

    def __init__(
        self,
        block: Type[SBUnit],
        layers: List[int],
        num_classes: int = 1000,
        zero_init_residual: bool = False,
        groups: int = 1,
        width_per_group: int = 64,
        replace_stride_with_dilation: Optional[List[bool]] = None,
        norm: Optional[Callable[..., nn.Module]] = nn.BatchNorm2d,
        img_size: int = 224,
    ) -> None:
        if type(img_size) is int:
            img_size = (img_size, img_size)
        self.img_size = img_size
        self.img_size = tuple(length // 4 for length in self.img_size)
        super(SKBot, self).__init__(block, layers, num_classes, zero_init_residual, groups, width_per_group,
                                    replace_stride_with_dilation, norm)

    def _make_layer(self, block: Type[SBUnit], planes: int, blocks: int,
                    stride: int = 1, dilate: bool = False) -> nn.Sequential:
        norm = self._norm
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm(planes * block.expansion),
            )
            self.img_size = tuple(length // stride for length in self.img_size)

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm, img_size=self.img_size))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm=norm, img_size=self.img_size))

        return nn.Sequential(*layers)


def sb50(**kwargs):
    return SKBot(SBUnit, [3, 4, 6, 3])
