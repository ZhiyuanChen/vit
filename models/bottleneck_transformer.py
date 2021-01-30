import torch
from torch import Tensor
import torch.nn as nn
from typing import Type, Any, Callable, Union, List, Optional

from .resnet import conv1x1


class Attention(nn.Module):
    def __init__(self, planes, height=14, width=14, attn_bias=False):
        super(Attention, self).__init__()

        self.in_proj = nn.Conv2d(planes, planes * 3, kernel_size=1, bias=attn_bias)

        self.rel_h = nn.Parameter(torch.randn([1, planes, height, 1]))
        self.rel_w = nn.Parameter(torch.randn([1, planes, 1, width]))

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        b, c, h, w = x.shape
        q, k, v = self.in_proj(x).chunk(3, dim=1).view(b, c, -1)

        content_content = torch.bmm(q.permute(0, 2, 1), k)
        content_position = (self.rel_h + self.rel_w).view(1, c, -1).permute(0, 2, 1)
        content_position = torch.matmul(content_position, q)

        energy = content_content + content_position
        attention = self.softmax(energy)

        out = torch.bmm(v, attention.permute(0, 2, 1))
        out = out.view(b, c, h, w)

        return out


class MHSABlock(nn.Module):

    expansion: int = 4

    def __init__(
        self,
        inplanes: int,
        planes: int,
        height: int,
        width: int,
        downsample: Optional[nn.Module] = None,
        base_width: int = 64,
        attn_bias: bool = False,
        norm: Optional[Callable[..., nn.Module]] = nn.BatchNorm2d
    ) -> None:
        super(MHSABlock, self).__init__()
        planes = int(planes * (base_width / 64.))
        self.conv1 = conv1x1(inplanes, planes)
        self.bn1 = norm(planes)
        self.attention = Attention(planes, height, width, attn_bias)
        self.bn2 = norm(planes)
        self.conv3 = conv1x1(planes, planes * self.expansion)
        self.bn3 = norm(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.attention(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class BoTNet(nn.Module):

    def _make_mhsa(self, planes: int, blocks: int = 3, stride: int = 1,
                   height: int = 14, width: int = 14) -> nn.Sequential:
        norm = self._norm
        downsample = None

        if stride != 1:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * MHSABlock.expansion, stride),
                norm(planes * MHSABlock.expansion),
            )

        layer0 = MHSABlock(
            self.inplanes, planes, height=height, width=width, norm=norm,
            base_width=self.base_width, downsample=downsample)
        height //= stride
        width //= stride
        layers = [MHSABlock(
            self.inplanes, planes, height=height, width=width, norm=norm,
            base_width=self.base_width)
            for _ in range(1, blocks)]

        return nn.Sequential(layer0, *layers)
