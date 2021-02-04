import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Type, Any, Callable, Union, List, Optional, Tuple

from .resnet import ResNet, Bottleneck, conv1x1


class MHSA(nn.Module):
    def __init__(self, planes, width=14, height=14):
        super(MHSA, self).__init__()

        self.query = nn.Conv2d(planes, planes, kernel_size=1)
        self.key = nn.Conv2d(planes, planes, kernel_size=1)
        self.value = nn.Conv2d(planes, planes, kernel_size=1)

        self.rel_h = nn.Parameter(torch.randn([1, planes, 1, height]), requires_grad=True)
        self.rel_w = nn.Parameter(torch.randn([1, planes, width, 1]), requires_grad=True)

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        n_batch, C, width, height = x.size()
        q = self.query(x).view(n_batch, C, -1)
        k = self.key(x).view(n_batch, C, -1)
        v = self.value(x).view(n_batch, C, -1)

        content_content = torch.bmm(q.permute(0, 2, 1), k)

        content_position = (self.rel_h + self.rel_w).view(1, C, -1).permute(0, 2, 1)
        content_position = torch.matmul(content_position, q)

        energy = content_content + content_position
        attention = self.softmax(energy)

        out = torch.bmm(v, attention.permute(0, 2, 1))
        out = out.view(n_batch, C, width, height)

        return out


class MHSABlock(Bottleneck):

    expansion = 4

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
        super(MHSABlock, self).__init__(inplanes, planes, stride, downsample, groups, base_width, dilation, norm)
        height, width = img_size
        mhsa = MHSA(planes, width=width, height=height)
        self.conv2 = mhsa if stride != 2 else nn.Sequential(mhsa, nn.AvgPool2d(2, 2))


class BoTNet(ResNet):

    def __init__(
        self,
        block: Type[Bottleneck],
        layers: List[int],
        num_classes: int = 1000,
        zero_init_residual: bool = False,
        groups: int = 1,
        width_per_group: int = 64,
        replace_stride_with_dilation: Optional[List[bool]] = None,
        norm: Optional[Callable[..., nn.Module]] = nn.BatchNorm2d,
        img_size: int = 224,
        **kwargs
    ) -> None:
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        super(BoTNet, self).__init__(block, layers, num_classes, zero_init_residual, groups, width_per_group,
                                     replace_stride_with_dilation, norm)
        if type(img_size) is int:
            img_size = (img_size, img_size)
        self.img_size = img_size
        self.img_size = tuple(img_size // 16 for img_size in self.img_size)
        self.inplanes //= 2
        self.layer4 = self._make_mhsa_layer(512, layers[3], stride=2, dilate=replace_stride_with_dilation[2])

    def _make_mhsa_layer(self, planes: int, blocks: int, stride: int = 2,
                         dilate: bool = False) -> nn.Sequential:
        norm = self._norm
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * MHSABlock.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * MHSABlock.expansion, stride),
                norm(planes * MHSABlock.expansion),
            )

        layers = [MHSABlock(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm)]
        self.inplanes = planes * MHSABlock.expansion
        for _ in range(1, blocks):
            layers.append(MHSABlock(self.inplanes, planes, groups=self.groups,
                                    base_width=self.base_width, dilation=self.dilation,
                                    norm=norm))

        return nn.Sequential(*layers)


def b50(**kwargs):
    return BoTNet(Bottleneck, [3, 4, 6, 3], **kwargs)
