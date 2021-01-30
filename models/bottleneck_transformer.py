import torch
import torch.nn as nn

from torch import Tensor, einsum
from einops import rearrange, repeat

from typing import Type, Any, Callable, Union, List, Optional

from .resnet import ResNet, Bottleneck, BasicBlock, conv1x1


def rel_to_abs(x):
    b, h, l, _ = x.shape
    col_pad = torch.zeros((b, h, l, 1), device=x.device, dtype=x.dtype)
    x = torch.cat((x, col_pad), dim=3)
    flat_x = rearrange(x, 'b h l c -> b h (l c)')
    flat_pad = torch.zeros((b, h, l - 1), dtype=x.dtype)
    flat_x_padded = torch.cat((flat_x, flat_pad), dim=2)
    final_x = flat_x_padded.reshape(b, h, l + 1, 2 * l - 1)
    final_x = final_x[:, :, :l, (l - 1):]
    return final_x


def relative_logits_1d(q, rel_k):
    b, heads, h, w, dim = q.shape
    logits = einsum('b h x y d, r d -> b h x y r', q, rel_k) * dim ** -0.5
    logits = rearrange(logits, 'b h x y r -> b (h x) y r')
    logits = rel_to_abs(logits)
    logits = logits.reshape(b, heads, h, w, w)
    logits = repeat(logits, 'b h x y j -> b h x i y j', i=h)
    return logits


class AbsPosEmb(nn.Module):
    def __init__(
        self,
        fmap_size,
        dim_head
    ):
        super().__init__()
        scale = dim_head ** -0.5
        self.scale = scale
        self.height = nn.Parameter(torch.randn(fmap_size, dim_head) * scale)
        self.width = nn.Parameter(torch.randn(fmap_size, dim_head) * scale)

    def forward(self, q):
        emb = rearrange(self.height, 'h d -> h () d') + rearrange(self.width, 'w d -> () w d')
        emb = rearrange(emb, ' h w d -> (h w) d')
        logits = einsum('b h i d, j d -> b h i j', q, emb) * self.scale
        return logits


class RelPosEmb(nn.Module):
    def __init__(
        self,
        fmap_size,
        dim_head
    ):
        super().__init__()
        scale = dim_head ** -0.5
        self.fmap_size = fmap_size
        self.scale = scale
        self.rel_height = nn.Parameter(torch.randn(fmap_size * 2 - 1, dim_head) * scale)
        self.rel_width = nn.Parameter(torch.randn(fmap_size * 2 - 1, dim_head) * scale)

    def forward(self, q):
        q = rearrange(q, 'b h (x y) d -> b h x y d', x=self.fmap_size)
        rel_logits_w = relative_logits_1d(q, self.rel_width)
        rel_logits_w = rearrange(rel_logits_w, 'b h x i y j-> b h (x y) (i j)')

        q = rearrange(q, 'b h x y d -> b h y x d')
        rel_logits_h = relative_logits_1d(q, self.rel_height)
        rel_logits_h = rearrange(rel_logits_h, 'b h x i y j -> b h (y x) (j i)')
        return rel_logits_w + rel_logits_h


class Attention(nn.Module):
    def __init__(self, planes, num_heads=4, height=14, width=14, attn_bias=False, rel_pos_emb=False, **kwargs):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        dim_head = planes // num_heads
        self.scale = torch.sqrt(torch.tensor(dim_head, dtype=torch.float))
        self.in_proj = nn.Conv2d(planes, planes * 3, kernel_size=1, bias=attn_bias)

        pos_emb = AbsPosEmb if not rel_pos_emb else RelPosEmb
        self.pos_emb = pos_emb(53, dim_head)

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        N, (B, C, H, W) = self.num_heads, x.shape
        q, k, v = self.in_proj(x).chunk(3, dim=1)
        q, k, v = map(lambda t: rearrange(t, 'b (h d) x y -> b h (x y) d', h=N), (q, k, v))
        sim = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale
        sim += self.pos_emb(q)
        attn = sim.softmax(dim=-1, dtype=q.dtype)
        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h (x y) d -> b (h d) x y', x=H, y=W)
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


class BoTNet(ResNet):

    def __init__(
        self,
        block: Type[Union[BasicBlock, Bottleneck, MHSABlock]],
        layers: List[int],
        num_classes: int = 1000,
        zero_init_residual: bool = False,
        groups: int = 1,
        width_per_group: int = 64,
        replace_stride_with_dilation: Optional[List[bool]] = None,
        norm: Optional[Callable[..., nn.Module]] = nn.BatchNorm2d
    ) -> None:
        super(BoTNet, self).__init__(block=block, layers=layers, num_classes=num_classes,
                                     zero_init_residual=zero_init_residual, groups=groups,
                                     width_per_group=width_per_group,
                                     replace_stride_with_dilation=replace_stride_with_dilation, norm=norm)
        self.layer4 = self._make_mhsa(layers[3], 512)

    def _make_mhsa(self, blocks: int, planes: int, stride: int = 1,
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
