import torch
import torch.nn.functional as F
from torch import nn
from torch import Tensor

from typing import Type, Any, Callable, Union, List, Optional

from .transformer import Encoder
from .resnet import ResNet, BasicBlock, Bottleneck, conv1x1


class ConvTokenizer(nn.Module):
    def __init__(self, L=16, CT=1024, C=1024, head=16, groups=16, input_channels=1024):
        super(ConvTokenizer, self).__init__()
        # Code for adjusting the channel sizes in case C is not equal to CT
        self.feature = nn.Conv2d(input_channels, C, kernel_size=1)
        self.conv_token_coef = nn.Conv2d(C, L, kernel_size=1)
        self.head = head
        self.conv_value = nn.Conv2d(C, C,kernel_size=1, groups=groups)
        self.C = C

    def forward(self, feature, token=0):
        N, C, H, W = feature.shape
        if C != self.C:
            feature = self.feature(feature)
        # compute token coefficients
        # feature: N, C, H, W, token: N, CT, L
        token_coef = self.conv_token_coef(feature)
        N, L, H, W = token_coef.shape
        token_coef = token_coef.view(N, 1, L, H * W)
        token_coef = token_coef.permute(0, 1, 3, 2)  # N, 1, HW, L
        token_coef = token_coef / torch.sqrt(torch.tensor(feature.shape[1], device=token_coef.device, dtype=torch.float))
        N, C, H, W = feature.shape
        token_coef = F.softmax(token_coef, dim=2)
        value = self.conv_value(feature).view(N, self.head, C // self.head, H * W)  # N, h, C//h, HW
        # extract token from the feature map
        # conv token: N, C, L. recurrent token: N, C, L_a
        token = torch.Tensor.matmul(value, token_coef).view(N, C, -1)
        token = token.view(N, L, C)
        return feature, token


class RecurrentTokenizer(nn.Module):
    def __init__(self, L=16, CT=1024, C=1024, head=16, groups=16, input_channels=1024):
        super(RecurrentTokenizer, self).__init__()
        # Code for adjusting the channel sizes in case C is not equal to CT
        self.feature = nn.Conv2d(input_channels, C, kernel_size=1)
        # use previous token to compute a query weight, which is
        # then used to compute token coefficients.
        self.conv_query = nn.Conv1d(CT, C, kernel_size=1)
        self.conv_key = nn.Conv2d(C, C, kernel_size=1, groups=groups)
        self.conv_value = nn.Conv2d(C, C, kernel_size=1, groups=groups)
        self.head = head
        self.C = C

    def forward(self, feature, token=0):
        N, C, H, W = feature.shape
        if C != self.C:
            feature = self.feature(feature)
        # compute token coefficients
        # feature: N, C, H, W, token: N, CT, L
        L = token.shape[2]
        # Split input token
        # T_a, T_b = token[:, :, :L // 2], token[:, :, L // 2:]
        query = self.conv_query(token)
        N, C, L_a = query.shape
        query = query.view(N, self.head, C // self.head, L_a)
        N, C, H, W = feature.shape
        key = self.conv_key(feature).view(N, self.head, C // self.head, H * W)  # N, h, C//h, HW
        # Compute token coefficients.
        # N, h, HW, L_a
        token_coef = torch.Tensor.matmul(key.permute(0, 1, 3, 2), query)
        token_coef = token_coef / torch.sqrt(torch.tensor(C / self.head, device=token_coef.device, dtype=torch.float))
        N, C, H, W = feature.shape
        token_coef = F.softmax(token_coef, dim=2)
        value = self.conv_value(feature).view(N, self.head, C // self.head,
                                              H * W)  # N, h, C//h, HW
        # extract token from the feature map
        # conv token: N, C, L. recurrent token: N, C, L_a
        token = torch.Tensor.matmul(value, token_coef).view(N, C, -1)
        token = token.view(N, L, C)
        return feature, token


class Projector(nn.Module):
    def __init__(self, CT, C, head=16, groups=16):
        super(Projector, self).__init__()
        self.proj_value_conv = nn.Conv1d(CT, C, kernel_size=1)
        self.proj_key_conv = nn.Conv1d(CT, C, kernel_size=1)
        self.proj_query_conv = nn.Conv2d(C, CT, kernel_size=1,groups=groups)
        self.head = head

    def forward(self, feature, token):
        N, L, CT = token.shape
        token = token.view(N, CT, L)
        h = self.head
        proj_v = self.proj_value_conv(token).view(N, h, -1, L)
        proj_k = self.proj_key_conv(token).view(N, h, -1, L)
        proj_q = self.proj_query_conv(feature)
        N, C, H, W = proj_q.shape
        proj_q = proj_q.view(N, h, C // h, H * W).permute(0, 1, 3, 2)
        proj_coef = F.softmax(torch.Tensor.matmul(proj_q, proj_k) / torch.sqrt(torch.tensor(C / h, device=proj_q.device, dtype=torch.float)), dim=3)
        proj = torch.Tensor.matmul(proj_v, proj_coef.permute(0, 1, 3, 2))
        _, _, H, W = feature.shape
        return feature + proj.view(N, -1, H, W), token


class VTModule(nn.Module):
    def __init__(self, tokenizer, L=16, CT=1024, C=1024, channel=1024, hidden_size=1024,
                 num_layers=5, num_heads=16, dropout=0.5, norm=nn.LayerNorm):
        super(VTModule, self).__init__()
        self.norm = nn.BatchNorm2d(channel)
        self.tokenizer = tokenizer(L=L, CT=CT, C=C)
        # self.encoder = Encoder(
        #     hidden_size=hidden_size, num_layers=num_layers, num_heads=num_heads,
        #     dropout=dropout, norm=norm)
        self.encoder = nn.Transformer(
            d_model=1024, nhead=16, num_encoder_layers=5, num_decoder_layers=0,
            dim_feedforward=1024, activation='relu', dropout=0.5)
        self.projector = Projector(CT=CT, C=C)

    def forward(self, x, token=0):
        if type(x) is tuple:
            x, token = x
        identity = x
        x = self.norm(x)
        x, token = self.tokenizer(x, token)
        # import pdb; pdb.set_trace()
        token = self.encoder(token, token)
        x, token = self.projector(x, token)
        x = x + identity
        return x, token


class VisualTransformer(ResNet):

    def __init__(
        self,
        block: Type[Union[BasicBlock, Bottleneck]],
        layers: List[int],
        num_classes: int = 1000,
        zero_init_residual: bool = False,
        groups: int = 1,
        width_per_group: int = 64,
        replace_stride_with_dilation: Optional[List[bool]] = None,
        res_norm: Optional[Callable[..., nn.Module]] = nn.BatchNorm2d,
        vt_norm: Optional[Callable[..., nn.Module]] = nn.LayerNorm,
        L=16,
        CT=1024,
        C=1024,
        channel=1024,
        hidden_size=1024,
        num_layers=5,
        num_heads=16,
        dropout=0.5
    ) -> None:
        super(VisualTransformer, self).__init__(
            block, layers, num_classes, zero_init_residual, groups,
            width_per_group, replace_stride_with_dilation, res_norm)

        self.layer4 = self._make_vt(layers[3], L, CT, C, channel, hidden_size, num_layers, num_heads, dropout, vt_norm)
        self.avgpool = nn.AdaptiveAvgPool2d((1024, 1))
        self.fc = nn.Linear(1024, num_classes)

    def _make_vt(self, blocks, L=16, CT=1024, C=1024, channel=1024, hidden_size=1024,
                 num_layers=5, num_heads=16, dropout=0.5, norm=nn.LayerNorm):
        layers = []
        layers.append(VTModule(ConvTokenizer, L, CT, C, channel, hidden_size, num_layers, num_heads, dropout, norm))
        for _ in range(1, blocks):
            layers.append(VTModule(RecurrentTokenizer, L, CT, C, channel, hidden_size, num_layers, num_heads, dropout, norm))

        return nn.Sequential(*layers)


    def forward(self, x: Tensor) -> Tensor:
        # See note [TorchScript super()]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x, token = self.layer4(x)
        token = self.avgpool(token)
        token = torch.flatten(token, 1)
        out = self.fc(token)

        return out


def v50(**kwargs):
    return VisualTransformer(Bottleneck, [3, 4, 6, 3])
