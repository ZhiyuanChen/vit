import torch
import torch.nn as nn
from functools import partial

from torchvision.models.resnet import Bottleneck as ConvBlock, conv1x1


class DropModule(nn.Module):
    def __init__(self, drop_prob=0., epislon=1e-7):
        self.drop_prob = drop_prob
        self.epislon = epislon

    def forward(self, x):
        if not self.training or self.drop_prob < self.epislon:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        output = x.div(keep_prob) * random_tensor
        return output


class MLPBlock(nn.Module):
    def __init__(self, in_channels, hidden_channels=None, out_channels=None,
                 dropout=0.):
        super().__init__()
        hidden_channels = hidden_channels or in_channels
        out_channels = out_channels or in_channels
        self.fc1 = nn.Linear(in_channels, hidden_channels)
        self.gelu = nn.GELU()
        self.fc2 = nn.Linear(hidden_channels, out_channels)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.fc1(x)
        x = self.gelu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class Attention(nn.Module):
    def __init__(self, hidden_size, num_heads=16, attn_bias=True, attn_scaling=None,
                 attn_dropout=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = hidden_size // num_heads
        self.scaling = attn_scaling or head_dim ** -0.5
        self.in_proj = nn.Linear(hidden_size, hidden_size * 3, bias=attn_bias)
        self.dropout = nn.Dropout(attn_dropout)
        self.out_proj = nn.Linear(hidden_size, hidden_size)

    def forward(self, x):
        B, N, C = x.shape
        q, k, v = self.in_proj(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        attention = (q @ k.transpose(-2, -1)) * self.scaling
        attention = attention.softmax(dim=-1)
        attention = self.dropout(attention)
        x = (attention @ v).transpose(1, 2).reshape(B, N, C)
        x = self.out_proj(x)
        return x


class Encoder1DBlock(nn.Module):
    def __init__(self, hidden_size, num_heads, mlp_ratio=4, dropout=0.,
                 attn_bias=True, attn_scaling=None, attn_dropout=0.,
                 norm=nn.LayerNorm, drop_prob=0.):
        super().__init__()
        self.hidden_size = hidden_size
        self.head_dim = hidden_size // num_heads
        self.norm1 = norm(self.hidden_size)
        self.attention = Attention(
            self.hidden_size, num_heads=num_heads, attn_bias=attn_bias,
            attn_scaling=attn_scaling, attn_dropout=attn_dropout)
        self.norm2 = norm(self.hidden_size)
        # mlp_dim = int(hidden_size * mlp_ratio)
        # self.mlp = MLPBlock(in_channels=hidden_size, hidden_channels=mlp_dim, dropout=dropout)
        # downsample = nn.Sequential(
        #     conv1x1(self.hidden_size * 4, self.hidden_size),
        #    norm(self.hidden_size),
        # )
        # self.mlp = ConvBlock(hidden_size, hidden_size, downsample=downsample)
        self.mlp = ConvBlock(hidden_size, hidden_size // 4)
        self.drop_module = DropModule(drop_prob) if drop_prob > 0. else nn.Identity()

    def forward(self, x):
        residual = x
        x = self.norm1(x)
        x = self.attention(x)
        x = self.drop_module(x)
        x = x + residual
        y = self.norm2(x)
        f, c = y[:, :-1, :], y[:, -1, :]
        b, h, k = f.shape
        m = int(torch.sqrt(torch.tensor(h, dtype=torch.float, device=x.device)).item())
        f = f.reshape(b, m, m, k).permute(0, 3, 1, 2)
        f = self.mlp(f)
        f = f.permute(0, 2, 3, 1).reshape(b, h, k)
        y = torch.cat((f, c.unsqueeze(1)), dim=1)
        # y = self.mlp(y)
        y = self.drop_module(y)
        return x + y


class Encoder(nn.Module):
    def __init__(self, img_size=384, patches=16, hidden_size=1024,
                 num_layers=12, num_heads=12, mlp_ratio=4, dropout=0.,
                 attn_bias=True, attn_scaling=None, attn_dropout=0.,
                 norm=nn.LayerNorm, drop_prob=0.):
        super().__init__()
        drop_probs = [x.item() for x in torch.linspace(0, drop_prob, num_layers)]
        self.blocks = nn.ModuleList(
            [Encoder1DBlock(
                hidden_size=hidden_size, num_heads=num_heads,
                mlp_ratio=mlp_ratio, dropout=dropout, attn_bias=attn_bias,
                attn_scaling=attn_scaling, attn_dropout=attn_dropout,
                norm=norm, drop_prob=drop_probs[i]
            ) for i in range(num_layers)]
        )
        self.norm = norm(hidden_size)

    def forward(self, x):
        for block in self.blocks:
           x = block(x)
        x = self.norm(x)
        return x[:, 0]


class VisionTransformer(nn.Module):
    def __init__(self, img_size=384, patches=16, num_classes=1000,
                 hidden_size=1024, num_layers=12, num_heads=12, mlp_ratio=4,
                 dropout=0., attn_bias=True, attn_scaling=None, attn_dropout=0.,
                 norm=nn.LayerNorm, pre_logits=True, *args, **kwargs):
        super().__init__()
        self.num_classes = num_classes
        self.hidden_size = hidden_size

        self.embedding = nn.Conv2d(
            3, hidden_size, kernel_size=patches, stride=patches)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, hidden_size))
        num_patches = (img_size // patches) ** 2
        self.pos_embed = nn.Parameter(
            torch.zeros(1, num_patches + 1, hidden_size))
        self.dropout = nn.Dropout(dropout)
        self.encoder = Encoder(img_size, patches, hidden_size, num_layers,
            num_heads, mlp_ratio, dropout, attn_bias, attn_scaling,
            attn_dropout, norm)
        self.pre_logits = nn.Linear(hidden_size, hidden_size) if pre_logits else nn.Identity()
        self.tanh = nn.Tanh() if pre_logits else nn.Identity()
        self.head = nn.Linear(hidden_size, num_classes)

        self.apply(self._init)

    def _init(self, module):
        nn.init.normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.2)
        if isinstance(module, nn.Linear):
            nn.init.trunc_normal_(module.weight, std=.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.LayerNorm):
            nn.init.constant_(module.bias, 0)
            nn.init.constant_(module.weight, 1.0)

    def forward(self, x):
        B = x.shape[0]
        x = self.embedding(x).flatten(2).transpose(1, 2)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.dropout(x)
        x = self.encoder(x)
        x = self.pre_logits(x)
        x = self.tanh(x)
        x = self.head(x)
        return x


def s16(pretrained=False, **kwargs):
    model = VisionTransformer(
        patches=16, hidden_size=768, num_layers=8, num_heads=8, mlp_ratio=3,
        **kwargs)
    return model

def b16(pretrained=False, **kwargs):
    model = VisionTransformer(
        patches=16, hidden_size=768, num_layers=12,
        num_heads=12, norm=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

def b32(pretrained=False, **kwargs):
    model = VisionTransformer(
        patches=32, hidden_size=768, num_layers=12,
        num_heads=12, norm=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

def l16(pretrained=False, **kwargs):
    model = VisionTransformer(
        patches=16, hidden_size=1024, num_layers=24,
        num_heads=16, norm=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

def l32(pretrained=False, **kwargs):
    model = VisionTransformer(
        patches=32, hidden_size=1024, num_layers=24,
        num_heads=16, norm=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

def h14(pretrained=False, **kwargs):
    model = VisionTransformer(
        patches=14, hidden_size=1280, num_layers=32, num_heads=16, **kwargs)
    return model
