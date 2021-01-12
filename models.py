# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the CC-by-NC license found in the
# LICENSE file in the root directory of this source tree.
#
import torch
import torch.nn as nn
from functools import partial

from timm.models.vision_transformer import VisionTransformer, _cfg
from timm.models.registry import register_model


<<<<<<< HEAD
@register_model
def t16(pretrained=False, num_classes=1000, drop_path=0.0, **kwargs):
=======
    def forward(self, x):
        x = self.fc1(x)
        x = self.gelu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=16, attn_bias=True, attn_scaling=None,
                 attn_dropout=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scaling = attn_scaling or head_dim ** -0.5
        self.in_proj = nn.Linear(dim, dim * 3, bias=attn_bias)
        self.dropout = nn.Dropout(attn_dropout)
        self.out_proj = nn.Linear(dim, dim)

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
                 norm=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm(hidden_size)
        self.attention = Attention(
            hidden_size, num_heads=num_heads, attn_bias=attn_bias,
            attn_scaling=attn_scaling, attn_dropout=attn_dropout)
        self.norm2 = norm(hidden_size)
        mlp_dim = int(hidden_size * mlp_ratio)
        self.mlp = MLPBlock(in_channels=hidden_size, hidden_channels=mlp_dim, dropout=dropout)

    def forward(self, x):
        x = x + self.attention(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class Encoder(nn.Module):
    def __init__(self, img_size=384, patches=16, hidden_size=1024,
                 num_layers=12, num_heads=12, mlp_ratio=4, dropout=0.,
                 attn_bias=True, attn_scaling=None, attn_dropout=0.,
                 norm=nn.LayerNorm):
        super().__init__()
        num_patches = (img_size // patches) ** 2
        self.pos_embed = nn.Parameter(
            torch.zeros(1, num_patches + 1, hidden_size))
        self.dropout = nn.Dropout(dropout)
        self.blocks = nn.Sequential(
            *[Encoder1DBlock(
                hidden_size=hidden_size, num_heads=num_heads,
                mlp_ratio=mlp_ratio, dropout=dropout, attn_bias=attn_bias,
                attn_scaling=attn_scaling, attn_dropout=attn_dropout,
                norm=norm
            ) for _ in range(num_layers)]
        )
        self.norm = norm(hidden_size)
        self.apply(self._init)

    def _init(self, module):
        nn.init.normal_(self.pos_embed, std=0.02)

    def forward(self, x):
        x = x + self.pos_embed
        x = self.dropout(x)
        x = self.blocks(x)
        x = self.norm(x)
        return x[:, 0]


class VisionTransformer(nn.Module):
    def __init__(self, img_size=384, patches=16, num_classes=1000,
                 hidden_size=1024, num_layers=12, num_heads=12, mlp_ratio=4,
                 dropout=0., attn_bias=True, attn_scaling=None, attn_dropout=0.,
                 norm=nn.LayerNorm, pre_size=True, *args, **kwargs):
        super().__init__()
        self.num_classes = num_classes
        self.hidden_size = hidden_size

        self.embedding = nn.Conv2d(
            3, hidden_size, kernel_size=patches, stride=patches)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, hidden_size))
        self.encoder = Encoder(img_size, patches, hidden_size, num_layers,
            num_heads, mlp_ratio, dropout, attn_bias, attn_scaling,
            attn_dropout, norm)
        if pre_size:
            pre_size = pre_size if type(pre_size) is int else hidden_size
        self.pre_logits = nn.Linear(hidden_size, pre_size) if pre_size else nn.Identity()
        self.tanh = nn.Tanh() if pre_size else nn.Identity()
        self.head = nn.Linear(pre_size or hidden_size, num_classes)

        self.apply(self._init)

    def _init(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            nn.init.normal_(module.bias, std=1e-6)
        nn.init.zeros_(self.cls_token)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)
        # elif isinstance(module, nn.LayerNorm):
        #     nn.init.ones_(module.weight)
        #     nn.init.zeros_(module.bias)

    def forward(self, x):
        B = x.shape[0]
        x = self.embedding(x).flatten(2).transpose(1, 2)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = self.encoder(x)
        x = self.pre_logits(x)
        x = self.tanh(x)
        x = self.head(x)
        return x


def s16(pretrained=False, **kwargs):
>>>>>>> master
    model = VisionTransformer(
        patch_size=16, embed_dim=192, depth=12, num_heads=3, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), drop_path_rate=drop_path,
        num_classes=num_classes)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            url="https://dl.fbaipublicfiles.com/deit/deit_tiny_patch16_224-a1311bcf.pth",
            map_location="cpu", check_hash=True
        )
        model.load_state_dict(checkpoint["model"])
    return model

<<<<<<< HEAD
=======
def b16(pretrained=False, **kwargs):
    model = VisionTransformer(
        patches=16, hidden_size=768, num_layers=12,
        num_heads=12, norm=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model
>>>>>>> master

@register_model
def s16(pretrained=False, num_classes=1000, drop_path=0.0, **kwargs):
    model = VisionTransformer(
<<<<<<< HEAD
        patch_size=16, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), drop_path_rate=drop_path,
        num_classes=num_classes)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            url="https://dl.fbaipublicfiles.com/deit/deit_small_patch16_224-cd65a155.pth",
            map_location="cpu", check_hash=True
        )
        model.load_state_dict(checkpoint["model"])
    return model

=======
        patches=32, hidden_size=768, num_layers=12,
        num_heads=12, norm=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

def l16(pretrained=False, **kwargs):
    model = VisionTransformer(
        patches=16, hidden_size=1024, num_layers=24,
        num_heads=16, norm=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model
>>>>>>> master

@register_model
def b16(pretrained=False, num_classes=1000, drop_path=0.0, **kwargs):
    model = VisionTransformer(
<<<<<<< HEAD
        patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), drop_path_rate=drop_path,
        num_classes=num_classes)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            url="https://dl.fbaipublicfiles.com/deit/deit_base_patch16_224-b5f2ef4d.pth",
            map_location="cpu", check_hash=True
        )
        model.load_state_dict(checkpoint["model"])
=======
        patches=32, hidden_size=1024, num_layers=24,
        num_heads=16, norm=partial(nn.LayerNorm, eps=1e-6), **kwargs)
>>>>>>> master
    return model


@register_model
def l16(pretrained=False, num_classes=1000, drop_path=0.0, **kwargs):
    model = VisionTransformer(
        patch_size=16, embed_dim=1024, depth=24, num_heads=16, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), drop_path_rate=drop_path,
        num_classes=num_classes)
    model.default_cfg = _cfg()
    if pretrained:
        print('pretrained not available for this model')
    return model
