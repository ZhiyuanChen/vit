import torch
import torch.nn as nn

from functools import partial

from .transformer import Encoder


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
        self.encoder = Encoder(
            hidden_size, num_layers, num_heads, mlp_ratio, dropout, attn_bias,
            attn_scaling, attn_dropout, norm)
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
