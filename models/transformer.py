import torch
import torch.nn as nn

from functools import partial

from .utils import DropModule


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
                 norm=nn.LayerNorm, drop_prob=0.):
        super().__init__()
        self.norm1 = norm(hidden_size)
        self.attention = Attention(
            hidden_size, num_heads=num_heads, attn_bias=attn_bias,
            attn_scaling=attn_scaling, attn_dropout=attn_dropout)
        self.norm2 = norm(hidden_size)
        mlp_dim = int(hidden_size * mlp_ratio)
        self.mlp = MLPBlock(in_channels=hidden_size, hidden_channels=mlp_dim, dropout=dropout)
        self.drop_module = DropModule(drop_prob) if drop_prob > 0. else nn.Identity()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        identity = x
        x = self.norm1(x)
        x = self.attention(x)
        x = self.dropout(x)
        x = self.drop_module(x)
        x = x + identity
        y = self.norm2(x)
        y = self.mlp(y)
        y = self.drop_module(y)
        return x + y


class Encoder(nn.Module):
    def __init__(self, hidden_size=1024, num_layers=12, num_heads=12,
                 mlp_ratio=4, dropout=0., attn_bias=True, attn_scaling=None,
                 attn_dropout=0., norm=nn.LayerNorm, drop_prob=0.):
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
