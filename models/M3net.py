import math

import torch
import torch.nn as nn

from einops import rearrange, repeat
import torch.nn.functional as F


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Module):
    def __init__(self, embed_dim, mlp_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim , mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, embed_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


# class MultiHeadAttention(nn.Module):
#     def __init__(self, embed_dim, num_heads, dropout):
#         super().__init__()
#         self.embed_dim = embed_dim
#         self.num_heads = num_heads
#         self.head_dim = embed_dim // num_heads
#         assert (
#             self.head_dim * num_heads == embed_dim
#         ), "Embedding dimension needs to be divisible by number of heads."
#
#         self.scale = self.head_dim ** -0.5
#
#         self.qkv = nn.Linear(embed_dim, embed_dim * 3, bias=False)
#         self.attn_drop = nn.Dropout(dropout)
#         self.proj = nn.Sequential(
#             nn.Linear(embed_dim, embed_dim),
#             nn.Dropout(dropout)
#         )
#
#     def forward(self, x):
#         B, N, _ = x.shape
#         qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
#         q, k, v = qkv[0], qkv[1], qkv[2]  # each has shape [B, H, N, D]
#
#         attn = (q @ k.transpose(-2, -1)) * self.scale  # [B, H, N, N]
#         attn = attn.softmax(dim=-1)
#         attn = self.attn_drop(attn)
#
#         out = (attn @ v).transpose(1, 2).reshape(B, N, self.embed_dim)  # [B, N, E]
#         out = self.proj(out)
#         return out


class MultiHeadConvAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert (
            self.head_dim * num_heads == embed_dim
        ), "Embedding dimension needs to be divisible by number of heads."

        self.x_proj = nn.Linear(embed_dim, embed_dim)

        self.spatial_attn = nn.Sequential(
            nn.Conv2d(num_heads, num_heads, kernel_size=1),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        self.spectral_attn = nn.Sequential(
            nn.Linear(self.head_dim, self.head_dim//2),
            nn.GELU(),
            nn.Linear(self.head_dim//2, self.head_dim),
            nn.Dropout(dropout)
        )
        self.out_proj = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, y=None):
        B, N, _ = x.shape
        x = (self.x_proj(x)).reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        avg_pool = x.mean(dim=-1, keepdim=True)
        std_pool = x.std(dim=-1, keepdim=True)
        spatial_pool = torch.cat((avg_pool, std_pool), dim=-1)

        spa_avg = x.mean(dim=-2, keepdim=True)  # [B, H, 1, D]
        spa_std = x.std(dim=-2, keepdim=True)  # [B, H, 1, D]
        spectral_pool = torch.cat((spa_avg, spa_std), dim=-2)

        spatial_attn = F.sigmoid(self.spatial_attn(spatial_pool).sum(dim=-1))  # [B, H, N]
        spectral_attn = F.sigmoid(self.spectral_attn(spectral_pool).sum(dim=-2))  # [B, H, D]
        out = (x * spatial_attn.unsqueeze(-1) * spectral_attn.unsqueeze(-2)).reshape(B, N, self.embed_dim)

        out = self.out_proj(out)
        return out


class SpectralInception(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        # 多分支卷积设计
        self.branch1 = nn.Sequential(
            nn.Conv1d(in_channels, math.ceil(in_channels / 3), kernel_size=1),
            nn.Dropout(0.1),
            nn.GELU()
        )
        self.branch2 = nn.Sequential(
            nn.Conv1d(in_channels, math.ceil(in_channels / 3), kernel_size=3, padding=1),
            nn.Dropout(0.1),
            nn.GELU()
        )
        self.branch3 = nn.Sequential(
            nn.Conv1d(in_channels, math.ceil(in_channels / 3), kernel_size=5, padding=2),
            nn.Dropout(0.1),
            nn.GELU()
        )

    def forward(self, x):
        # 并行多尺度特征提取
        return torch.cat([
            self.branch1(x),
            self.branch2(x),
            self.branch3(x)
        ], dim=1)


class ConvAttentionTransformer(nn.Module):
    def __init__(self, num, embed_dim, depth, heads, mlp_dim, dropout):
        super().__init__()

        self.depth = depth
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(embed_dim, MultiHeadConvAttention(embed_dim, num_heads=heads, dropout=dropout)),
                Residual(PreNorm(embed_dim, FeedForward(embed_dim, mlp_dim, dropout=dropout)))
            ]))

        self.SI = nn.Sequential(
            SpectralInception(num),
            nn.Conv1d(math.ceil(num / 3)*3, num, kernel_size=1),  # 通道数恢复
            nn.GELU()
        )
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        for i, (attn, ff) in enumerate(self.layers):
            x = attn(x)
            x = self.norm(self.SI(x))
            x = ff(x)
        return x

class MultiScaleTokenEmbedding(nn.Module):
    def __init__(self, patch_size, in_channels, embed_dim):
        super().__init__()
        self.patch_size = patch_size

        # 空间重组分支
        self.spatial_branch = nn.Sequential(
            nn.Linear(in_channels * patch_size*2, embed_dim * patch_size),
            nn.GELU(),
            nn.Dropout(0.1)
        )

        # 2D深度可分离卷积分支
        self.conv2_branch = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, groups=in_channels),
            nn.Conv2d(in_channels, embed_dim, kernel_size=1),
            nn.GELU(),
            nn.Dropout(0.1)
        )

        # 1D局部光谱卷积分支
        self.conv1_branch = nn.Sequential(
            nn.Conv1d(patch_size * patch_size, patch_size * patch_size, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Linear(in_channels, embed_dim),
            nn.Dropout(0.1)
        )

    def forward(self, x):
        # 分支1
        b, c, _ = x.shape
        x_x = rearrange(x, 'b c (h w) -> b w (c h)', h=self.patch_size)
        x_y = rearrange(x, 'b c (h w) -> b h (c w)', w=self.patch_size)
        x_spatial = torch.cat((x_x, x_y), dim=-1)
        x_spatial = self.spatial_branch(x_spatial)
        x_spatial = rearrange(x_spatial, 'b w (c h) -> b (h w) c', h=self.patch_size)

        # 分支2
        x_conv2 = rearrange(x, 'b c (h w) -> b c h w', h=self.patch_size)
        x_conv2 = self.conv2_branch(x_conv2) # [b, dim, h, w]
        x_conv2 = rearrange(x_conv2, 'b c h w -> b (h w) c')

        # 分支3
        x_conv1 = x.transpose(1, 2)
        x_conv1 = self.conv1_branch(x_conv1)

        # 特征融合
        return x_spatial + x_conv2 + x_conv1

class M3net(nn.Module):
    def __init__(self, patch_size, patch_dim, num_classes, embed_dim, depth, heads, mlp_dim, pool='cls',
                dropout=0., emb_dropout=0.):
        super().__init__()
        self.patch_size = patch_size
        self.num_classes = num_classes
        self.num_patches = patch_size ** 2
        self.patch_dim = patch_dim
        self.embed_dim = embed_dim

        self.MSTE = MultiScaleTokenEmbedding(patch_size, patch_dim, embed_dim)

        self.pos_embedding = nn.Parameter(torch.randn(1, self.num_patches + 1, embed_dim))
        # self.patch_to_embedding = nn.Linear(patch_dim, embed_dim)
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))

        self.dropout = nn.Dropout(emb_dropout)
        self.CAT = ConvAttentionTransformer(self.num_patches+1, embed_dim, depth, heads, mlp_dim, dropout)

        self.norm = nn.Identity() if pool=='pooling' else nn.LayerNorm(embed_dim)
        self.fc_norm = nn.LayerNorm(embed_dim) if pool=='pooling' else None
        self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def forward(self, x, mask=None, return_features=False):
        x = self.MSTE(x)
        # x = x.transpose(1, 2)  # [b,c,l] -> [b,l,c]
        # x = self.patch_to_embedding(x)  # [b,n,dim]

        b, n, _ = x.shape

        # add position embedding
        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b=b)  # [b,1,dim]
        x = torch.cat((cls_tokens, x), dim=1)  # [b,n+1,dim]
        x += self.pos_embedding[:, :(n + 1)]
        x = self.dropout(x)

        # transformer: x[b,n + 1,dim] -> x[b,n + 1,dim]
        x = self.CAT(x)

        # classification: using cls_token output
        x = self.norm(x)

        if self.fc_norm is not None:
            t = x[:, 1:, :]
            t = self.fc_norm(t.mean(1))
        else:
            t = x[:, 0]

        if return_features:
            return t

        x = self.head(t)
        return x


def build_m3net(args, **kwargs):
    model = M3net(
        patch_size=args.patch_size,
        patch_dim=kwargs['band'],
        num_classes=kwargs['num_classes'],
        embed_dim=96,
        depth=kwargs['depth'],
        heads=kwargs['heads'],
        mlp_dim=256,
        dropout=0.1,
        emb_dropout=0.1,
    )
    return model