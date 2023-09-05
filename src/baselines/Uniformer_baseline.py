import torch
import torch.nn as nn
import torch.nn.functional as F


class PatchEmbed(nn.Module):
    def __init__(self, feature_size=(120, 22), out_channels=64, in_channels=3, patch_size=4):
        super(PatchEmbed, self).__init__()
        self.patch_num = (feature_size[0] // patch_size) * (feature_size[1] // patch_size)
        self.fs = feature_size
        self.patch_size = patch_size
        self.norm = nn.LayerNorm(out_channels)
        self.conv = nn.Conv2d(in_channels=in_channels,
                              out_channels=out_channels,
                              kernel_size=patch_size,
                              stride=patch_size)

    def forward(self, x):
        """
        x: input dataset, in shape [batch, channels, frames, joints]
        """
        B, C, H, W = x.shape
        assert H == self.fs[0] and W == self.fs[1], \
            f"Input size ({H}*{W}) doesn't match model ({self.fs[0]}*{self.fs[1]})."
        x = self.conv(x)  # convolution on each patch
        B, C, H, W = x.shape  # H' = (H + 0 - patch_size) / patch_size + 1
        # transform each patch in to a long vector belong to R^out_channels

        x = x.flatten(2).transpose(1, 2)  # flatten form dim=2,   1. to [B, C, num_patches]   2. to [B, num_x, C]
        x = self.norm(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()  # [B, C, patch_num, patch_num]
        return x


class CMlp(nn.Module):
    def __init__(self, in_channels, hidden=None, out_channels=None):
        super(CMlp, self).__init__()
        out_channels = out_channels or in_channels
        hidden = hidden or in_channels
        self.fc1 = nn.Conv2d(in_channels, hidden, 1)
        self.act = nn.GELU()
        self.fc2 = nn.Conv2d(hidden, out_channels, 1)
        self.drop = nn.Dropout(0.3)

    def forward(self, x):
        """
        _param: x
                is the result of shallow layer attention (relation aggregator)
        """
        return self.drop(self.fc2(self.drop(self.act(self.fc1(x)))))


class CBlock(nn.Module):
    """
    shallow layer MHRA:
        The idea is to use a learnable affinity matrix to replace SA
        However, to efficiently implement this, group convolution is applied
    """

    def __init__(self, in_channels):
        super(CBlock, self).__init__()
        self.pos_embed = nn.Conv2d(in_channels, in_channels, 3, padding=1, groups=in_channels)
        self.norm1 = nn.BatchNorm2d(in_channels)
        self.conv1 = nn.Conv2d(in_channels, in_channels, 1)  # linear transformation of each patch
        self.shallow_sa = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, groups=in_channels)
        self.conv2 = nn.Conv2d(in_channels, in_channels, 1)
        self.norm2 = nn.BatchNorm2d(in_channels)
        self.mlp = CMlp(in_channels=in_channels)

    def forward(self, x):
        """
        e.g.
         the expected shape of "x" after first patch embedding is [batch, 64, 30, 6]
        """
        # conditional positional encoding
        pos = self.pos_embed(x)
        x = x + pos

        # shallow layer affinity learning (mainly focus on the neighbors)
        # so, use depth-wise convolution instead
        """
        procedures:
            1.  batch normalization
            2.  conv1: 1x1 conv, to emulate the linear transformation of each patch (token)
            3.  shallow_sa: depth-wise convolution, to simulate multi-heads attention in local area
            4.  conv2: 1x1 conv, 
        """
        x = x + self.conv2(self.shallow_sa(self.conv1(self.norm1(x))))

        # after attention, use a linear map to fuse features better.
        x = x + self.mlp(self.norm2(x))
        return x


class MLP(nn.Module):
    def __init__(self, in_channels, hidden=None, out_channels=None):
        super(MLP, self).__init__()
        out_channels = out_channels or in_channels
        hidden = hidden or in_channels
        self.fc1 = nn.Linear(in_channels, hidden)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden, out_channels)
        self.drop = nn.Dropout(0.3)

    def forward(self, x):
        """
        _param: x
                is the result of deep layer attention (relation aggregator)
                for original SA, it's usually in shape [batch, seq_len, token_len]
                so Linear Layer is used instead of conv1x1 in shallow layers
        """
        return self.drop(self.fc2(self.drop(self.act(self.fc1(x)))))


"""
copied from Uniformer Paper
"""


class Attention(nn.Module):
    #
    # just same machenism as in the "Attention is all you need" paper.
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape  # N = num_patches
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        # query key value all in shape
        # [Batch, heads, num_patches, hidden_dim]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class SABlock(nn.Module):
    def __init__(self, dim, num_heads=2, qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 norm_layer=nn.LayerNorm):
        super(SABlock, self).__init__()
        self.pos_embed = nn.Conv2d(dim, dim, 3, padding=1, groups=dim)
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop)

        self.norm2 = norm_layer(dim)
        self.mlp = MLP(in_channels=dim)

    def forward(self, x):
        x = x + self.pos_embed(x)
        B, N, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)

        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))

        x = x.transpose(1, 2).reshape(B, N, H, W)
        return x


class Uniformer(nn.Module):
    def __init__(self, in_channels, num_classes=3,
                 qkv_bias=True, qk_scale=None):
        super(Uniformer, self).__init__()
        self.num_classes = num_classes
        self.patch_embed1 = PatchEmbed(in_channels=in_channels, out_channels=64, patch_size=4)
        self.cblock = CBlock(in_channels=64)
        self.patch_embed2 = PatchEmbed(feature_size=(30, 5), in_channels=64, out_channels=128, patch_size=2)
        self.sa = SABlock(dim=128, qkv_bias=qkv_bias, qk_scale=qk_scale, drop=0.4)

        self.classify_head = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.patch_embed1(x)
        x = self.cblock(x)
        x = self.patch_embed2(x)
        x = self.sa(x)

        x = F.avg_pool2d(x, x.size()[2:]).squeeze()
        if len(x.shape) == 1:   # test instances are fed to the model one by one, so the squeeze above generate errors
            x = torch.unsqueeze(x, dim=0)

        out = self.classify_head(x)

        return [out]
