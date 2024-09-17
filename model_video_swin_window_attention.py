import torch.nn.functional as F
from torch import nn
from torch.nn import Module
import torch
from audioUtils.hparams import hparams

class MyUpsample(Module):
    __constants__ = ['size', 'scale_factor', 'mode', 'align_corners', 'name']

    def __init__(self, size=None, scale_factor=None, mode='nearest', align_corners=None):
        super(MyUpsample, self).__init__()
        self.name = type(self).__name__
        self.size = size
        self.scale_factor = scale_factor if scale_factor else None
        self.mode = mode
        self.align_corners = align_corners

    def forward(self, input):
        return F.interpolate(input, self.size, self.scale_factor, self.mode, self.align_corners)

    def extra_repr(self):
        if self.scale_factor is not None:
            info = 'scale_factor=' + str(self.scale_factor)
        else:
            info = 'size=' + str(self.size)
        info += ', mode=' + self.mode
        return info


class VideoGenerator(nn.Module):
    # initializers
    def __init__(self, d=128, dim_neck=32, use_window=True, use_256=False):
        super(VideoGenerator, self).__init__()
        self.deconv1 = nn.ConvTranspose2d(256, d*8, 4, 1, 0)
        self.deconv1_bn = nn.BatchNorm2d(d*8)
        self.deconv2 = nn.ConvTranspose2d(d*8, d*4, 4, 2, 1)
        self.deconv2_bn = nn.BatchNorm2d(d*4)
        self.deconv3 = nn.ConvTranspose2d(d*4, d*2, 4, 2, 1)
        self.deconv3_bn = nn.BatchNorm2d(d*2)
        self.deconv4 = nn.ConvTranspose2d(d*2, d, 4, 2, 1)
        self.deconv4_bn = nn.BatchNorm2d(d)
        self.deconv5 = nn.ConvTranspose2d(d, d//2, 4, 2, 1)
        self.deconv5_bn = nn.BatchNorm2d(d//2)
        if use_256:
            self.deconv6 = nn.ConvTranspose2d(d // 2, d // 4, 4, 2, 1)
            self.deconv6_bn = nn.BatchNorm2d(d // 4)
            self.deconv7 = nn.ConvTranspose2d(d // 4, 3, 4, 2, 1)
        else:
            self.deconv7 = nn.ConvTranspose2d(d // 2, 3, 4, 2, 1)
        if not use_window:
            self.lstm = nn.LSTM(dim_neck*2, 256, 1, batch_first=True)
        else:
            self.window = nn.Conv1d(in_channels=dim_neck*2, out_channels=256, kernel_size=64, stride=4, padding=30)
        self.use_window = use_window
        self.use_256 = use_256

    # weight_init
    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)

    # forward method
    def forward(self, input, return_feature=False):
        # x = F.relu(self.deconv1(input))
        # print(input.shape)
        if self.use_window:
            input = self.window(input.transpose(1,2)).transpose(1,2)
        else:
            input, _ = self.lstm(input)
        # print(input.shape)
        batch_sz, num_frames, feat_dim = input.shape
        input = input.reshape(-1, feat_dim, 1, 1)
        x = F.relu(self.deconv1_bn(self.deconv1(input)))
        x = F.relu(self.deconv2_bn(self.deconv2(x)))
        x = F.relu(self.deconv3_bn(self.deconv3(x)))
        x = F.relu(self.deconv4_bn(self.deconv4(x)))
        x = F.relu(self.deconv5_bn(self.deconv5(x)))
        if self.use_256:
            x = F.relu(self.deconv6_bn(self.deconv6(x)))
        x = torch.tanh(self.deconv7(x))
        x = x.reshape(batch_sz, num_frames, x.shape[1], x.shape[2], x.shape[3])
        if return_feature:
            return x, input
        return x


def normal_init(m, mean, std):
    if isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Conv2d):
        m.weight.data.normal_(mean, std)
        m.bias.data.zero_()
        

def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


def conv3d(in_planes, out_planes, stride=1):
    return nn.Conv3d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)

# Upsale the spatial size by a factor of 2
def upBlock(in_planes, out_planes):
    block = nn.Sequential(
        # nn.Upsample(scale_factor=2, mode='nearest'),
        # conv3x3(in_planes, out_planes),
        MyUpsample(scale_factor=(1,2,2), mode='nearest'),
        conv3d(in_planes, out_planes),
        nn.BatchNorm3d(out_planes),
        nn.ReLU(True))
    return block

class ResBlock(nn.Module):
    def __init__(self, channel_num):
        super(ResBlock, self).__init__()
        self.block = nn.Sequential(
            conv3x3(channel_num, channel_num),
            nn.BatchNorm2d(channel_num),
            nn.ReLU(True),
            conv3x3(channel_num, channel_num),
            nn.BatchNorm2d(channel_num))
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = x
        out = self.block(x)
        out += residual
        out = self.relu(out)
        return out

class STAGE2_G(nn.Module):
    def __init__(self, residual=False):
        super(STAGE2_G, self).__init__()
        self.STAGE1_G = VideoGenerator()
        # fix parameters of stageI GAN
#         for param in self.STAGE1_G.parameters():
#             param.requires_grad = False
        self.define_module()
        self.residual_video = residual

    def _make_layer(self, block, channel_num):
        layers = []
        for i in range(4):
            layers.append(block(channel_num))
        return nn.Sequential(*layers)

    def define_module(self):
        ngf = 32
        # TEXT.DIMENSION -> GAN.CONDITION_DIM
        # --> 4ngf x 32 x 32
        self.encoder = nn.Sequential(
            conv3x3(3, ngf),
            nn.ReLU(True),
            nn.Conv2d(ngf, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            nn.Conv2d(ngf * 2, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True))
        self.hr_joint = nn.Sequential(
            conv3x3(256 + ngf * 4, ngf * 4),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True))
        self.residual = self._make_layer(ResBlock, ngf * 4)
        # --> 2ngf x 64 x 64
        self.upsample1 = upBlock(ngf * 4, ngf * 2)
        # --> ngf x 128 x 128
        self.upsample2 = upBlock(ngf * 2, ngf)
        # --> ngf // 2 x 256 x 256
        self.upsample3 = upBlock(ngf, ngf // 2)
        # --> ngf // 4 x 512 x 512
        self.upsample4 = upBlock(ngf // 2, ngf // 4)
        # --> 3 x 512 x 512
        self.img = nn.Sequential(
            conv3d(ngf // 4, 3),
            nn.Tanh())

    def forward(self, input, train=False):
        stage1_video, audio_embedding = self.STAGE1_G(input, return_feature=True)
        batch_sz, num_frames, _,_,_ = stage1_video.shape
        encoded_frames = self.encoder(stage1_video.reshape(batch_sz*num_frames,3,128,128))

        c_code = audio_embedding.reshape(batch_sz*num_frames,256,1,1)
        c_code = c_code.repeat(1, 1, 32, 32)
        i_c_code = torch.cat([encoded_frames, c_code], 1)
        h_code = self.hr_joint(i_c_code)
        h_code = self.residual(h_code) # (bs*num_frame)*4ngf*32*32

        h_code = h_code.reshape(batch_sz, num_frames, -1, 32, 32).transpose(2,1)
        h_code = self.upsample1(h_code)
        h_code = self.upsample2(h_code)
        h_code = self.upsample3(h_code)
        h_code = self.upsample4(h_code)

        stage2_video = self.img(h_code)
        stage2_video = stage2_video.transpose(2,1).reshape(batch_sz, num_frames, 3, 512, 512)

        if self.residual_video:
            stage2_video = MyUpsample(scale_factor=(1,4,4), mode='nearest')(stage1_video) + stage2_video

        if train:
            return stage1_video, stage2_video
        return stage2_video

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

from __future__ import annotations

import itertools
from collections.abc import Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from torch.nn import LayerNorm
from typing_extensions import Final

from monai.networks.blocks import MLPBlock as Mlp
from monai.networks.blocks import PatchEmbed, UnetOutBlock, UnetrBasicBlock, UnetrUpBlock
from monai.networks.layers import DropPath, trunc_normal_
from monai.utils import ensure_tuple_rep, look_up_option, optional_import
from monai.utils.deprecate_utils import deprecated_arg

def window_reverse(windows, window_size, dims):
    """window reverse operation based on: "Liu et al.,
    Swin Transformer: Hierarchical Vision Transformer using Shifted Windows
    <https://arxiv.org/abs/2103.14030>"
    https://github.com/microsoft/Swin-Transformer

     Args:
        windows: windows tensor.
        window_size: local window size.
        dims: dimension values.
    """
    if len(dims) == 4:
        b, d, h, w = dims
        x = windows.view(
            b,
            d // window_size[0],
            h // window_size[1],
            w // window_size[2],
            window_size[0],
            window_size[1],
            window_size[2],
            -1,
        )
        x = x.permute(0, 1, 4, 2, 5, 3, 6, 7).contiguous().view(b, d, h, w, -1)

    elif len(dims) == 3:
        b, h, w = dims
        x = windows.view(b, h // window_size[0], w // window_size[1], window_size[0], window_size[1], -1)
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(b, h, w, -1)
    return x


def get_window_size(x_size, window_size, shift_size=None):
    """Computing window size based on: "Liu et al.,
    Swin Transformer: Hierarchical Vision Transformer using Shifted Windows
    <https://arxiv.org/abs/2103.14030>"
    https://github.com/microsoft/Swin-Transformer

     Args:
        x_size: input size.
        window_size: local window size.
        shift_size: window shifting size.
    """

    use_window_size = list(window_size)
    if shift_size is not None:
        use_shift_size = list(shift_size)
    for i in range(len(x_size)):
        if x_size[i] <= window_size[i]:
            use_window_size[i] = x_size[i]
            if shift_size is not None:
                use_shift_size[i] = 0

    if shift_size is None:
        return tuple(use_window_size)
    else:
        return tuple(use_window_size), tuple(use_shift_size)


class WindowAttention(nn.Module):
    """
    Window based multi-head self attention module with relative position bias based on: "Liu et al.,
    Swin Transformer: Hierarchical Vision Transformer using Shifted Windows
    <https://arxiv.org/abs/2103.14030>"
    https://github.com/microsoft/Swin-Transformer
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        window_size: Sequence[int],
        qkv_bias: bool = False,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ) -> None:
        """
        Args:
            dim: number of feature channels.
            num_heads: number of attention heads.
            window_size: local window size.
            qkv_bias: add a learnable bias to query, key, value.
            attn_drop: attention dropout rate.
            proj_drop: dropout rate of output.
        """

        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5
        mesh_args = torch.meshgrid.__kwdefaults__

        if len(self.window_size) == 3:
            self.relative_position_bias_table = nn.Parameter(
                torch.zeros(
                    (2 * self.window_size[0] - 1) * (2 * self.window_size[1] - 1) * (2 * self.window_size[2] - 1),
                    num_heads,
                )
            )
            coords_d = torch.arange(self.window_size[0])
            coords_h = torch.arange(self.window_size[1])
            coords_w = torch.arange(self.window_size[2])
            if mesh_args is not None:
                coords = torch.stack(torch.meshgrid(coords_d, coords_h, coords_w, indexing="ij"))
            else:
                coords = torch.stack(torch.meshgrid(coords_d, coords_h, coords_w))
            coords_flatten = torch.flatten(coords, 1)
            relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
            relative_coords = relative_coords.permute(1, 2, 0).contiguous()
            relative_coords[:, :, 0] += self.window_size[0] - 1
            relative_coords[:, :, 1] += self.window_size[1] - 1
            relative_coords[:, :, 2] += self.window_size[2] - 1
            relative_coords[:, :, 0] *= (2 * self.window_size[1] - 1) * (2 * self.window_size[2] - 1)
            relative_coords[:, :, 1] *= 2 * self.window_size[2] - 1
        elif len(self.window_size) == 2:
            self.relative_position_bias_table = nn.Parameter(
                torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads)
            )
            coords_h = torch.arange(self.window_size[0])
            coords_w = torch.arange(self.window_size[1])
            if mesh_args is not None:
                coords = torch.stack(torch.meshgrid(coords_h, coords_w, indexing="ij"))
            else:
                coords = torch.stack(torch.meshgrid(coords_h, coords_w))
            coords_flatten = torch.flatten(coords, 1)
            relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
            relative_coords = relative_coords.permute(1, 2, 0).contiguous()
            relative_coords[:, :, 0] += self.window_size[0] - 1
            relative_coords[:, :, 1] += self.window_size[1] - 1
            relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1

        relative_position_index = relative_coords.sum(-1)
        self.register_buffer("relative_position_index", relative_position_index)
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        trunc_normal_(self.relative_position_bias_table, std=0.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask):
        b, n, c = x.shape
        qkv = self.qkv(x).reshape(b, n, 3, self.num_heads, c // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        q = q * self.scale
        attn = q @ k.transpose(-2, -1)
        relative_position_bias = self.relative_position_bias_table[
            self.relative_position_index.clone()[:n, :n].reshape(-1)
        ].reshape(n, n, -1)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        attn = attn + relative_position_bias.unsqueeze(0)
        if mask is not None:
            nw = mask.shape[0]
            attn = attn.view(b // nw, nw, self.num_heads, n, n) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, n, n)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn).to(v.dtype)
        x = (attn @ v).transpose(1, 2).reshape(b, n, c)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x
    
    
class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout))
            ]))
    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x

class VideoEncoder(nn.Module):
    def __init__(self, *, image_size, patch_size, num_classes, dim, depth, heads, mlp_dim, pool = 'cls', channels = 3, dim_head = 64, dropout = 0., emb_dropout = 0.):
        super().__init__()
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)

        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'

        num_patches = (image_height // patch_height) * (image_width // patch_width)
        patch_dim = channels * patch_height * patch_width
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_height, p2 = patch_width),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim),
            nn.LayerNorm(dim),
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

        self.pool = pool
        self.to_latent = nn.Identity()

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )

    def forward(self, img):
        x = self.to_patch_embedding(img)
        b, n, _ = x.shape

        cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b = b)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :(n + 1)]
        x = self.dropout(x)

        x = self.transformer(x)

        x = x.mean(dim = 1) if self.pool == 'mean' else x[:, 0]

        x = self.to_latent(x)
        return self.mlp_head(x)

#class VideoEncoder(nn.Module):
 #   def __init__(self):
  #      super(VideoEncoder, self).__init__()
   #     self.encoder = nn.Sequential(
    #        nn.Conv3d(3, 64, kernel_size=(3,4,4), stride=(1,2,2), padding=1, bias=False), # 32*256*256
     #       nn.BatchNorm3d(64),
      #      nn.ReLU(True),
       #     nn.Conv3d(64, 128, kernel_size=(3, 4, 4), stride=(1, 2, 2), padding=1, bias=False), # 32*128*128
        #    nn.BatchNorm3d(128),
         #   nn.ReLU(True),
            #nn.Conv3d(128, 256, kernel_size=(3, 4, 4), stride=(1, 2, 2), padding=1, bias=False), # 32*64*64
            #nn.BatchNorm3d(256),
            #nn.ReLU(True),
            #nn.Conv3d(256, 256, kernel_size=(3, 4, 4), stride=(1, 2, 2), padding=1, bias=False),  # 32*32*32
            #nn.BatchNorm3d(256),
            #nn.ReLU(True),
            #nn.Conv3d(256, 256, kernel_size=(3, 4, 4), stride=(1, 2, 2), padding=1, bias=False),  # 32*16*16
            #nn.BatchNorm3d(256),
            #nn.ReLU(True),
            #nn.Conv3d(256, 128, kernel_size=(3, 4, 4), stride=(1, 2, 2), padding=1, bias=False),  # 32*8*8
            #nn.BatchNorm3d(128),
            #nn.ReLU(True),
            #nn.Conv3d(128, 128, kernel_size=(3, 4, 4), stride=(1, 2, 2), padding=1, bias=False),  # 32*4*4
            #nn.BatchNorm3d(128),
            #nn.ReLU(True),
            #nn.Conv3d(128, 128, kernel_size=(3, 4, 4), stride=(1, 2, 2), padding=1, bias=False),  # 32*2*2
            #nn.BatchNorm3d(128),
           # nn.ReLU(True),
            #nn.Conv3d(128, 128, kernel_size=(3, 4, 4), stride=(1, 2, 2), padding=1, bias=False),  # 32*1*1
            #nn.BatchNorm3d(128),
            #nn.ReLU(True),
       # )
        #self.projection = nn.Sequential(
#            nn.Conv1d(128, 64, kernel_size=4, stride=2, padding=1, bias=False),
 #           nn.BatchNorm1d(64),
#            nn.ReLU(True),
 #           nn.Conv1d(64, 64, kernel_size=4, stride=2, padding=1, bias=False),
  #          nn.BatchNorm1d(64),
   #         nn.ReLU(True),
   #         nn.Conv1d(64, 64, kernel_size=4, stride=2, padding=1, bias=False),
   #         nn.BatchNorm1d(64),
   #         nn.ReLU(True),
   #         nn.Conv1d(64, 64, kernel_size=3, stride=1, padding=1, bias=True)
   #     )

   # def forward(self, x):
        # batch * time * channel * 512 * 512
    #    batch_sz, num_frames, _, _, _ = x.shape
    #    x = x.transpose(2, 1)
    #    x = self.encoder(x) # batch * 128 * time * 1 * 1
    #    x = x.reshape(batch_sz, 128, num_frames)
     #   x = self.projection(x)
        # print(x.shape)
     #   return x.transpose(1,2)

