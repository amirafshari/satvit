from functools import partial
from typing import Any, Callable, Dict, List, NamedTuple, Optional
from collections import OrderedDict
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
import time
from torch.utils.data.dataset import random_split
from torch import nn, einsum
from einops import rearrange
from einops.layers.torch import Rearrange
from torch import nn, einsum
from einops import rearrange
from einops.layers.torch import Rearrange




class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)



class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
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


class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        # print(x.shape)
        b, n, _, h = *x.shape, self.heads
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), qkv)
        # print(q.shape, k.shape, v.shape)
        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale

        attn = dots.softmax(dim=-1)

        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.to_out(out)
        return out



class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout):
        super().__init__()
        self.layers = nn.ModuleList([])
        self.norm = nn.LayerNorm(dim)
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout))
            ]))

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return self.norm(x)

class Segmentation(nn.Module):
    def __init__(self, img_height=24, img_width=24, in_channel=10,
                       patch_size=3, embed_dim=128, max_time=60,
                       num_classes=20, num_head=4, dim_feedforward=2048,
                       num_layers=4 , dropoutratio=0.5

                ):
        super().__init__()
        
        self.H = img_height
        self.W = img_width
        self.P = patch_size
        self.C = in_channel
        self.d = embed_dim
        self.T = max_time
        self.K = num_classes
        self.d_model = self.d
        self.num_head = num_head
        self.dim_feedforward = self.d
        self.num_layers = num_layers

        self.N = int(self.H * self.W // self.P**2)
        self.nh = int(self.H / self.P)
        self.nw = int(self.W / self.P)

        self.dropout = nn.Dropout(p=dropoutratio)

        '''
        PARAMETERS
        '''
        # Transformer Encoder

        # PyTorch Encoder
        # self.encoderLayer = nn.TransformerEncoderLayer(d_model=self.d_model, nhead=self.num_head, dim_feedforward=self.dim_feedforward)
        # self.encoder = nn.TransformerEncoder(self.encoderLayer, num_layers=self.num_layers)

        # DeepSat Encoder
        self.encoder = Transformer(self.d, self.num_layers, self.num_head, 32, self.d*4, dropoutratio)



        # torchvision Encoder
        # self.encoder = Encoder(seq_length=self.N, num_heads=4, num_layers=4, hidden_dim=self.d, mlp_dim=self.d*4, dropout=0., attention_dropout=0.)


        # Patches
        self.projection = nn.Conv3d(self.C, self.d, kernel_size=(1, self.P, self.P), stride=(1, self.P, self.P))
        '''
        def __init__():
            self.linear = nn.Linear(self.C*self.P**2, self.d)
        def forward():
            x = x.view(B, T, H // P, W // P, C*P**2)
            x = self.linear(x)
        '''

        # Temporal
        self.temporal_emb = nn.Linear(366, self.d)
        self.temporal_cls_token = nn.Parameter(torch.randn(1, self.N, self.K, self.d)) # (N, K, d)
        self.temporal_transformer = self.encoder

        # Spatial
        self.spatial_emb = nn.Parameter(torch.randn(1, self.N, self.d)) # (1, N, d)
        # self.spatial_cls_token = nn.Parameter(torch.randn(1, self.K, self.d)) # (1, K, d)
        self.spatial_transformer = self.encoder

        # Segmentation Head
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(self.d),
            nn.Linear(self.d, self.P**2)
            )



    def forward(self, x):
        '''
        Tekenization

        Convert the images to a sequence of patches
        '''
        x_sits = x[:, :, :-1, :, :] # (B, T, C, H, W) -- > Exclude DOY Channel
        B, T, C, H, W = x_sits.shape # (B, T, C, H, W)
        x_sits = x_sits.reshape(B, C, T, H, W) # (B, C, T, H, W)
        x_sits = self.projection(x_sits) # (B, d, T, nw, nh)
        x_sits = self.dropout(x_sits) 

        x_sits = x_sits.reshape(B, self.d, T, self.nh*self.nw) # (B, d, T, N)
        # x_sits = x_sits + self.pos_emb # (B, d, T, N)  we dont add pos embedding here, cuz we need the pure data for the temporal encoder
        x_sits = x_sits.permute(0,3,2,1) # (B, N, T, d)



        '''
        Temporal Encoding

        (DOY -> One-Hot -> Projection)
        '''
        xt = x[:, :, -1, 0, 0] # (B, T, C, H, W) in the last channel lies the DOY feature
        xt = F.one_hot(xt.to(torch.int64), num_classes=366).to(torch.float32) # (B, T, 366)
        Pt = self.temporal_emb(xt) # (B, T, d) (DOY, one-hot encoded to represent the DOY feature and then encoded to d dimensions)




        '''
        Temporal Encoder: cat(Z+Pt)

        add temporal embeddings (N*K) to the Time Series patches (T)
        '''
        x = x_sits + Pt.unsqueeze(1) # (B, N, T, d)
        x = self.dropout(x)

        temporal_cls_token = self.temporal_cls_token # (1, N, K, d)
        temporal_cls_token = temporal_cls_token.repeat(B, 1, 1, 1) # (B, N, K, d)
        temporal_cls_token = temporal_cls_token.reshape(B*self.N, self.K, self.d) # (B*N, K, d)
        x = x.reshape(B*self.N, T, self.d) # (B*N, T, d)
        # Temporal Tokens (N*K)
        x = torch.cat([temporal_cls_token, x], dim=1) # (B*N, K+T, d)
        # Temporal Transformer
        x = self.temporal_transformer(x) # (B*N, K+T, d)
        x = x.reshape(B, self.N, self.K + T, self.d) # (B, N, K+T, d)
        x = x[:,:,:self.K,:] # (B, N, K, d)
        x = x.permute(0, 2, 1, 3) # (B, K, N, d)
        x = x.reshape(B*(self.K), self.N, self.d) # (B*K, N, d)




        '''
        Spatial Encoding
        '''
        Ps = self.spatial_emb # (1, N, d)
        x = x + Ps # (B*K, N, d)
        x = self.dropout(x)

        '''
        # For Classification Only
        # spatial_cls_token = self.spatial_cls_token # (1, K, d)
        # spatial_cls_token = spatial_cls_token.unsqueeze(2) # (1, K, 1, d)
        # spatial_cls_token = spatial_cls_token.repeat(B, 1, 1, 1) # (B, K, 1, d)
        # x = torch.cat([spatial_cls_token, x], dim=2) # (B, K, 1+N, d)
        '''
        x = self.spatial_transformer(x) # (B*K, N, d)
        x = x.reshape(B, self.K, self.N, self.d) # (B, K, N, d)
        x = x.permute(0, 2, 1, 3) # (B, N, K, d)


        '''
        Segmentation Head
        '''
        # classes = x[:,:,0,:] # (B, K, d)
        # x = x[:,:,1:,:] # (B, K, N, d)
        x = self.dropout(x)

        x = self.mlp_head(x) # (B, N, K, P*P)


        '''
        Reassemble
        '''
        x = x.permute(0, 2, 3, 1) # (B, N, P*P, K)
        x = x.reshape(B, self.N, self.P, self.P, self.K) # (B, N, P, P, K)
        x = x.reshape(B, self.H, self.W, self.K) # (B, H, W, K)
        # x = x.permute(0, 3, 1, 2) # (B, K, H, W)


        return x
