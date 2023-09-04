import torch
from torch import nn
from torchsummary import summary


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.1):
        '''dim is the length of the input sequences'''

        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias) 
        self.kv = nn.Linear(dim, dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, H, W):
        B, N, C = x.shape           # [Batchsize (B) x num_patch (N) x embed_size (C)]

        # Q matrix [B x N x C] ----> [B x N x h x (C/h)] ----> [B x h x N x S]; S = C/h
        q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3) 

        # We use a reduction technique to reduce the computational complex of 
        # [B x N x C] ----> [B x N/2 x 2 x h x S] ----> [2 x B x h x N/2 x S]
        kv = self.kv(x).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4) 
        k, v = kv[0], kv[1] # [B x h x N/2 x S], [B x h x N/2 x S]

        # Calculate attention weight [B x h x N x S] x [B x h x S x N/2] = [B x h x N x N/2]
        attn = (q @ k.transpose(-2, -1)) * self.scale 
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        # Calculate attention [B x h x N x N/2] x [B x h x N/2 x S] = [B x h x N x S] 
        # [B x h x N x S] ----> [B x N x h x S] ----> [B x N x (hxS)] = [B x N x C]
        x = (attn @ v).transpose(1, 2).reshape(B, N, C) 
        x = self.proj(x)
        x = self.proj_drop(x)

        return x
ax = torch.rand(1, 3136, 768)
att = Attention(768)(ax, 56,56)
print(att.shape) # torch.Size([1, 3136, 768])



'''
# For more information

# h = 4
# x = torch.rand((1, 10, 64))
# B, N, C = x.shape 
# q = x.reshape(B, N, h, C//h).permute(0, 2, 1, 3)
# print('Query ', q.shape)

# kv =x.reshape(B, -1, 2,  h, C//h).permute(2, 0, 3, 1, 4)
# print(kv.shape)

# k, v = kv[0], kv[1]
# print('Key ', k.shape, 'Value', v.shape)

'''