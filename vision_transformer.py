import os
import math
from functools import partial

import torch
import torch.nn as nn
from torch.nn.init import trunc_normal_

def drop_path(x, drop_prob: float =0, training: bool =False):
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()
    output = x.div(keep_prob) * random_tensor
    return output

class DropPath(nn.Module):
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class Attentions(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False,qk_scale =None ,attn_drop=0.,proj_drop=0.):
        super().__init__() 
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
    def forward(self, x):
        B,N,C = x.shape
        qkv = self.qkv(x).reshape(B,N,3,self.num_heads,C//self.num_heads).permute(2,0,3,1,4)
        q,k,v = qkv[0],qkv[1],qkv[2]
        attn = (q @ k.transpose(-2,-1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1,2).reshape(B,N,C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x, attn
    
class Block(nn.Module):
    def __init__(self, num_heads, dim, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attentions(dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
                               attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim
                          , act_layer=act_layer, drop=drop)
        
    def forward(self, x,return_attention=False):
        y, attn = self.attn(self.norm1(x))
        if return_attention:
            return attn
        x = x + self.drop_path(y)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x
        
class PatchEmbed(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        num_patches = (img_size // patch_size) * (img_size // patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        B, C, H, W = x.shape
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x

class VisionTransformer(nn.Module):
    def __init__(self, img_size=[224], patch_size=16, in_chans=3,num_register_tokens =0 ,num_classes=0,
                 embed_dim=768, depth=12, num_heads=12, mlp_ratio=4., qkv_bias=True,
                 qk_scale=None, drop_rate=0., attn_drop_rate=0., drop_path_rate=0.,
                 norm_layer=nn.LayerNorm,**kwargs):
        super().__init__()
        self.num_features = self.embed_dim = embed_dim
        self.patch_embed = PatchEmbed(
            img_size=img_size[0], patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches
        self.num_register_tokens = num_register_tokens 
        if num_register_tokens > 0:
            self.register_tokens = nn.Parameter(torch.zeros(1, num_register_tokens, embed_dim)) 
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1 + num_register_tokens, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()
        trunc_normal_(self.pos_embed, std=.02)
        trunc_normal_(self.cls_token, std=.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        
    def interpolate_pos_encoding(self, x, w, h):
        num_extra_tokens = 1 + self.num_register_tokens
        npatch = x.shape[1] -1
        N = self.pos_embed.shape[1] - num_extra_tokens
        if npatch == N:
            return self.pos_embed
        extra_tokens = self.pos_embed[:, :num_extra_tokens]
        patch_pos_embed = self.pos_embed[:, num_extra_tokens:]
        dim = x.shape[-1]
        w0_src = int(math.sqrt(N))
        h0_src = int(math.sqrt(N))
        w0 = w // self.patch_embed.patch_size
        h0 = h // self.patch_embed.patch_size
        patch_pos_embed = patch_pos_embed.reshape(1, w0_src, h0_src, dim).permute(0, 3, 1, 2)
        patch_pos_embed = nn.functional.interpolate(
            patch_pos_embed,
            size=(w0, h0), 
            mode='bicubic',
            align_corners=False
        )
        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
        return torch.cat((extra_tokens, patch_pos_embed), dim=1)
        
    def prepare_tokens(self, x):
        B, nc, w, h = x.shape
        x = self.patch_embed(x)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        if self.num_register_tokens > 0:
            register_tokens = self.register_tokens.expand(B, -1, -1)
            x = torch.cat((cls_tokens, register_tokens, x), dim=1)
        else:
            x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.interpolate_pos_encoding(x, w, h)
        return self.pos_drop(x)
        
    def forward(self, x):
        x = self.prepare_tokens(x)
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        return x[:,0]
        
    def get_last_selfattention(self, x):
        x = self.prepare_tokens(x)
        for i,blk in enumerate(self.blocks):
            if i < len(self.blocks) -1:
                x = blk(x)
            else:
                return blk(x, return_attention=True)
        
    def get_intermediate_layers(self, x, n):
        x = self.prepare_tokens(x)
        output = []
        for i,blk in enumerate(self.blocks):
            x = blk(x)
            if len(self.blocks) - i <= n:
                output.append(self.norm(x))
        return output
        
def vit_tiny(patch_size=16, **kwargs):
    model = VisionTransformer(
        patch_size=patch_size, embed_dim=192, depth=12, num_heads=3, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

def vit_small(patch_size=16, **kwargs):
    model = VisionTransformer(
        patch_size=patch_size, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

def vit_small_patch14(patch_size=14, **kwargs):
    model = VisionTransformer(
        patch_size=patch_size, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

def vit_base(patch_size=16, **kwargs):
    model = VisionTransformer(
        patch_size=patch_size, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

def vit_large(patch_size=16, **kwargs):
    model = VisionTransformer(
        patch_size=patch_size, embed_dim=1024, depth=24, num_heads=16, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

def vit_huge(patch_size=14, **kwargs):
    model = VisionTransformer(
        patch_size=patch_size, embed_dim=1280, depth=32, num_heads=16, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

def vit_base_patch14(patch_size=14, **kwargs):
    model = VisionTransformer(
        patch_size=patch_size, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

def vit_base_r4(patch_size=14,num_register_tokens=4, **kwargs):
    model = VisionTransformer(
        patch_size=patch_size, embed_dim=768, depth=4, num_heads=12,num_register_tokens=num_register_tokens, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

class ViTFeat(nn.Module):
    def __init__(self, model,feat_dim, vit_feat ='k', patch_size=16):
        super().__init__()
        self.model = model
        self.vit_feat = vit_feat
        self.patch_size = patch_size
        self.feat_dim = feat_dim
    def forward(self, img):
        feat_out ={}
        def hook(module, input, output):
            feat_out["qkv"] = output

        self.model.blocks[-1].attn.qkv.register_forward_hook(hook)
        with torch.no_grad():
            h,w = img.shape[2], img.shape[3]
            feat_h, feat_w = h // self.patch_size, w // self.patch_size
            num_patches = feat_w * feat_h
            attentions = self.model.get_last_selfattention(img)
            bs, nb_head, nb_tokens = attentions.shape[0], attentions.shape[1], attentions.shape[2]
            qkv = (feat_out["qkv"].reshape(bs, nb_tokens, 3, nb_head, -1)
                   .permute(2, 0, 3, 1, 4))
            q, k, v = qkv[0], qkv[1], qkv[2]
            k = k.transpose(1, 2).reshape(bs, nb_tokens, -1)
            q = q.transpose(1, 2).reshape(bs, nb_tokens, -1)
            v = v.transpose(1, 2).reshape(bs, nb_tokens, -1)
            total_tokens = k.shape[1]
            num_extra_tokens = total_tokens - num_patches
            if self.vit_feat == 'k':
                vit_feat = k[:, num_extra_tokens:, :].reshape(bs, feat_h, feat_w, -1).permute(0, 3, 1, 2)
            elif self.vit_feat == 'q':
                vit_feat = q[:, num_extra_tokens:, :].reshape(bs, feat_h, feat_w, -1).permute(0, 3, 1, 2)
            elif self.vit_feat == 'v':
                vit_feat = v[:, num_extra_tokens:, :].reshape(bs, feat_h, feat_w, -1).permute(0, 3, 1, 2)
            elif self.vit_feat == "kqv":
                k = k[:, num_extra_tokens:, :].reshape(bs, feat_h, feat_w, -1).permute(0, 3, 1, 2)
                q = q[:, num_extra_tokens:, :].reshape(bs, feat_h, feat_w, -1).permute(0, 3, 1, 2)
                v = v[:, num_extra_tokens:, :].reshape(bs, feat_h, feat_w, -1).permute(0, 3, 1, 2)
                vit_feat = torch.cat([k, q, v], dim=1)

            return vit_feat


            

        