import torch
import torch.nn as nn
from functools import partial
from timm.models.vision_transformer import Block, Mlp

class DVT(nn.Module):
    def __init__(
            self,
            feature_dim: int,
            num_transformer_blocks: int = 1,
    ):
        super().__init__()
        num_heads = max(1, feature_dim // 64)
        self.transformer_blocks = nn.Sequential(
            *[
                Block(
                    dim=feature_dim,
                    num_heads=num_heads,
                    mlp_ratio=4,
                    qkv_bias=True,
                    qk_norm=False,
                    init_values=None,
                    norm_layer=partial(nn.LayerNorm, eps=1e-6),
                    act_layer=nn.GELU,
                    mlp_layer=Mlp,
                )
                for _ in range(num_transformer_blocks)
            ]
        )
        self.pos_embed = None
    def forward(self, x):
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)  
        x = self.transformer_blocks(x)
        x = x.transpose(1, 2).reshape(B, C, H, W)
        return x
