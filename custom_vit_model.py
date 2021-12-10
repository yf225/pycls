import math

import numpy as np
import torch
from pycls.core.config import cfg
from pycls.models.blocks import (
    # MultiheadAttention,
    activation,
    conv2d,
    conv2d_cx,
    layernorm,
    layernorm_cx,
    linear,
    linear_cx,
    norm2d,
    norm2d_cx,
    patchify2d,
    patchify2d_cx,
)
from torch.nn import Module, Parameter, init
from timm.models.vision_transformer import Attention


class ViTHead(Module):
    """Transformer classifier, an fc layer."""

    def __init__(self, w_in, num_classes):
        super().__init__()
        self.head_fc = linear(w_in, num_classes, bias=True)

    def forward(self, x):
        return self.head_fc(x)


class MLPBlock(Module):
    """Transformer MLP block, fc, gelu, fc."""

    def __init__(self, w_in, mlp_d):
        super().__init__()
        self.linear_1 = linear(w_in, mlp_d, bias=True)
        self.af = activation("gelu")
        self.linear_2 = linear(mlp_d, w_in, bias=True)

    def forward(self, x):
        return self.linear_2(self.af(self.linear_1(x)))


class ViTEncoderBlock(Module):
    """Transformer encoder block, following https://arxiv.org/abs/2010.11929."""

    def __init__(self, hidden_d, n_heads, mlp_d):
        super().__init__()
        self.ln_1 = layernorm(hidden_d)
        # self.self_attention = MultiheadAttentionSeparateProjection(hidden_d, n_heads)
        print(hidden_d, n_heads, True, 0., 0.)
        self.self_attention = Attention(dim=hidden_d, num_heads=n_heads, qkv_bias=True, attn_drop=0., proj_drop=0.)
        self.ln_2 = layernorm(hidden_d)
        self.mlp_block = MLPBlock(hidden_d, mlp_d)

    def forward(self, x):
        x_p = self.ln_1(x)
        # x_p, _ = self.self_attention(x_p, x_p, x_p)
        x_p = self.self_attention(x_p)
        x = x + x_p
        x_p = self.mlp_block(self.ln_2(x))
        return x + x_p


class ViTEncoder(Module):
    """Transformer encoder (sequence of ViTEncoderBlocks)."""

    def __init__(self, n_layers, hidden_d, n_heads, mlp_d):
        super(ViTEncoder, self).__init__()
        for i in range(n_layers):
            self.add_module(f"block_{i}", ViTEncoderBlock(hidden_d, n_heads, mlp_d))
        self.ln = layernorm(hidden_d)

    def forward(self, x):
        for block in self.children():
            x = block(x)
        return x


# class ViTStemPatchify(Module):
#     """The patchify vision transformer stem as per https://arxiv.org/abs/2010.11929."""

#     def __init__(self, w_in, w_out, k):
#         super(ViTStemPatchify, self).__init__()
#         self.patchify = patchify2d(w_in, w_out, k, bias=True)

#     def forward(self, x):
#         return self.patchify(x)

class PatchEncoder(torch.nn.Module):
    def __init__(self, img_size, patch_size, in_chans, embed_dim):
        super().__init__()
        img_size = (img_size, img_size)
        patch_size = (patch_size, patch_size)
        self.patch_size = patch_size
        self.in_chans = in_chans
        self.flatten_dim = self.patch_size[0] * self.patch_size[1] * in_chans
        self.img_size = img_size
        self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.projection = torch.nn.Linear(
            self.flatten_dim, embed_dim
        )
        self.position_embedding = torch.nn.Embedding(
            num_embeddings=self.num_patches, embedding_dim=embed_dim
        )

    def forward(self, input):
        rearranged_input = input.view(-1, self.grid_size[0] * self.grid_size[1], self.patch_size[0] * self.patch_size[1] * self.in_chans)
        # rearranged_input = einops.rearrange(
        #     input,
        #     "b c (h p1) (w p2) -> b (h w) (p1 p2 c)",
        #     p1=self.patch_size[0],
        #     p2=self.patch_size[1],
        # )
        positions = torch.arange(start=0, end=self.num_patches, step=1).to(input.device)
        ret = self.projection(rearranged_input)
        ret = ret + self.position_embedding(positions)
        return ret

# import warnings
# from typing import Optional, Tuple

# import torch
# from torch import Tensor
# from torch.nn.modules.linear import NonDynamicallyQuantizableLinear
# from torch.nn.init import constant_, xavier_normal_, xavier_uniform_
# from torch.nn.parameter import Parameter
# from torch.nn import functional as F

# class MultiheadAttentionSeparateProjection(Module):
#     __constants__ = ['batch_first']
#     bias_k: Optional[torch.Tensor]
#     bias_v: Optional[torch.Tensor]

#     def __init__(self, embed_dim, num_heads, dropout=0., bias=True, add_bias_kv=False, add_zero_attn=False,
#                  kdim=None, vdim=None, batch_first=False, device=None, dtype=None) -> None:
#         factory_kwargs = {'device': device, 'dtype': dtype}
#         super(MultiheadAttentionSeparateProjection, self).__init__()
#         self.embed_dim = embed_dim
#         self.kdim = kdim if kdim is not None else embed_dim
#         self.vdim = vdim if vdim is not None else embed_dim

#         self.num_heads = num_heads
#         self.dropout = dropout
#         self.batch_first = batch_first
#         self.head_dim = embed_dim // num_heads
#         assert self.head_dim * num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"

#         self.q_proj_weight = Parameter(torch.empty((embed_dim, embed_dim), **factory_kwargs))
#         self.k_proj_weight = Parameter(torch.empty((embed_dim, self.kdim), **factory_kwargs))
#         self.v_proj_weight = Parameter(torch.empty((embed_dim, self.vdim), **factory_kwargs))
#         self.register_parameter('in_proj_weight', None)

#         if bias:
#             self.in_proj_bias = Parameter(torch.empty(3 * embed_dim, **factory_kwargs))
#         else:
#             self.register_parameter('in_proj_bias', None)
#         self.out_proj = NonDynamicallyQuantizableLinear(embed_dim, embed_dim, bias=bias, **factory_kwargs)

#         if add_bias_kv:
#             self.bias_k = Parameter(torch.empty((1, 1, embed_dim), **factory_kwargs))
#             self.bias_v = Parameter(torch.empty((1, 1, embed_dim), **factory_kwargs))
#         else:
#             self.bias_k = self.bias_v = None

#         self.add_zero_attn = add_zero_attn

#         self._reset_parameters()

#     def _reset_parameters(self):
#         xavier_uniform_(self.q_proj_weight)
#         xavier_uniform_(self.k_proj_weight)
#         xavier_uniform_(self.v_proj_weight)

#         if self.in_proj_bias is not None:
#             constant_(self.in_proj_bias, 0.)
#             constant_(self.out_proj.bias, 0.)
#         if self.bias_k is not None:
#             xavier_normal_(self.bias_k)
#         if self.bias_v is not None:
#             xavier_normal_(self.bias_v)

#     def forward(self, query: Tensor, key: Tensor, value: Tensor, key_padding_mask: Optional[Tensor] = None,
#                 need_weights: bool = True, attn_mask: Optional[Tensor] = None) -> Tuple[Tensor, Optional[Tensor]]:
#         if self.batch_first:
#             query, key, value = [x.transpose(1, 0) for x in (query, key, value)]

#         attn_output, attn_output_weights = F.multi_head_attention_forward(
#             query, key, value, self.embed_dim, self.num_heads,
#             self.in_proj_weight, self.in_proj_bias,
#             self.bias_k, self.bias_v, self.add_zero_attn,
#             self.dropout, self.out_proj.weight, self.out_proj.bias,
#             training=self.training,
#             key_padding_mask=key_padding_mask, need_weights=need_weights,
#             attn_mask=attn_mask, use_separate_proj_weight=True,
#             q_proj_weight=self.q_proj_weight, k_proj_weight=self.k_proj_weight,
#             v_proj_weight=self.v_proj_weight)
#         if self.batch_first:
#             return attn_output.transpose(1, 0), attn_output_weights
#         else:
#             return attn_output, attn_output_weights

# # PyCls original version (use Conv2D for patchify, torch.zeros for pos embed)
# class ViT(Module):
#     """Vision transformer as per https://arxiv.org/abs/2010.11929."""

#     @staticmethod
#     def check_params(params):
#         p = params
#         err_str = "Input shape indivisible by patch size"
#         assert p["image_size"] % p["patch_size"] == 0, err_str

#     def __init__(self, params=None):
#         super(ViT, self).__init__()
#         p = ViT.get_params() if not params else params
#         ViT.check_params(p)
#         self.stem = ViTStemPatchify(3, p["hidden_d"], p["patch_size"])
#         seq_len = (p["image_size"] // cfg.VIT.PATCH_SIZE) ** 2
#         self.class_token = None
#         self.pos_embedding = Parameter(torch.zeros(seq_len, 1, p["hidden_d"]))
#         self.encoder = ViTEncoder(
#             p["n_layers"], p["hidden_d"], p["n_heads"], p["mlp_d"]
#         )
#         self.head = ViTHead(p["hidden_d"], p["num_classes"])

#     def forward(self, x):
#         # (n, c, h, w) -> (n, hidden_d, n_h, n_w)
#         x = self.stem(x)
#         # (n, hidden_d, n_h, n_w) -> (n, hidden_d, (n_h * n_w))
#         x = x.reshape(x.size(0), x.size(1), -1)
#         # (n, hidden_d, (n_h * n_w)) -> ((n_h * n_w), n, hidden_d)
#         x = x.permute(2, 0, 1)
#         x = x + self.pos_embedding
#         x = self.encoder(x)
#         x = x[0, :, :]
#         return self.head(x)


# Our modified version (use Dense for patchify, torch.zeros for pos embed)
class ViT(Module):
    """Vision transformer as per https://arxiv.org/abs/2010.11929."""

    @staticmethod
    def check_params(params):
        p = params
        err_str = "Input shape indivisible by patch size"
        assert p["image_size"] % p["patch_size"] == 0, err_str

    def __init__(self, params=None):
        super(ViT, self).__init__()
        p = ViT.get_params() if not params else params
        ViT.check_params(p)
        self.embed_layer = PatchEncoder(p["image_size"], p["patch_size"], 3, p["hidden_d"])
        self.encoder = ViTEncoder(
            p["n_layers"], p["hidden_d"], p["n_heads"], p["mlp_d"]
        )
        self.head = ViTHead(p["hidden_d"], p["num_classes"])

    def forward(self, x):
        # (n, c, h, w) -> (n, n_h, n_w, hidden_d)
        x = self.embed_layer(x)
        # (n, n_h, n_w, hidden_d) -> (n, (n_h * n_w), hidden_d)
        x = x.reshape(x.size(0), -1, x.size(-1))
        # (n, (n_h * n_w), hidden_d) -> ((n_h * n_w), n, hidden_d)
        x = x.transpose(0, 1)

        x = self.encoder(x)
        x = x[0, :, :]
        return self.head(x)
