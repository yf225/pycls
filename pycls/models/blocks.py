#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Common model blocks."""

import numpy as np
import torch
import torch.nn as nn
from pycls.core.config import cfg
from torch.nn import Module


# ----------------------- Shortcuts for common torch.nn layers ----------------------- #


def conv2d(w_in, w_out, k, *, stride=1, groups=1, bias=False):
    """Helper for building a conv2d layer."""
    assert k % 2 == 1, "Only odd size kernels supported to avoid padding issues."
    s, p, g, b = stride, (k - 1) // 2, groups, bias
    return nn.Conv2d(w_in, w_out, k, stride=s, padding=p, groups=g, bias=b)


def patchify2d(w_in, w_out, k, *, bias=True):
    """Helper for building a patchify layer as used by ViT models."""
    return nn.Conv2d(w_in, w_out, k, stride=k, padding=0, bias=bias)


def norm2d(w_in):
    """Helper for building a norm2d layer."""
    return nn.BatchNorm2d(num_features=w_in, eps=cfg.BN.EPS, momentum=cfg.BN.MOM)


def pool2d(_w_in, k, *, stride=1):
    """Helper for building a pool2d layer."""
    assert k % 2 == 1, "Only odd size kernels supported to avoid padding issues."
    return nn.MaxPool2d(k, stride=stride, padding=(k - 1) // 2)


def gap2d(_w_in):
    """Helper for building a gap2d layer."""
    return nn.AdaptiveAvgPool2d((1, 1))


def layernorm(w_in):
    """Helper for building a layernorm layer."""
    return nn.LayerNorm(w_in, eps=cfg.LN.EPS)


def linear(w_in, w_out, *, bias=False):
    """Helper for building a linear layer."""
    return nn.Linear(w_in, w_out, bias=bias)


def activation(activation_fun=None):
    """Helper for building an activation layer."""
    activation_fun = (activation_fun or cfg.MODEL.ACTIVATION_FUN).lower()
    if activation_fun == "relu":
        return nn.ReLU(inplace=cfg.MODEL.ACTIVATION_INPLACE)
    elif activation_fun == "silu" or activation_fun == "swish":
        try:
            return torch.nn.SiLU()
        except AttributeError:
            return SiLU()
    elif activation_fun == "gelu":
        return torch.nn.GELU()
    else:
        raise AssertionError("Unknown MODEL.ACTIVATION_FUN: " + activation_fun)


# --------------------------- Complexity (cx) calculations --------------------------- #


def conv2d_cx(cx, w_in, w_out, k, *, stride=1, groups=1, bias=False):
    """Accumulates complexity of conv2d into cx = (h, w, flops, params, acts)."""
    assert k % 2 == 1, "Only odd size kernels supported to avoid padding issues."
    h, w, flops, params, acts = cx["h"], cx["w"], cx["flops"], cx["params"], cx["acts"]
    h, w = (h - 1) // stride + 1, (w - 1) // stride + 1
    flops += k * k * w_in * w_out * h * w // groups + (w_out * h * w if bias else 0)
    params += k * k * w_in * w_out // groups + (w_out if bias else 0)
    acts += w_out * h * w
    return {"h": h, "w": w, "flops": flops, "params": params, "acts": acts}


def patchify2d_cx(cx, w_in, w_out, k, *, bias=True):
    """Accumulates complexity of patchify2d into cx = (h, w, flops, params, acts)."""
    err_str = "Only kernel sizes divisible by the input size are supported."
    assert cx["h"] % k == 0 and cx["w"] % k == 0, err_str
    h, w, flops, params, acts = cx["h"], cx["w"], cx["flops"], cx["params"], cx["acts"]
    h, w = h // k, w // k
    flops += k * k * w_in * w_out * h * w + (w_out * h * w if bias else 0)
    params += k * k * w_in * w_out + (w_out if bias else 0)
    acts += w_out * h * w
    return {"h": h, "w": w, "flops": flops, "params": params, "acts": acts}


def norm2d_cx(cx, w_in):
    """Accumulates complexity of norm2d into cx = (h, w, flops, params, acts)."""
    h, w, flops, params, acts = cx["h"], cx["w"], cx["flops"], cx["params"], cx["acts"]
    params += 2 * w_in
    return {"h": h, "w": w, "flops": flops, "params": params, "acts": acts}


def pool2d_cx(cx, w_in, k, *, stride=1):
    """Accumulates complexity of pool2d into cx = (h, w, flops, params, acts)."""
    assert k % 2 == 1, "Only odd size kernels supported to avoid padding issues."
    h, w, flops, params, acts = cx["h"], cx["w"], cx["flops"], cx["params"], cx["acts"]
    h, w = (h - 1) // stride + 1, (w - 1) // stride + 1
    acts += w_in * h * w
    return {"h": h, "w": w, "flops": flops, "params": params, "acts": acts}


def gap2d_cx(cx, _w_in):
    """Accumulates complexity of gap2d into cx = (h, w, flops, params, acts)."""
    flops, params, acts = cx["flops"], cx["params"], cx["acts"]
    return {"h": 1, "w": 1, "flops": flops, "params": params, "acts": acts}


def layernorm_cx(cx, w_in):
    """Accumulates complexity of layernorm into cx = (h, w, flops, params, acts)."""
    h, w, flops, params, acts = cx["h"], cx["w"], cx["flops"], cx["params"], cx["acts"]
    params += 2 * w_in
    return {"h": h, "w": w, "flops": flops, "params": params, "acts": acts}


def linear_cx(cx, w_in, w_out, *, bias=False, num_locations=1):
    """Accumulates complexity of linear into cx = (h, w, flops, params, acts)."""
    h, w, flops, params, acts = cx["h"], cx["w"], cx["flops"], cx["params"], cx["acts"]
    flops += w_in * w_out * num_locations + (w_out * num_locations if bias else 0)
    params += w_in * w_out + (w_out if bias else 0)
    acts += w_out * num_locations
    return {"h": h, "w": w, "flops": flops, "params": params, "acts": acts}


# ---------------------------------- Shared blocks ----------------------------------- #


class SiLU(Module):
    """SiLU activation function (also known as Swish): x * sigmoid(x)."""

    # Note: will be part of Pytorch 1.7, at which point can remove this.

    def __init__(self):
        super(SiLU, self).__init__()

    def forward(self, x):
        return x * torch.sigmoid(x)


class SE(Module):
    """Squeeze-and-Excitation (SE) block: AvgPool, FC, Act, FC, Sigmoid."""

    def __init__(self, w_in, w_se):
        super(SE, self).__init__()
        self.avg_pool = gap2d(w_in)
        self.f_ex = nn.Sequential(
            conv2d(w_in, w_se, 1, bias=True),
            activation(),
            conv2d(w_se, w_in, 1, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return x * self.f_ex(self.avg_pool(x))

    @staticmethod
    def complexity(cx, w_in, w_se):
        h, w = cx["h"], cx["w"]
        cx = gap2d_cx(cx, w_in)
        cx = conv2d_cx(cx, w_in, w_se, 1, bias=True)
        cx = conv2d_cx(cx, w_se, w_in, 1, bias=True)
        cx["h"], cx["w"] = h, w
        return cx


import warnings
from typing import Optional, Tuple

import torch
from torch import Tensor
from torch.nn.init import constant_, xavier_normal_, xavier_uniform_
from torch.nn.parameter import Parameter


class MultiheadAttentionUseSeparateProjWeight(nn.Module):
    __constants__ = ['batch_first']
    bias_k: Optional[torch.Tensor]
    bias_v: Optional[torch.Tensor]

    def __init__(self, embed_dim, num_heads, dropout=0., bias=True, add_bias_kv=False, add_zero_attn=False,
                 kdim=None, vdim=None, batch_first=False, device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(MultiheadAttention, self).__init__()
        self.embed_dim = embed_dim
        self.kdim = kdim if kdim is not None else embed_dim
        self.vdim = vdim if vdim is not None else embed_dim

        self.num_heads = num_heads
        self.dropout = dropout
        self.batch_first = batch_first
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"

        self.q_proj_weight = Parameter(torch.empty((embed_dim, embed_dim), **factory_kwargs))
        self.k_proj_weight = Parameter(torch.empty((embed_dim, self.kdim), **factory_kwargs))
        self.v_proj_weight = Parameter(torch.empty((embed_dim, self.vdim), **factory_kwargs))
        self.register_parameter('in_proj_weight', None)

        if bias:
            self.in_proj_bias = Parameter(torch.empty(3 * embed_dim, **factory_kwargs))
        else:
            self.register_parameter('in_proj_bias', None)
        self.out_proj = NonDynamicallyQuantizableLinear(embed_dim, embed_dim, bias=bias, **factory_kwargs)

        if add_bias_kv:
            self.bias_k = Parameter(torch.empty((1, 1, embed_dim), **factory_kwargs))
            self.bias_v = Parameter(torch.empty((1, 1, embed_dim), **factory_kwargs))
        else:
            self.bias_k = self.bias_v = None

        self.add_zero_attn = add_zero_attn

        self._reset_parameters()

    def _reset_parameters(self):
        xavier_uniform_(self.q_proj_weight)
        xavier_uniform_(self.k_proj_weight)
        xavier_uniform_(self.v_proj_weight)

        if self.in_proj_bias is not None:
            constant_(self.in_proj_bias, 0.)
            constant_(self.out_proj.bias, 0.)
        if self.bias_k is not None:
            xavier_normal_(self.bias_k)
        if self.bias_v is not None:
            xavier_normal_(self.bias_v)

    def __setstate__(self, state):
        # Support loading old MultiheadAttention checkpoints generated by v1.1.0
        if '_qkv_same_embed_dim' not in state:
            state['_qkv_same_embed_dim'] = True

        super(MultiheadAttention, self).__setstate__(state)

    def forward(self, query: Tensor, key: Tensor, value: Tensor, key_padding_mask: Optional[Tensor] = None,
                need_weights: bool = True, attn_mask: Optional[Tensor] = None) -> Tuple[Tensor, Optional[Tensor]]:
        r"""
    Args:
        query: Query embeddings of shape :math:`(L, N, E_q)` when ``batch_first=False`` or :math:`(N, L, E_q)`
            when ``batch_first=True``, where :math:`L` is the target sequence length, :math:`N` is the batch size,
            and :math:`E_q` is the query embedding dimension ``embed_dim``. Queries are compared against
            key-value pairs to produce the output. See "Attention Is All You Need" for more details.
        key: Key embeddings of shape :math:`(S, N, E_k)` when ``batch_first=False`` or :math:`(N, S, E_k)` when
            ``batch_first=True``, where :math:`S` is the source sequence length, :math:`N` is the batch size, and
            :math:`E_k` is the key embedding dimension ``kdim``. See "Attention Is All You Need" for more details.
        value: Value embeddings of shape :math:`(S, N, E_v)` when ``batch_first=False`` or :math:`(N, S, E_v)` when
            ``batch_first=True``, where :math:`S` is the source sequence length, :math:`N` is the batch size, and
            :math:`E_v` is the value embedding dimension ``vdim``. See "Attention Is All You Need" for more details.
        key_padding_mask: If specified, a mask of shape :math:`(N, S)` indicating which elements within ``key``
            to ignore for the purpose of attention (i.e. treat as "padding"). Binary and byte masks are supported.
            For a binary mask, a ``True`` value indicates that the corresponding ``key`` value will be ignored for
            the purpose of attention. For a byte mask, a non-zero value indicates that the corresponding ``key``
            value will be ignored.
        need_weights: If specified, returns ``attn_output_weights`` in addition to ``attn_outputs``.
            Default: ``True``.
        attn_mask: If specified, a 2D or 3D mask preventing attention to certain positions. Must be of shape
            :math:`(L, S)` or :math:`(N\cdot\text{num\_heads}, L, S)`, where :math:`N` is the batch size,
            :math:`L` is the target sequence length, and :math:`S` is the source sequence length. A 2D mask will be
            broadcasted across the batch while a 3D mask allows for a different mask for each entry in the batch.
            Binary, byte, and float masks are supported. For a binary mask, a ``True`` value indicates that the
            corresponding position is not allowed to attend. For a byte mask, a non-zero value indicates that the
            corresponding position is not allowed to attend. For a float mask, the mask values will be added to
            the attention weight.
    Outputs:
        - **attn_output** - Attention outputs of shape :math:`(L, N, E)` when ``batch_first=False`` or
          :math:`(N, L, E)` when ``batch_first=True``, where :math:`L` is the target sequence length, :math:`N` is
          the batch size, and :math:`E` is the embedding dimension ``embed_dim``.
        - **attn_output_weights** - Attention output weights of shape :math:`(N, L, S)`, where :math:`N` is the batch
          size, :math:`L` is the target sequence length, and :math:`S` is the source sequence length. Only returned
          when ``need_weights=True``.
        """
        if self.batch_first:
            query, key, value = [x.transpose(1, 0) for x in (query, key, value)]

        attn_output, attn_output_weights = F.multi_head_attention_forward(
            query, key, value, self.embed_dim, self.num_heads,
            self.in_proj_weight, self.in_proj_bias,
            self.bias_k, self.bias_v, self.add_zero_attn,
            self.dropout, self.out_proj.weight, self.out_proj.bias,
            training=self.training,
            key_padding_mask=key_padding_mask, need_weights=need_weights,
            attn_mask=attn_mask, use_separate_proj_weight=True,
            q_proj_weight=self.q_proj_weight, k_proj_weight=self.k_proj_weight,
            v_proj_weight=self.v_proj_weight)
        if self.batch_first:
            return attn_output.transpose(1, 0), attn_output_weights
        else:
            return attn_output, attn_output_weights



class MultiheadAttention(Module):
    """Multi-head Attention block from Transformer models."""

    def __init__(self, hidden_d, n_heads):
        super(MultiheadAttention, self).__init__()
        self.block = MultiheadAttentionUseSeparateProjWeight(hidden_d, n_heads)

    def forward(self, query, key, value, need_weights=False):
        return self.block(query=query, key=key, value=value, need_weights=need_weights)

    @staticmethod
    def complexity(cx, hidden_d, n_heads, seq_len):
        # See https://github.com/pytorch/pytorch/blob/master/torch/nn/functional.py
        h, w = cx["h"], cx["w"]
        flops, params, acts = cx["flops"], cx["params"], cx["acts"]
        # q, k, v = linear(input).chunk(3)
        flops += seq_len * (hidden_d * hidden_d * 3 + hidden_d * 3)
        params += hidden_d * hidden_d * 3 + hidden_d * 3
        acts += hidden_d * 3 * seq_len
        # attn_output_weights = torch.bmm(q, k.transpose)
        head_d = hidden_d // n_heads
        flops += n_heads * (seq_len * head_d * seq_len)
        acts += n_heads * seq_len * seq_len
        # attn_output = torch.bmm(attn_output_weights, v)
        flops += n_heads * (seq_len * seq_len * head_d)
        acts += n_heads * seq_len * head_d
        # attn_output = linear(attn_output)
        flops += seq_len * (hidden_d * hidden_d + hidden_d)
        params += hidden_d * hidden_d + hidden_d
        acts += hidden_d * seq_len
        return {"h": h, "w": w, "flops": flops, "params": params, "acts": acts}


# ---------------------------------- Miscellaneous ----------------------------------- #


def adjust_block_compatibility(ws, bs, gs):
    """Adjusts the compatibility of widths, bottlenecks, and groups."""
    assert len(ws) == len(bs) == len(gs)
    assert all(w > 0 and b > 0 and g > 0 for w, b, g in zip(ws, bs, gs))
    assert all(b < 1 or b % 1 == 0 for b in bs)
    vs = [int(max(1, w * b)) for w, b in zip(ws, bs)]
    gs = [int(min(g, v)) for g, v in zip(gs, vs)]
    ms = [np.lcm(g, int(b)) if b > 1 else g for g, b in zip(gs, bs)]
    vs = [max(m, int(round(v / m) * m)) for v, m in zip(vs, ms)]
    ws = [int(v / b) for v, b in zip(vs, bs)]
    assert all(w * b % g == 0 for w, b, g in zip(ws, bs, gs))
    return ws, bs, gs


def init_weights(m):
    """Performs ResNet-style weight initialization."""
    if isinstance(m, nn.Conv2d):
        # Note that there is no bias due to BN
        fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        m.weight.data.normal_(mean=0.0, std=np.sqrt(2.0 / fan_out))
    elif isinstance(m, nn.BatchNorm2d):
        zero_init_gamma = cfg.BN.ZERO_INIT_FINAL_GAMMA
        zero_init_gamma = hasattr(m, "final_bn") and m.final_bn and zero_init_gamma
        m.weight.data.fill_(0.0 if zero_init_gamma else 1.0)
        m.bias.data.zero_()
    elif isinstance(m, nn.Linear):
        m.weight.data.normal_(mean=0.0, std=0.01)
        m.bias.data.zero_()


def drop_connect(x, drop_ratio):
    """Drop connect (adapted from DARTS)."""
    keep_ratio = 1.0 - drop_ratio
    mask = torch.empty([x.shape[0], 1, 1, 1], dtype=x.dtype, device=x.device)
    mask.bernoulli_(keep_ratio)
    x.div_(keep_ratio)
    x.mul_(mask)
    return x
