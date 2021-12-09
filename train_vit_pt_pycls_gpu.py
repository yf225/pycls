#!/usr/bin/env python3

"""On AWS GPU node
conda activate torch-1.10

cd /fsx/users/willfeng/repos

rm -rf ./pytorch-image-models || true
git clone https://github.com/yf225/pytorch-image-models.git -b vit_dummy_data
export PYTHONPATH=/fsx/users/willfeng/repos/pytorch-image-models:${PYTHONPATH}

rm -rf ./pycls || true
git clone https://github.com/yf225/pycls.git -b vit_dummy_data
cd pycls && git pull

CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 \
train_vit_pt_pycls_gpu.py --mode=eager --micro_batch_size=4

CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 \
train_vit_pt_pycls_gpu.py --mode=graph --micro_batch_size=4
"""
import argparse
import time
import os
import logging
import statistics
from collections import OrderedDict
from contextlib import suppress
from datetime import datetime

import torch
import torch.nn as nn
import torchvision.utils
from torch.nn.parallel import DistributedDataParallel as NativeDDP
import einops

from timm.data import create_dataset, create_loader, resolve_data_config, Mixup, FastCollateMixup, AugMixDataset
from timm.models import create_model, safe_model_name, resume_checkpoint, load_checkpoint,\
    convert_splitbn_model, model_parameters
from timm.utils import *
from timm.loss import *
from timm.optim import create_optimizer_v2, optimizer_kwargs
from timm.scheduler import create_scheduler
from timm.models.helpers import build_model_with_cfg
from timm.models.vision_transformer import VisionTransformer

torch.backends.cudnn.benchmark = True


# Hyperparams

should_profile = False
VERBOSE = False
num_attention_heads = 16
hidden_size = 1280
num_layers = 32

image_size = 224
patch_size = 16  # Size of the patches to be extract from the input images

num_classes = 1000
num_epochs = 3


parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')

# Batch size
parser.add_argument("--micro_batch_size", default=32, type=int)

# Misc
parser.add_argument('--channels-last', action='store_true', default=False,
                    help='Use channels_last memory layout')
parser.add_argument("--local_rank", default=0, type=int)
parser.add_argument('--mode', type=str,
                    help='"eager" or "graph"')
parser.add_argument('--log-interval', type=int, default=1, metavar='N',
                    help='how many batches to wait before logging training status')

def print_if_verbose(msg):
    if VERBOSE:
        print(msg, flush=True)

class VitDummyDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_size, crop_size, num_classes):
        self.dataset_size = dataset_size
        self.crop_size = crop_size
        self.num_classes = num_classes

    def __len__(self):
        return self.dataset_size

    def __getitem__(self, index):
        return (torch.rand(3, self.crop_size, self.crop_size).to(torch.half), torch.randint(self.num_classes, (1,)).to(torch.long))


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
        self.self_attention = MultiheadAttentionSeparateProjection(hidden_d, n_heads)
        self.ln_2 = layernorm(hidden_d)
        self.mlp_block = MLPBlock(hidden_d, mlp_d)

    def forward(self, x):
        x_p = self.ln_1(x)
        x_p, _ = self.self_attention(x_p, x_p, x_p)
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


class ViTStemPatchify(Module):
    """The patchify vision transformer stem as per https://arxiv.org/abs/2010.11929."""

    def __init__(self, w_in, w_out, k):
        super(ViTStemPatchify, self).__init__()
        self.patchify = patchify2d(w_in, w_out, k, bias=True)

    def forward(self, x):
        return self.patchify(x)

import warnings
from typing import Optional, Tuple

import torch
from torch import Tensor
from torch.nn.modules.linear import NonDynamicallyQuantizableLinear
from torch.nn.init import constant_, xavier_normal_, xavier_uniform_
from torch.nn.parameter import Parameter
from torch.nn import functional as F

class MultiheadAttentionSeparateProjection(torch.nn.Module):
    __constants__ = ['batch_first']
    bias_k: Optional[torch.Tensor]
    bias_v: Optional[torch.Tensor]

    def __init__(self, embed_dim, num_heads, dropout=0., bias=True, add_bias_kv=False, add_zero_attn=False,
                 kdim=None, vdim=None, batch_first=False, device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(MultiheadAttentionSeparateProjection, self).__init__()
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

    def forward(self, query: Tensor, key: Tensor, value: Tensor, key_padding_mask: Optional[Tensor] = None,
                need_weights: bool = True, attn_mask: Optional[Tensor] = None) -> Tuple[Tensor, Optional[Tensor]]:
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
        self.stem = ViTStemPatchify(3, p["hidden_d"], p["patch_size"])
        seq_len = (p["image_size"] // cfg.VIT.PATCH_SIZE) ** 2
        self.class_token = None
        self.pos_embedding = Parameter(torch.zeros(seq_len, 1, p["hidden_d"]))
        self.encoder = ViTEncoder(
            p["n_layers"], p["hidden_d"], p["n_heads"], p["mlp_d"]
        )
        self.head = ViTHead(p["hidden_d"], p["num_classes"])

    def forward(self, x):
        # (n, c, h, w) -> (n, hidden_d, n_h, n_w)
        x = self.stem(x)
        # (n, hidden_d, n_h, n_w) -> (n, hidden_d, (n_h * n_w))
        x = x.reshape(x.size(0), x.size(1), -1)
        # (n, hidden_d, (n_h * n_w)) -> ((n_h * n_w), n, hidden_d)
        x = x.permute(2, 0, 1)
        x = x + self.pos_embedding
        x = self.encoder(x)
        x = x[0, :, :]
        return self.head(x)


step_duration_list = []

def main():
    global should_profile

    args = parser.parse_args()

    args.distributed = False
    if 'WORLD_SIZE' in os.environ:
        args.distributed = int(os.environ['WORLD_SIZE']) > 1
    args.device = 'cuda:0'
    args.world_size = 1
    args.num_devices = 1
    args.rank = 0  # global rank
    if args.distributed:
        args.device = 'cuda:%d' % args.local_rank
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(backend='nccl', init_method='env://')
        args.world_size = torch.distributed.get_world_size()
        args.num_devices = torch.distributed.get_world_size()
        args.rank = torch.distributed.get_rank()
        print_if_verbose('Training in distributed mode with multiple processes, 1 GPU per process. Process %d, total %d.'
                     % (args.rank, args.world_size))
    else:
        print_if_verbose('Training with a single process on 1 GPUs.')
    assert args.rank >= 0

    random_seed(42, args.rank)

    model = ViT(
        {
            "image_size": image_size,
            "patch_size": patch_size,
            "n_layers": num_layers,
            "n_heads": num_attention_heads,
            "hidden_d": hidden_size,
            "mlp_d": 4 * hidden_size,
            "num_classes": num_classes,
        }
    )

    if args.local_rank == 0:
        print_if_verbose(
            f'Model created, param count:{sum([m.numel() for m in model.parameters()])}')

    # move model to GPU, enable channels last layout if set
    model = model.to(torch.half)
    if args.channels_last:
        model = model.to(memory_format=torch.channels_last)

    assert args.mode in ["eager", "graph"]
    if args.mode == "graph":
        model = torch.jit.script(torch.fx.symbolic_trace(model))

    model = model.cuda()

    optimizer = create_optimizer_v2(model, 'adam', lr=1e-6)

    # setup distributed training
    if args.distributed:
        model = NativeDDP(model, device_ids=[args.local_rank])

    start_epoch = 0

    if args.local_rank == 0:
        print_if_verbose('Scheduled epochs: {}'.format(num_epochs))

    # create train dataset
    dataset_train = VitDummyDataset(args.micro_batch_size * args.num_devices * 10, image_size, num_classes)
    loader_train = create_loader(
        dataset_train,
        input_size=(3, 224, 224),
        batch_size=args.micro_batch_size,  # NOTE: this should be batch size per GPU, re. https://discuss.pytorch.org/t/72769/2
        is_training=True,
        no_aug=True,
        fp16=True,
        distributed=args.distributed,
    )

    sample_batch = next(iter(torch.utils.data.DataLoader(
        dataset_train,
        batch_size=args.micro_batch_size,  # NOTE: this should be batch size per TPU core, re. https://discuss.pytorch.org/t/72769/2
        sampler=torch.utils.data.distributed.DistributedSampler(
            dataset_train,
            num_replicas=args.world_size,
            rank=args.local_rank,
        ),
        num_workers=1,
    )))
    print("sample_batch[0].shape: ", sample_batch[0].shape)
    assert list(sample_batch[0].shape) == [args.micro_batch_size, 3, image_size, image_size]

    # setup loss function
    train_loss_fn = nn.CrossEntropyLoss().to(torch.half)
    train_loss_fn = train_loss_fn.cuda()

    try:
        from fvcore.nn import FlopCountAnalysis
        from fvcore.nn import flop_count_table
        if args.local_rank == 0:
            flops = FlopCountAnalysis(model, sample_batch[0].to("cuda:0"))
            print(flop_count_table(flops))

        for epoch in range(start_epoch, num_epochs):
            if should_profile and args.local_rank == 0:
                def recorder_enter_hook(module, input):
                    module._torch_profiler_recorder = torch.autograd.profiler.record_function(str(module.__class__))
                    module._torch_profiler_recorder.__enter__()

                def recorder_exit_hook(module, input, output):
                    module._torch_profiler_recorder.__exit__(None, None, None)

                torch.nn.modules.module.register_module_forward_pre_hook(recorder_enter_hook)
                torch.nn.modules.module.register_module_forward_hook(recorder_exit_hook)

                prof = torch.profiler.profile(
                    activities=[
                        torch.profiler.ProfilerActivity.CPU,
                        torch.profiler.ProfilerActivity.CUDA,
                    ]
                )
                prof.__enter__()

            train_metrics = train_one_epoch(
                epoch, model, loader_train, optimizer, train_loss_fn, args)

            if should_profile and args.local_rank == 0:
                prof.__exit__(None, None, None)
                trace_dir_path = "train_vit_pt_pycls_gpu_trace"
                if not os.path.isdir(trace_dir_path):
                    os.mkdir(trace_dir_path)
                prof.export_chrome_trace(os.path.join(trace_dir_path, "trace_{}_{}_{}.json".format(str(int(time.time())), args.num_devices, args.local_rank)))
                should_profile = False  # NOTE: only profile one epoch
        if args.local_rank == 0:
            print("micro_batch_size: {}, median step duration: {:.3f}".format(args.micro_batch_size, statistics.median(step_duration_list)))
    except KeyboardInterrupt:
        pass


def train_one_epoch(epoch, model, loader, optimizer, loss_fn, args):

    second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
    batch_time_m = AverageMeter()
    data_time_m = AverageMeter()

    model.train()

    end = time.time()
    last_idx = len(loader) - 1
    num_updates = epoch * len(loader)
    for batch_idx, (input, target) in enumerate(loader):
        last_batch = batch_idx == last_idx
        data_time_m.update(time.time() - end)
        if args.channels_last:
            input = input.contiguous(memory_format=torch.channels_last)

        with torch.autograd.profiler.record_function("### forward ###"):
            output = model(input)
            loss = loss_fn(output, target)

        with torch.autograd.profiler.record_function("### zero_grad ###"):
            optimizer.zero_grad()

        with torch.autograd.profiler.record_function("### backward ###"):
            loss.backward(create_graph=second_order)

        with torch.autograd.profiler.record_function("### optimizer step ###"):
            optimizer.step()

        torch.cuda.synchronize()
        num_updates += 1
        batch_time = time.time() - end
        batch_time_m.update(batch_time)
        step_duration_list.append(batch_time)
        if last_batch or batch_idx % args.log_interval == 0:
            lrl = [param_group['lr'] for param_group in optimizer.param_groups]
            lr = sum(lrl) / len(lrl)

            if args.local_rank == 0:
                print_if_verbose(
                    'Train: {} [{:>4d}/{} ({:>3.0f}%)]  '
                    'Time: {batch_time.val:.3f}s, {rate:>7.2f}/s  '
                    '({batch_time.avg:.3f}s, {rate_avg:>7.2f}/s)  '
                    'LR: {lr:.3e}  '
                    'Data: {data_time.val:.3f} ({data_time.avg:.3f})'.format(
                        epoch,
                        batch_idx, len(loader),
                        100. * batch_idx / last_idx,
                        batch_time=batch_time_m,
                        rate=input.size(0) * args.world_size / batch_time_m.val,
                        rate_avg=input.size(0) * args.world_size / batch_time_m.avg,
                        lr=lr,
                        data_time=data_time_m))

        end = time.time()
        # end for

    return OrderedDict([('loss', -1)])


if __name__ == '__main__':
    main()
