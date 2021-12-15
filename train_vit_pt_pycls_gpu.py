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

export PATH=/usr/local/cuda-11.1/bin:${PATH}
export LD_LIBRARY_PATH=/usr/local/cuda-11.1/lib64:/usr/local/cuda-11.1/extras/CUPTI/lib64:${LD_LIBRARY_PATH}

CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 \
train_vit_pt_pycls_gpu.py --mode=eager --micro_batch_size=64

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

from timm.data import create_dataset, create_loader
from timm.utils import *
from timm.loss import *
from timm.optim import create_optimizer_v2
from custom_vit_model import create_vit_model

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
    def __init__(self, dataset_size, image_size, num_classes):
        self.dataset_size = dataset_size
        self.image_size = image_size
        self.num_classes = num_classes

    def __len__(self):
        return self.dataset_size

    def __getitem__(self, index):
        return (torch.zeros(self.image_size, self.image_size, 3).to(torch.half), torch.zeros(1).to(torch.long))

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

    model = create_vit_model(
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

    optimizer = create_optimizer_v2(model, 'adam', lr=0.)

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
        input_size=(image_size, image_size, 3),
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
    assert list(sample_batch[0].shape) == [args.micro_batch_size, image_size, image_size, 3]

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
