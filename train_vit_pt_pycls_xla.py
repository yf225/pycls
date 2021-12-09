#!/usr/bin/env python3

# On Cloud TPU node (use alpha!!!), run
"""
pip install iopath yacs submitit

export XRT_TPU_CONFIG="localservice;0;localhost:51011"
cd $HOME

rm -rf ./pytorch-image-models || true
git clone https://github.com/yf225/pytorch-image-models.git -b vit_dummy_data
export PYTHONPATH=$HOME/pytorch-image-models:${PYTHONPATH}

rm -rf ./pycls || true
git clone https://github.com/yf225/pycls.git -b vit_dummy_data
cd pycls && git pull

python3 train_vit_pt_pycls_xla.py --bits=16 --micro_batch_size=224
"""

import argparse
import os
import sys
import time
import statistics

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch_xla
from torch_xla.distributed import parallel_loader as pl
from torch_xla.distributed import xla_multiprocessing as xmp
from custom_vit_model import ViT

# !rm -rf ./pytorch-image-models || true
# !git clone https://github.com/yf225/pytorch-image-models.git -b vit_dummy_data
# !cd pytorch-image-models && git pull

import sys
if './pytorch-image-models' not in sys.path:
  sys.path.append('./pytorch-image-models')

DEBUG = False
VERBOSE = False

num_attention_heads = 16
hidden_size = 1280
num_layers = 32

image_size = 224
patch_size = 16  # Size of the patches to be extract from the input images

num_classes = 1000
num_epochs = 3

if "CUDA_VISIBLE_DEVICES" in os.environ:
  num_devices = len(os.environ["CUDA_VISIBLE_DEVICES"].split(","))
else:
  num_devices = 8

# if 'COLAB_TPU_ADDR' in os.environ:  # Colab, meaning debug mode
#   DEBUG = True

if DEBUG:
  print("Overwriting hyperparams since we are in debug mode...")
  num_attention_heads = 1
  hidden_size = 128
  num_layers = 1
  bits = 16
  micro_batch_size = 1
else:
  parser = argparse.ArgumentParser()
  parser.add_argument("--bits", type=int)
  parser.add_argument("--micro_batch_size", type=int)
  args = parser.parse_args()
  bits = args.bits
  micro_batch_size = args.micro_batch_size

global_batch_size = micro_batch_size * num_devices

assert bits in [16, 32]
if bits == 16:
  default_dtype = torch.bfloat16
elif bits == 32:
  default_dtype = torch.float32

def xm_master_print_if_verbose(message):
  if VERBOSE:
    torch_xla.core.xla_model.master_print(message, flush=True)

class VitDummyDataset(torch.utils.data.Dataset):
  def __init__(self, dataset_size, num_classes):
    self.dataset_size = dataset_size
    self.num_classes = num_classes

  def __len__(self):
    return self.dataset_size

  def __getitem__(self, index):
    return (torch.rand(3, image_size, image_size), torch.randint(self.num_classes, (1,)).item())


def create_dataloader(dataset):
  sampler = torch.utils.data.distributed.DistributedSampler(
    dataset,
    num_replicas=torch_xla.core.xla_model.xrt_world_size(),
    rank=torch_xla.core.xla_model.get_ordinal(),
  )
  return torch.utils.data.DataLoader(
    dataset,
    batch_size=micro_batch_size,  # NOTE: this should be batch size per TPU core, re. https://discuss.pytorch.org/t/72769/2
    sampler=sampler,
    num_workers=1,
  )

def train_vit():
  assert torch_xla.core.xla_model.xrt_world_size() == num_devices
  torch_xla.core.xla_model.master_print("Working on: bits: {}, global_batch_size: {}, micro_batch_size: {}".format(bits, global_batch_size, micro_batch_size))
  # create train dataset
  train_dataset = VitDummyDataset(micro_batch_size * torch_xla.core.xla_model.xrt_world_size() * 10, num_classes)
  train_loader = create_dataloader(train_dataset)
  debug_train_loader = create_dataloader(train_dataset)
  sample_batch = next(iter(debug_train_loader))
  print("sample_batch[0].shape: ", sample_batch[0].shape)
  assert list(sample_batch[0].shape) == [micro_batch_size, 3, image_size, image_size]

  torch.manual_seed(42)

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

  device = torch_xla.core.xla_model.xla_device()
  model = model.to(device).train()
  optim_cls = optim.Adam
  optimizer = optim_cls(
      model.parameters(),
      lr=0.001,
  )
  loss_fn = nn.CrossEntropyLoss()

  step_duration_list = []

  def train_loop_fn(loader, epoch):
    model.train()
    step_start_time = time.time()
    for step, (data, target) in enumerate(loader):
      optimizer.zero_grad()
      output = model(data)
      loss = loss_fn(output, target)
      loss.backward()
      # Note: optimizer_step uses the implicit Cloud TPU context to
      #  coordinate and synchronize gradient updates across processes.
      #  This means that each process's network has the same weights after
      #  this is called.
      # Warning: this coordination requires the actions performed in each
      #  process are the same. In more technical terms, the graph that
      #  PyTorch/XLA generates must be the same across processes.
      torch_xla.core.xla_model.optimizer_step(optimizer)  # Note: barrier=True not needed when using ParallelLoader
      step_duration = time.time() - step_start_time
      step_duration_list.append(step_duration)
      xm_master_print_if_verbose("Step {}, time taken: {}".format(step, step_duration))
      step_start_time = time.time()

  train_device_loader = pl.MpDeviceLoader(train_loader, device)
  for epoch in range(1, num_epochs + 1):
    xm_master_print_if_verbose('Epoch {} train begin'.format(epoch))
    train_loop_fn(train_device_loader, epoch)

  torch_xla.core.xla_model.master_print("bits: {}, global_batch_size: {}, micro_batch_size: {}, median step duration: {:.3f}".format(bits, global_batch_size, micro_batch_size, statistics.median(step_duration_list)))

# "Map function": acquires a corresponding Cloud TPU core, creates a tensor on it,
# and prints its core
def map_fn(index, flags):
  # Sets a common random seed - both for initialization and ensuring graph is the same
  torch.manual_seed(42)

  # Acquires the (unique) Cloud TPU core corresponding to this process's index
  device = torch_xla.core.xla_model.xla_device()
  if VERBOSE:
    print("Process", index ,"is using", torch_xla.core.xla_model.xla_real_devices([str(device)])[0])

  # # Barrier to prevent master from exiting before workers connect.
  # torch_xla.core.xla_model.rendezvous('init')

  torch.set_default_dtype(default_dtype)
  train_vit()

# Spawns eight of the map functions, one for each of the eight cores on
# the Cloud TPU
flags = {}

if 'COLAB_TPU_ADDR' in os.environ:
  # Note: Colab only supports start_method='fork'
  xmp.spawn(map_fn, args=(flags,), nprocs=num_devices, start_method='fork')

if __name__ == "__main__":
  xmp.spawn(map_fn, args=(flags,), nprocs=num_devices, start_method='fork')
