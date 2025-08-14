import os
from pathlib import Path

import numpy as np
import scipy

import torch
from accelerate import Accelerator, DeepSpeedPlugin

import random
import sys

from types import SimpleNamespace


from .nsd_access import NSDAccess

from .trainer import *


def config_multi_cpu():
    # Multi-GPU config
    accelerator = Accelerator(
        split_batches=False,
        mixed_precision="no",
        cpu=True,  # 强制 CPU
        device_placement=True,  # 自动放置到 CPU 上
    )
    accelerator.print("PID of this process =", os.getpid())
    device = accelerator.device
    accelerator.print("device:", device)
    num_devices = torch.cuda.device_count()
    if num_devices == 0:
        num_devices = 1
    accelerator.print(accelerator.state)
    local_rank = accelerator.state.local_process_index
    world_size = accelerator.state.num_processes
    distributed = not accelerator.state.distributed_type == "NO"
    accelerator.print(
        "distributed =",
        distributed,
        "num_devices =",
        num_devices,
        "local rank =",
        local_rank,
        "world size =",
        world_size,
    )

    return accelerator, device, local_rank


def seed_everything(seed=0, cudnn_deterministic=True):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if cudnn_deterministic:
        torch.backends.cudnn.deterministic = True
    else:
        ## needs to be False to use conv3D
        print("Note: not using cudnn.deterministic")


def main():
    args = {
        "seed": int(os.environ.get("PARAM_SEED", "42")),
        "max_lr": float(os.environ.get("PARAM_MAX_LR", "0.0003")),
    }

    args = SimpleNamespace(**args)

    accelerator, device, local_rank = config_multi_cpu()
    if local_rank != 0:  # suppress print for non-local_rank=0
        sys.stdout = open(os.devnull, "w")

    # need non-deterministic CuDNN for conv3D to work
    seed_everything(args.seed, cudnn_deterministic=False)

    # learning rate will be changed by "acclerate" based on number of processes(GPUs)
    args.max_lr *= accelerator.num_processes

    return [accelerator, device, local_rank, args]


if __name__ == "__main__":
    # data = load()
    #
    # os.environ['PARAM_A'] = "param_a"
    # ret = main(data)
    pass
