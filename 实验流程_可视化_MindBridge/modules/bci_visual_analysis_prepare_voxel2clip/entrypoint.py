import os
from pathlib import Path

import numpy as np
import scipy

from . import utils

from .models import Clipper, MindBridge, MindSingle

from types import SimpleNamespace
import ast


def main(out_dim_image, out_dim_text, device):
    subj_list = ast.literal_eval(os.environ.get("PARAM_SUBJ_LIST", "[1]"))
    subj_target = ast.literal_eval(os.environ.get("PARAM_SUBJ_TARGET", "[1]"))
    subj_source = ast.literal_eval(os.environ.get("PARAM_SUBJ_SOURCE", "[1]"))
    adapting = ast.literal_eval(os.environ.get("PARAM_ADAPTING", "False"))
    h_size = int(os.environ.get("PARAM_H_SIZE", 512))
    n_blocks = int(os.environ.get("PARAM_N_BLOCKS", 2))
    pool_num = int(os.environ.get("PARAM_POOL_NUM", 1024))

    # Prepare voxel2clip
    if adapting:
        # 如果 adapting 为 True，说明模型需要进行适配（可能是迁移学习或微调）。
        # 此时，将 subj_target 添加到 subj_source 中，更新 subj_list。
        subj_list = subj_source + [subj_target]

    voxel2clip_kwargs = dict(
        in_dim=pool_num,
        out_dim_image=out_dim_image,
        out_dim_text=out_dim_text,
        h=h_size,
        n_blocks=n_blocks,
        subj_list=subj_list,
        adapting=adapting,
    )
    if len(subj_list) == 1:  # Single subject does not need "brain builder"
        voxel2clip_kwargs.pop("adapting")  # Single subject does not need "adapting"
        voxel2clip = MindSingle(**voxel2clip_kwargs).to(device)
    else:
        voxel2clip = MindBridge(**voxel2clip_kwargs).to(device)

    if adapting:  # reset-tuning
        # Only let the parameters of embedder and builder in the voxel2clip trainable, keeping other parameters frozen
        # 如果 adapting 为 True，则：
        # 将 voxel2clip 中的所有参数设置为不可训练（冻结）。
        # 仅允许 embedder 和 builder 中与目标主题（subj_target）相关的参数可训练，用于微调。
        voxel2clip.requires_grad_(False)
        voxel2clip.embedder[str(subj_target)].requires_grad_(True)
        voxel2clip.builder[str(subj_target)].requires_grad_(True)

    print("voxel2clip loaded.")
    print("params of voxel2clip:")
    utils.count_params(voxel2clip)

    args = SimpleNamespace(
        subj_list=subj_list,
        subj_target=subj_target,
        subj_source=subj_source,
        adapting=adapting,
        h_size=h_size,
        n_blocks=n_blocks,
        pool_num=pool_num,
    )

    return [voxel2clip, args]


if __name__ == "__main__":
    # data = load()
    #
    # os.environ['PARAM_A'] = "param_a"
    # ret = main(data)
    pass
