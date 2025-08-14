import os
from pathlib import Path

import numpy as np
import scipy

from types import SimpleNamespace

from .models import Clipper, MindBridge, MindSingle

import ast


def count_params(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("param counts:\n{:,} total\n{:,} trainable".format(total, trainable))


def main(device):
    # Prepare CLIP
    # 初始化 CLIP 模型，并计算图像和文本特征的维度。
    clip_variant = os.environ.get("PARAM_CLIP_VARIANT", "ViT-L/14")
    norm_embs = ast.literal_eval(os.environ.get("PARAM_NORM_EMBS", "True"))
    clip_sizes = {"RN50": 1024, "ViT-L/14": 768, "ViT-B/32": 512, "ViT-H-14": 1024}
    clip_size = clip_sizes[clip_variant]

    print("Using hidden layer CLIP space (Versatile Diffusion)")
    if not norm_embs:
        print("WARNING: YOU WANT NORMED EMBEDDINGS FOR VERSATILE DIFFUSION!")
    clip_extractor = Clipper(
        clip_variant, device=device, hidden_state=True, norm_embs=norm_embs
    )

    out_dim_image = 257 * clip_size  # 257*768 = 197376
    out_dim_text = 77 * clip_size  # 77*768  = 59136

    print("clip_extractor loaded.")
    print("out_dim_image:", out_dim_image)
    print("out_dim_text:", out_dim_text)

    args = SimpleNamespace(
        clip_variant=clip_variant,
        norm_embs=norm_embs,
        clip_sizes=clip_sizes,
        clip_size=clip_size,
    )

    return [clip_extractor, out_dim_image, out_dim_text, args]


if __name__ == "__main__":
    # data = load()
    #
    # os.environ['PARAM_A'] = "param_a"
    # ret = main(data)
    pass
