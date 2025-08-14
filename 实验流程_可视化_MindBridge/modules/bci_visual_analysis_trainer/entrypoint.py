import os
from pathlib import Path

import numpy as np
import scipy

from types import SimpleNamespace

from .trainer import *
import ast


def main(
    accelerator,
    voxel2clip,
    clip_extractor,
    prompts_list,
    device,
    local_rank,
    data_path,
    model_config,
    mapper_config,
):
    args = SimpleNamespace(**vars(model_config), **vars(mapper_config))

    args.data_path = data_path

    args.model_name = os.environ.get("PARAM_MODEL_NAME", "mindbridge_subj1")
    args.wandb_mode = os.environ.get("WANDB_MODE", "offline")
    args.batch_size = int(os.environ.get("PARAM_BATCH_SIZE", "20"))
    args.val_batch_size = int(os.environ.get("PARAM_VAL_BATCH_SIZE", "20"))
    args.wandb_log = ast.literal_eval(os.environ.get("PARAM_WANDB_LOG", "True"))
    args.wandb_project = os.environ.get("PARAM_WANDB_PROJECT", "MindBridge")
    args.resume = ast.literal_eval(os.environ.get("PARAM_RESUME", "False"))
    args.resume_id = os.environ.get("PARAM_RESUME_ID", "")
    args.load_from = os.environ.get("PARAM_LOAD_FROM", "")
    args.num_workers = int(os.environ.get("PARAM_NUM_WORKERS", "2"))
    args.max_lr = float(os.environ.get("PARAM_MAX_LR", "0.0003"))
    args.pool_type = os.environ.get("PARAM_POOL_TYPE", "max")
    args.lr_scheduler_type = os.environ.get("PARAM_LR_SCHEDULER_TYPE", "cycle")
    args.length = int(os.environ.get("PARAM_LENGTH", "77"))

    args.use_image_aug = ast.literal_eval(os.environ.get("PARAM_USE_IMAGE_AUG", "True"))
    args.ckpt_interval = int(os.environ.get("PARAM_CKPT_INTERVAL", "100"))
    args.eval_interval = int(os.environ.get("PARAM_EVAL_INTERVAL", "100"))
    args.mse_mult = float(os.environ.get("PARAM_MSE_MULT", "1000.0"))
    args.rec_mult = float(os.environ.get("PARAM_REC_MULT", "0.0"))
    args.cyc_mult = float(os.environ.get("PARAM_CYC_MULT", "0.0"))
    args.autoencoder_name = os.environ.get("PARAM_AUTOENCODER_NAME", None)
    args.subj_load = os.environ.get("PARAM_SUBJ_LOAD", None)
    args.subj_test = int(os.environ.get("PARAM_SUBJ_TEST", "1"))
    args.samples = os.environ.get("PARAM_SAMPLES", None)
    args.img2img_strength = float(os.environ.get("PARAM_IMG2IMG_STRENGTH", "0.85"))
    args.guidance_scale = float(os.environ.get("PARAM_GUIDANCE_SCALE", "3.5"))
    args.num_inference_steps = int(os.environ.get("PARAM_NUM_INFERENCE_STEPS", "20"))
    args.recons_per_sample = int(os.environ.get("PARAM_RECONS_PER_SAMPLE", "16"))
    args.plotting = ast.literal_eval(os.environ.get("PARAM_PLOTTING", "True"))
    args.vd_cache_dir = os.environ.get("PARAM_VD_CACHE_DIR", "../weights")
    args.ckpt_from = os.environ.get("PARAM_CKPT_FROM", "last")
    args.text_image_ratio = float(os.environ.get("PARAM_TEXT_IMAGE_RATIO", "0.5"))
    args.test_start = int(os.environ.get("PARAM_TEST_START", "0"))
    args.test_end = os.environ.get("PARAM_TEST_END", None)  # 可以为 None

    if args.adapting:
        print("trainer 0")
        trainer = Trainer_adapt(
            args, accelerator, voxel2clip, clip_extractor, prompts_list, device
        )
    elif len(args.subj_list) == 1:
        print("trainer 1")
        trainer = Trainer_single(
            args, accelerator, voxel2clip, clip_extractor, prompts_list, device
        )
    else:
        print("trainer 2")
        trainer = Trainer_bridge(
            args, accelerator, voxel2clip, clip_extractor, prompts_list, device
        )

    ## Weights and Biases
    if local_rank == 0 and args.wandb_log:  # only use main process for wandb logging
        import wandb

        wandb_run = args.model_name
        wandb_notes = ""

        print(f"Wandb project {args.wandb_project} run {wandb_run}")
        if args.wandb_mode == "online":
            wandb.login(host="https://api.wandb.ai")
        else:
            os.environ["WANDB_MODE"] = "disabled"

        wandb_config = vars(args)
        print("wandb_config:\n", wandb_config)
        if args.resume:  # wandb_auto_resume
            if args.resume_id is None:
                args.resume_id = args.model_name
            print("wandb_id:", args.resume_id)
            wandb.init(
                id=args.resume_id,
                project=args.wandb_project,
                name=wandb_run,
                config=wandb_config,
                notes=wandb_notes,
                resume="allow",
            )
        else:
            wandb.init(
                project=args.wandb_project,
                name=wandb_run,
                config=wandb_config,
                notes=wandb_notes,
            )

    trainer.train(local_rank)

    return []


if __name__ == "__main__":
    # data = load()
    #
    # os.environ['PARAM_A'] = "param_a"
    # ret = main(data)
    pass
