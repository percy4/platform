import os
from pathlib import Path

import numpy as np
import scipy

from .nsd_access import NSDAccess


def main():
    # Preload coco captions
    data_path = os.environ.get(
        "PARAM_DATA_PATH",
        "/data/zx/MindBridge/data/natural-scenes-dataset",
    )
    nsda = NSDAccess(data_path)
    coco_73k = list(range(0, 73000))
    prompts_list = nsda.read_image_coco_info(coco_73k, info_type="captions")

    print("coco captions loaded.")

    return [prompts_list, data_path]


if __name__ == "__main__":
    # data = load()
    #
    # os.environ['PARAM_A'] = "param_a"
    # ret = main(data)
    pass
