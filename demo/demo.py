"""
This file may have been modified by Bytedance Ltd. and/or its affiliates (“Bytedance's Modifications”).
All Bytedance's Modifications are Copyright (year) Bytedance Ltd. and/or its affiliates. 

Reference: https://github.com/facebookresearch/Mask2Former/blob/main/demo/demo.py
"""

import argparse
import glob
import multiprocessing as mp
import os

# fmt: off
import sys
sys.path.insert(1, os.path.join(sys.path[0], '..'))
# fmt: on

import tempfile
import time
import warnings

import cv2
import numpy as np
import tqdm

from detectron2.config import get_cfg
from detectron2.data.detection_utils import read_image
from detectron2.projects.deeplab import add_deeplab_config
from detectron2.utils.logger import setup_logger

from maft import add_maskformer2_config, add_fcclip_config
from predictor import VisualizationDemo

import os
import json
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.colors as colors
import matplotlib.patches as Polygon
import matplotlib.collections as PatchCollection
import seaborn as sns

from random import shuffle
from PIL import Image

from pycocotools.coco import COCO

# Set up paths
dataDir = "./data/coco/"
dataType='val2017'
imageDir = '{}{}/'.format(dataDir, dataType)

# constants
WINDOW_NAME = "maft demo"


def setup_cfg(args):
    # load config from file and command-line arguments
    cfg = get_cfg()
    add_deeplab_config(cfg)
    add_maskformer2_config(cfg)
    add_fcclip_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.merge_from_list(['MODEL.META_ARCHITECTURE', "MAFT_Plus_DEMO"])
    cfg.freeze()
    return cfg


def get_parser():
    parser = argparse.ArgumentParser(description="fcclip demo for builtin configs")
    parser.add_argument(
        "--config-file",
        default="configs/semantic/demo.yaml",
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument("--webcam", action="store_true", help="Take inputs from webcam.")
    parser.add_argument("--video-input", help="Path to video file.")
    parser.add_argument(
        "--input",
        nargs="+",
        help="A list of space separated input images; "
        "or a single glob pattern such as 'directory/*.jpg'",
    )
    parser.add_argument(
        "--output",
        default="./output",
        help="A file or directory to save output visualizations. "
        "If none, will show output in an OpenCV window.",
    )

    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.5,
        help="Minimum score for instance predictions to be shown",
    )
    parser.add_argument(
        "--opts",
        help="Modify config options using the command-line 'KEY VALUE' pairs",
        default=[],
        nargs=argparse.REMAINDER,
    )
    return parser


def test_opencv_video_format(codec, file_ext):
    with tempfile.TemporaryDirectory(prefix="video_format_test") as dir:
        filename = os.path.join(dir, "test_file" + file_ext)
        writer = cv2.VideoWriter(
            filename=filename,
            fourcc=cv2.VideoWriter_fourcc(*codec),
            fps=float(30),
            frameSize=(10, 10),
            isColor=True,
        )
        [writer.write(np.zeros((10, 10, 3), np.uint8)) for _ in range(30)]
        writer.release()
        if os.path.isfile(filename):
            return True
        return False


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    args = get_parser().parse_args()
    setup_logger(name="fvcore")
    logger = setup_logger()
    logger.info("Arguments: " + str(args))

    cfg = setup_cfg(args)

    demo = VisualizationDemo(cfg)
    
    # ---- Single image test for image_id = 494869 ----
    image_id = 494869
    json_path = os.path.join(dataDir, "custom_coco_all.json")

    # Load JSON annotations
    with open(json_path, "r") as f:
        coco_data = json.load(f)

    # Find the corresponding image entry
    img_entry = coco_data[str(image_id)]
    file_name = img_entry["file_name"]
    command = img_entry["command"]
    image_path = os.path.join(imageDir, file_name)

    # Load image
    image = read_image(image_path, format="BGR")

    # Run inference and visualization
    user_classes = [command]
    predictions, vis_output, pooled_map = demo.run_on_image(image, user_classes)

    # Log the prediction
    logger.info(
        "{}: {} in {:.2f}s".format(
            image_path,
            "detected {} instances".format(len(predictions["instances"]))
            if "instances" in predictions
            else "finished",
            0.0,  # timing optional
        )
    )

    # Save or display results
    filename = os.path.splitext(os.path.basename(image_path))[0]
    output_path = os.path.join(args.output, f"{filename}_vis.jpg")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    vis_output.save(output_path)
    logger.info(f"Saved visualized output to: {output_path}")

    # Print pooled 8×8 score map
    if pooled_map is not None:
        print("8x8 pooled attention map:")
        print(np.round(pooled_map, 3))
