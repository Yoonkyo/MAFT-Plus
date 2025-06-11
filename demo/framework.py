"""
This file may have been modified by Bytedance Ltd. and/or its affiliates (“Bytedance's Modifications”).
All Bytedance's Modifications are Copyright (year) Bytedance Ltd. and/or its affiliates. 

Reference: https://github.com/facebookresearch/Mask2Former/blob/main/demo/demo.py
"""

import argparse
import glob
import multiprocessing as mp
import os
import json

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
import torch
import time

from detectron2.config import get_cfg
from detectron2.data.detection_utils import read_image
from detectron2.projects.deeplab import add_deeplab_config
from detectron2.utils.logger import setup_logger

from maft import add_maskformer2_config, add_fcclip_config
from predictor import VisualizationDemo
import pdb
import torchvision.transforms.functional as F2
import torch.nn.functional as F
import torch.nn as nn
from torch import nn
from torchvision import transforms
import matplotlib.pyplot as plt


# constants
WINDOW_NAME = "maft demo"

def transform_1(img):
    #img = transforms.Resize(img_size)(img)
    img = transforms.ToTensor()(img)
    return img

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
    
class CNNEncoder(nn.Module):
    def __init__(self, input_channels=3, patch_size=8, encoded_size=24):
        super(CNNEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(input_channels, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),  # Batch Normalization
            nn.ReLU(),
            
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),  # Batch Normalization
            nn.ReLU(),
            
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),  # Batch Normalization
            nn.ReLU(),
            
            nn.Flatten(),  # Flatten to (batch, 256 * 8 * 8)
            nn.Linear(256 * patch_size * patch_size, 512),
            nn.ReLU(),
            nn.Linear(512, encoded_size)  # Output: (batch, 24)
        )

    def forward(self, x):
        return self.encoder(x)

class CNNDecoder(nn.Module):
    def __init__(self, encoded_size=24, output_channels=3, patch_size=8):
        super(CNNDecoder, self).__init__()
        self.decoder = nn.Sequential(
            nn.Linear(encoded_size, 512),
            nn.ReLU(),
            
            nn.Linear(512, 256 * patch_size * patch_size),
            nn.ReLU(),
            
            nn.Unflatten(1, (256, patch_size, patch_size)),  # (batch, 256, 8, 8)
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),  # Batch Normalization
            nn.ReLU(),
            
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),  # Batch Normalization
            nn.ReLU(),
            
            nn.ConvTranspose2d(64, output_channels, kernel_size=3, stride=1, padding=1)  # (batch, 3, 8, 8)
        )

    def forward(self, x):
        return self.decoder(x)

class CNNEncoderDecoder(nn.Module):
    def __init__(self, input_channels=3, patch_size=8, encoded_size=24):
        super(CNNEncoderDecoder, self).__init__()
        self.encoder = CNNEncoder(input_channels, patch_size, encoded_size)
        self.decoder = CNNDecoder(encoded_size, input_channels, patch_size)

    def forward(self, patches):
        encoded = self.encoder(patches)
        decoded = self.decoder(encoded)
        return decoded

# Utility Functions to Convert Images to Patches and Vice Versa
def images_to_patches(images, patch_size):
    """
    Convert a batch of images (b, c, w, h) into patches (b * num_patches, c, patch_size, patch_size).
    """
    b, c, w, h = images.shape
    assert w % patch_size == 0 and h % patch_size == 0, "Width and height must be divisible by patch size."
    
    #pdb.set_trace()
    # Reshape the images into patches
    patches = images.unfold(2, patch_size, patch_size).unfold(3, patch_size, patch_size)
    patches = patches.permute(0, 2, 3, 1, 4, 5).reshape(-1, c, patch_size, patch_size)

    return patches

def patches_to_images(patches, image_size, patch_size):
    """
    Convert patches (b * num_patches, c, patch_size, patch_size) back into images (b, c, w, h).
    """
    b, c, w, h = image_size
    num_patches_w = w // patch_size  # Number of patches along width
    num_patches_h = h // patch_size  # Number of patches along height
    
    #pdb.set_trace()
    # Reshape patches back to the original image size
    patches = patches.view(b, num_patches_w, num_patches_h, c, patch_size, patch_size)
    patches = patches.permute(0, 3, 1, 4, 2, 5).contiguous()
    images = patches.view(b, c, w, h)

    return images

def expanding_mat(attention,patch_size):
    #pdb.set_trace()
    (row,col) = attention.shape
    exp_attn = np.zeros([row*patch_size,col*patch_size])
    for i in range(row):
        for j in range(col):
            if attention[i][j] == 1:
                exp_attn[i*patch_size:(i+1)*patch_size,j*patch_size:(j+1)*patch_size] = np.ones([patch_size,patch_size])
            elif attention[i][j] == 2:
                exp_attn[i*patch_size:(i+1)*patch_size,j*patch_size:(j+1)*patch_size] = 2*np.ones([patch_size,patch_size])
            elif attention[i][j] == 3:
                exp_attn[i*patch_size:(i+1)*patch_size,j*patch_size:(j+1)*patch_size] = 3*np.ones([patch_size,patch_size])
            elif attention[i][j] == 4:
                exp_attn[i*patch_size:(i+1)*patch_size,j*patch_size:(j+1)*patch_size] = 4*np.ones([patch_size,patch_size])
    return exp_attn

def LQ(y):
    x = y.copy()
    rate = {0:0, 1:12, 2:24, 3:48, 4:196}
    for i in range(len(x)):
        if x[i]<rate[1]:
            x[i] = rate[0]
        elif x[i]<rate[2]:
            x[i] = rate[1]
        elif x[i]<rate[3]:
            x[i] = rate[2]
        elif x[i]<rate[4]:
            x[i] = rate[3]
        else:
            x[i] = rate[4]
    return x

def UQ(y):
    x = y.copy()
    rate = {0:0, 1:12, 2:24, 3:48, 4:196}
    for i in range(len(x)):
        if x[i]>rate[3]:
            x[i] = rate[4]
        elif x[i]>rate[2]:
            x[i] = rate[3]
        elif x[i]>rate[1]:
            x[i] = rate[2]
        elif x[i]>rate[0]:
            x[i] = rate[1]
        else:
            x[i] = rate[0]
    return x

def A_to_R(attn,r):
    #pdb.set_trace()
    rate = {0:0, 1:12, 2:24, 3:48, 4:196}
    attn_flatten = attn.reshape([2400])
    sum_attn = attn_flatten.sum()
    attn_f = attn_flatten*(r/sum_attn)
    mask = np.zeros(attn_f.shape)
    attn_f_perm = np.argsort(-attn_f)
    if r <= rate[1]*2400:
        for ind in range(1,int(r/rate[1])):
            mask[attn_f_perm[ind]] = 1
    else:
        # Replace with LQ but once the LQ is 0, we replace it with 1
        LQ_attn = LQ(attn_f)
        UQ_attn = UQ(attn_f)
        attn_f[LQ_attn==0] = -1000
        attn_f[LQ_attn==196] = -10000
        LQ_attn[LQ_attn==0] = rate[1] # We will overshoot r because of replacing 0 with 1.
        sum_LQ_rate = sum(LQ_attn)
        while sum_LQ_rate<r: # We might overshoot in the last loop
            diff = UQ_attn - attn_f
            diff_perm = np.argsort(diff)
            LQ_attn[diff_perm[0]] = UQ_attn[diff_perm[0]]
            attn_f[diff_perm[0]] = LQ_attn[diff_perm[0]] + 0.01
            if UQ_attn[diff_perm[0]] == 196:
                #pdb.set_trace()
                attn_f[diff_perm[0]] = -10000
            sum_LQ_rate = sum(LQ_attn)
            #print(sum_LQ_rate)
            UQ_attn = UQ(attn_f)
            
    #pdb.set_trace()
    if r > rate[1]*2400:
        LQ_attn[LQ_attn==rate[1]] = 1
        LQ_attn[LQ_attn==rate[2]] = 2
        LQ_attn[LQ_attn==rate[3]] = 3
        LQ_attn[LQ_attn==rate[4]] = 4
    else:
        LQ_attn = mask
    mask_attn = LQ_attn.reshape([40,60])
    #pdb.set_trace()
    
    
    return mask_attn
    
if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    args = get_parser().parse_args()
    setup_logger(name="fvcore")
    logger = setup_logger()
    logger.info("Arguments: " + str(args))

    cfg = setup_cfg(args)
    #pdb.set_trace()
    demo = VisualizationDemo(cfg)
    
    num_channels = 3
    patch_size = 8
    device = 'cuda:0' # default GPU device
    Unet3 = CNNEncoderDecoder(input_channels=num_channels, patch_size=patch_size, encoded_size=3)
    Unet6 = CNNEncoderDecoder(input_channels=num_channels, patch_size=patch_size, encoded_size=6)
    Unet12 = CNNEncoderDecoder(input_channels=num_channels, patch_size=patch_size, encoded_size=12)
    Unet3.to(device)
    Unet6.to(device)
    Unet12.to(device)
    Unet3.load_state_dict(torch.load("demo/models/unet_3.pt"))
    Unet6.load_state_dict(torch.load("demo/models/unet_6.pt"))
    Unet12.load_state_dict(torch.load("demo/models/unet_12.pt"))

    with open('demo/custom_coco_all.json',"r") as f:
        coco_data = json.load(f)
        
    image_ids = list(coco_data.keys())
    count = 0

    # start_time = time.time()
    while count < 3:
        image_id = image_ids[count] # select from custom_coco_all.json
        img_entry = coco_data[str(image_id)]
        file_name = img_entry["file_name"]
        command = img_entry["command"]
        #pdb.set_trace()
        img_dir = 'demo/val2017/' + file_name
        img = read_image(img_dir, format="BGR")
        
        #pdb.set_trace()
        img_resized = cv2.resize(img, (480,320))
        user_classes = [command]
        predictions, vis_output = demo.run_on_image(img_resized, user_classes)
        vis_output.save(f"check_{image_id}.jpg")
        
        predictions = predictions['sem_seg'].unsqueeze(0)
        pooled = F.avg_pool2d(predictions, kernel_size = 8, stride= 8)
        pooled_attn = pooled.squeeze(0).squeeze(0).cpu().numpy()
        
        r = 2400*100
        #pdb.set_trace()
        masked_attn = A_to_R(pooled_attn,r)
        expanded_attn = expanding_mat(masked_attn,patch_size)
        #pdb.set_trace()
        img_resized_2 = img_resized[:, :, ::-1].copy()
        y = transform_1(img_resized_2).unsqueeze(0).to(device)
        #pdb.set_trace()
        #y = y.permute(0, 1, 3, 2)
        patches = images_to_patches(y, patch_size)
        patches = patches.to(device)
        output_patches3 = Unet3(patches)
        output_patches6 = Unet6(patches)
        output_patches12 = Unet12(patches)
        output_images3 = patches_to_images(output_patches3, y.shape, patch_size)
        output_images6 = patches_to_images(output_patches6, y.shape, patch_size)
        output_images12 = patches_to_images(output_patches12, y.shape, patch_size)
        reconstruct = torch.zeros(y.shape).to(device)
        
        reconstruct[0,0,expanded_attn==1] = output_images3[0,0,expanded_attn==1]
        reconstruct[0,1,expanded_attn==1] = output_images3[0,1,expanded_attn==1]
        reconstruct[0,2,expanded_attn==1] = output_images3[0,2,expanded_attn==1]
        reconstruct[0,0,expanded_attn==2] = output_images6[0,0,expanded_attn==2]
        reconstruct[0,1,expanded_attn==2] = output_images6[0,1,expanded_attn==2]
        reconstruct[0,2,expanded_attn==2] = output_images6[0,2,expanded_attn==2]
        reconstruct[0,0,expanded_attn==3] = output_images12[0,0,expanded_attn==3]
        reconstruct[0,1,expanded_attn==3] = output_images12[0,1,expanded_attn==3]
        reconstruct[0,2,expanded_attn==3] = output_images12[0,2,expanded_attn==3]
        reconstruct[0,0,expanded_attn==4] = y[0,0,expanded_attn==4]
        reconstruct[0,1,expanded_attn==4] = y[0,1,expanded_attn==4]
        reconstruct[0,2,expanded_attn==4] = y[0,2,expanded_attn==4]
        
        # pdb.set_trace()
        # image_pil = F2.to_pil_image(reconstruct.squeeze(0))
        # image_pil.save("reconstructed.jpg")
        reconstruct = reconstruct.squeeze(0)
        pil_image = F2.to_pil_image(reconstruct)
        pil_image.save(f"reconstructed_{image_id}.jpg")
        count += 1
    # print("--- %s seconds ---" % (time.time() - start_time))

