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
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
from pycocotools import mask as maskUtils
from PIL import Image, Resampling
from transformers import CLIPProcessor, CLIPModel
import requests
from functools import partial
import math
import skimage.measure


# constants
WINDOW_NAME = "maft demo"

def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
    # type: (Tensor, float, float, float, float) -> Tensor
    return _no_grad_trunc_normal_(tensor, mean, std, a, b)


def _no_grad_trunc_normal_(tensor, mean, std, a, b):
    # Cut & paste from PyTorch official master until it's in a few official releases - RW
    # Method based on https://people.sc.fsu.edu/~jburkardt/presentations/truncated_normal.pdf
    def norm_cdf(x):
        # Computes standard normal cumulative distribution function
        return (1. + math.erf(x / math.sqrt(2.))) / 2.


def drop_path(x, drop_prob: float = 0., training: bool = False):
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    # work with diff dim tensors, not just 2D ConvNets
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = keep_prob + \
        torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    #Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).


    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C //
                                  self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x, attn


class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(
            drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim,
                       act_layer=act_layer, drop=drop)

    def forward(self, x, return_attention=False):
        y, attn = self.attn(self.norm1(x))
        if return_attention:
            return attn
        x = x + self.drop_path(y)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class PatchEmbed(nn.Module):
    #Image to Patch Embedding


    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        num_patches = (img_size // patch_size) * (img_size // patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.proj = nn.Conv2d(in_chans, embed_dim,
                              kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        B, C, H, W = x.shape
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x

class VisionTransformer(nn.Module):
    #Vision Transformer

    def __init__(self, img_size=[224], patch_size=16, in_chans=3, num_classes=0, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., norm_layer=nn.LayerNorm, **kwargs):
        super().__init__()
        self.num_features = self.embed_dim = embed_dim

        self.patch_embed = PatchEmbed(
            img_size=img_size[0], patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(
            torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        # stochastic depth decay rule
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)

        # Classifier head
        self.head = nn.Linear(
            embed_dim, num_classes) if num_classes > 0 else nn.Identity()

        trunc_normal_(self.pos_embed, std=.02)
        trunc_normal_(self.cls_token, std=.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def interpolate_pos_encoding(self, x, w, h):
        npatch = x.shape[1] - 1
        N = self.pos_embed.shape[1] - 1
        if npatch == N and w == h:
            return self.pos_embed
        class_pos_embed = self.pos_embed[:, 0]
        patch_pos_embed = self.pos_embed[:, 1:]
        dim = x.shape[-1]
        w0 = w // self.patch_embed.patch_size
        h0 = h // self.patch_embed.patch_size
        # we add a small number to avoid floating point error in the interpolation
        # see discussion at https://github.com/facebookresearch/dino/issues/8
        w0, h0 = w0 + 0.1, h0 + 0.1
        patch_pos_embed = nn.functional.interpolate(
            patch_pos_embed.reshape(1, int(math.sqrt(N)), int(
                math.sqrt(N)), dim).permute(0, 3, 1, 2),
            scale_factor=(w0 / math.sqrt(N), h0 / math.sqrt(N)),
            mode='bicubic',
        )
        assert int(
            w0) == patch_pos_embed.shape[-2] and int(h0) == patch_pos_embed.shape[-1]
        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
        return torch.cat((class_pos_embed.unsqueeze(0), patch_pos_embed), dim=1)

    def prepare_tokens(self, x):
        B, nc, w, h = x.shape
        x = self.patch_embed(x)  # patch linear embedding


        # add the [CLS] token to the embed patch tokens
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)


        # add positional encoding to each token
        x = x + self.interpolate_pos_encoding(x, w, h)


        return self.pos_drop(x)

    def forward(self, x):
        #pdb.set_trace()
        x = self.prepare_tokens(x)
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        return x[:, 0]

    def output_encoder(self, x):
        x = self.prepare_tokens(x)
        for i, blk in enumerate(self.blocks):
            if i < len(self.blocks) - 1:
                x = blk(x)
            else:
                # return attention of the last block
                return x, blk(x, return_attention=True)

    def get_intermediate_layers(self, x, n=1):
        x = self.prepare_tokens(x)
        # we return the output tokens from the `n` last blocks
        output = []
        for i, blk in enumerate(self.blocks):
            x = blk(x)
            if len(self.blocks) - i <= n:
                output.append(self.norm(x))
        return output

#class VitGenerator(object):
class VitGenerator(nn.Module):
    def __init__(self, name_model, patch_size, device, evaluate=True, random=False, verbose=False):
        super(VitGenerator, self).__init__()
        self.name_model = name_model
        self.patch_size = patch_size
        self.evaluate = evaluate
        self.device = device
        self.verbose = verbose
        self.model = self._getModel()
        self._initializeModel()
        if not random:
            self._loadPretrainedWeights()

    def _getModel(self):
        if self.verbose:
            print(
                f"[INFO] Initializing {self.name_model} with patch size of {self.patch_size}")
        if self.name_model == 'vit_tiny':
            model = VisionTransformer(patch_size=self.patch_size, embed_dim=192, depth=12, num_heads=3, mlp_ratio=4,
                                      qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6))

        elif self.name_model == 'vit_small':
            model = VisionTransformer(patch_size=self.patch_size, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4,
                                      qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6))

        elif self.name_model == 'vit_base':
            model = VisionTransformer(patch_size=self.patch_size, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4,
                                      qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6))
        else:
            raise f"No model found with {self.name_model}"

        return model

    def _initializeModel(self):
        if self.evaluate:
            for p in self.model.parameters():
                p.requires_grad = False

            self.model.eval()

        self.model.to(self.device)

    def _loadPretrainedWeights(self):
        if self.verbose:
            print("[INFO] Loading weights")
        url = None
        if self.name_model == 'vit_small' and self.patch_size == 16:
            url = "dino_deitsmall16_pretrain/dino_deitsmall16_pretrain.pth"

        elif self.name_model == 'vit_small' and self.patch_size == 8:
            url = "dino_deitsmall8_300ep_pretrain/dino_deitsmall8_300ep_pretrain.pth"

        elif self.name_model == 'vit_base' and self.patch_size == 16:
            url = "dino_vitbase16_pretrain/dino_vitbase16_pretrain.pth"

        elif self.name_model == 'vit_base' and self.patch_size == 8:
            url = "dino_vitbase8_pretrain/dino_vitbase8_pretrain.pth"

        if url is None:
            print(
                f"Since no pretrained weights have been found with name {self.name_model} and patch size {self.patch_size}, random weights will be used")

        else:
            state_dict = torch.hub.load_state_dict_from_url(
                url="https://dl.fbaipublicfiles.com/dino/" + url)
            self.model.load_state_dict(state_dict, strict=True)

    def output_encoder(self, img):
        return self.model.output_encoder(img.to(self.device))

    def __call__(self, x):
        return self.model(x)
    
    

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

def visualize_predict(model, img, img_size, patch_size, device):
    #pdb.set_trace()
    img_pre = transform_1(img)
    img_pre = img_pre[:, :, :].unsqueeze(0)
    #pdb.set_trace()
    encoded_img, decoded_attention = visualize_attention(model, img_pre, patch_size, device)
    #plot_attention(img, decoded_attention)
    return encoded_img, decoded_attention

def visualize_attention(model, img, patch_size, device):
    #pdb.set_trace()
    # make the image divisible by the patch size
    #w, h = img.shape[2] - img.shape[2] % patch_size, img.shape[3] - \
        #img.shape[3] % patch_size
    w_featmap = img.shape[-2] // patch_size
    h_featmap = img.shape[-1] // patch_size

    with torch.no_grad():
        encoded_img, attentions = model.output_encoder(img.to(device))

    nh = attentions.shape[1]  # number of head

    # keep only the output patch attention
    attentions = attentions[0, :, 0, 1:].reshape(nh, -1)

    attentions = attentions.reshape(nh, w_featmap, h_featmap)
    attentions = nn.functional.interpolate(attentions.unsqueeze(
        0), scale_factor=patch_size, mode="nearest")[0].cpu().numpy()
    
    return encoded_img, attentions

    
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
    name_model = 'vit_small'
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
    model = CLIPModel.from_pretrained("./clip-vit-large-patch14-336")
    processor = CLIPProcessor.from_pretrained("./clip-vit-large-patch14-336")
    model1 = VitGenerator(name_model, patch_size,device, evaluate=True, random=False, verbose=True)
    model1.to(device)
    
    
    with open('demo/custom_coco_all.json',"r") as f:
        coco_data = json.load(f)
    image_ids = list(coco_data.keys())
    count = 0

    r_list = [2400*7, 2400*14, 2400*19.2, 2400*38.4, 2400*57.6, 2400*76.8, 2400*96, 2400*115.2, 2400*134.4, 2400*153.6, 2400*172.8, 2400*192]
    mse_all = []
    attn_all = []
    clip_all = []
    mse_vit_all = []
    attn_vit_all = []
    clip_vit_all = []

    start_time = time.time()
    while count < 20:
        image_id = image_ids[count] # select from custom_coco_all.json
        img_entry = coco_data[str(image_id)]
        file_name = img_entry["file_name"]
        command = img_entry["command"]
        segmentations = img_entry["segmentations"]
        width, height = img_entry["imagesize"]
        #pdb.set_trace()
        img_dir = 'demo/val2017/' + file_name
        img = read_image(img_dir, format="BGR")
        print(f"---- Started to check {file_name} ----.")
       
        #pdb.set_trace()
        img_resized = cv2.resize(img, (480,320))
        user_classes = [command]
        inputs_text = processor(text=user_classes, return_tensors="pt", padding=True)
        text_features = model.get_text_features(**inputs_text)
        predictions, vis_output = demo.run_on_image(img_resized, user_classes)
        #vis_output.save("check.jpg")
        
        predictions = predictions['sem_seg'].unsqueeze(0)
        pooled = F.avg_pool2d(predictions, kernel_size = 8, stride= 8)
        pooled_attn = pooled.squeeze(0).squeeze(0).cpu().numpy()

        # Row values for each r
        mse_row = []
        attn_row = []
        clip_row = []
        mse_vit_row = []
        attn_vit_row = []
        clip_vit_row = []
        
        # r = 2400*60
        for r in r_list:
            print(f"Reconstructing for rate {r}...")
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
            
            reconstruct = reconstruct.squeeze(0)
            pil_image = F2.to_pil_image(reconstruct)
            pil_image.save("reconstructed.jpg")
                
            # 2. Initialize empty mask
            binary_mask = np.zeros((height, width), dtype=np.uint8)
            
            # 3. For each segmentation polygon, update mask
            for seg in segmentations:
                if not seg:
                    continue
                rle = maskUtils.frPyObjects(seg, height, width)
                mask = maskUtils.decode(rle)

                # rle decode can return HxWxN array if multiple polygons
                if mask.ndim == 3:
                    mask = np.any(mask, axis=2).astype(np.uint8)
                binary_mask = np.maximum(binary_mask, mask)
            
            mask = cv2.resize(binary_mask, (480, 320), interpolation=cv2.INTER_LINEAR)
            # 4. Invert (1=black, 0=white) and convert to image
            inverted_mask = (1 - binary_mask) * 255  # to 0 (black) / 255 (white)
            mask_img = Image.fromarray(inverted_mask.astype(np.uint8), mode='L')

            # 5. Resize to (480, 320) → note: PIL uses (width, height)
            # resized_img = mask_img.resize((480, 320), resample=Resampling.NEAREST)
            # resized_img.save("mask.png")
            
            ##EVALUATION of reconstructed images
            #1. Evaluating MSE
            #pdb.set_trace()
            # Convert mask to a torch tensor and reshape to [1, 1, 320, 480]
            mask_torch = torch.from_numpy(mask).float().unsqueeze(0).unsqueeze(0).to(device)   # [1, 1, 320, 480]

            # Ensure mask is broadcastable to [1, 3, 320, 480]
            mask_torch = mask_torch.expand(-1, 3, -1, -1)  # [1, 3, 320, 480]
            diff = (y - reconstruct) ** 2
            masked_diff = diff * mask_torch  # Apply mask

            # Compute mean over masked region
            mse = masked_diff.sum() / mask_torch.sum()
            mse_row.append(mse.item())
            # print("Masked MSE:", mse.item())
            
            #2. Evaluating MAFT attention difference
            #pdb.set_trace()
            rec_img = read_image("reconstructed.jpg", format="BGR")
            predictions_rec, _ = demo.run_on_image(rec_img, user_classes)
            predictions_rec = predictions_rec['sem_seg'].unsqueeze(0)
            Diff_attn = torch.mean((predictions - predictions_rec) ** 2)
            attn_row.append(Diff_attn.item())
            # print("Diff_attn:", Diff_attn.item())
            
            #3. Evaluating CLIP score
            #pdb.set_trace()
            inputs_img_rec = processor(images=pil_image, return_tensors="pt", padding=True)
            image_features_rec = model.get_image_features(**inputs_img_rec)
            
            # inputs_img_org = processor(images=img_resized, return_tensors="pt", padding=True)
            # image_features_org = model.get_image_features(**inputs_img_org)

            similarity_score = torch.nn.functional.cosine_similarity(image_features_rec, text_features).item()
            # similarity_score_UP = torch.nn.functional.cosine_similarity(image_features_org, text_features).item()
            clip_row.append(similarity_score)
            # print(f"similarity_score: {similarity_score}") 
            
            
            ### My previous work - efficient semantic communication
            #pdb.set_trace()
            img_size = (480,320)
            image = Image.open(img_dir).convert('RGB')
            image_resized = cv2.resize(img, (480,320))
            img_encd, attn=visualize_predict(model1, image_resized, img_size, patch_size, device)
            attn_mean = np.mean(attn,0)
            pooled_attn = skimage.measure.block_reduce(attn_mean, (patch_size,patch_size), np.mean)
            masked_attn = pooled_attn[:]
            masked_attn = A_to_R(pooled_attn,r)
            expanded_attn = expanding_mat(masked_attn,patch_size)
            image_resized = img_resized[:, :, ::-1].copy()
            y = transform_1(image_resized).unsqueeze(0).to(device)
            patches = images_to_patches(y, patch_size)
            patches = patches.to(device)
            output_patches3 = Unet3(patches)
            output_patches6 = Unet6(patches)
            output_patches12 = Unet12(patches)
            output_images3 = patches_to_images(output_patches3, y.shape, patch_size)
            output_images6 = patches_to_images(output_patches6, y.shape, patch_size)
            output_images12 = patches_to_images(output_patches12, y.shape, patch_size)
            reconstruct = torch.zeros(y.shape).to(device)
            #pdb.set_trace()
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
            reconstruct = reconstruct.squeeze(0)
            pil_image_vit = F2.to_pil_image(reconstruct)
            pil_image_vit.save("reconstructed_vit.jpg")
            
            
            ##EVALUATION of reconstructed images obtained via VIT transformer
            #1. Evaluating MSE
            # Convert mask to a torch tensor and reshape to [1, 1, 320, 480]
            mask_torch = torch.from_numpy(mask).float().unsqueeze(0).unsqueeze(0).to(device)   # [1, 1, 320, 480]

            # Ensure mask is broadcastable to [1, 3, 320, 480]
            mask_torch = mask_torch.expand(-1, 3, -1, -1)  # [1, 3, 320, 480]
            diff = (y - reconstruct) ** 2
            masked_diff = diff * mask_torch  # Apply mask

            # Compute mean over masked region
            mse_vit = masked_diff.sum() / mask_torch.sum()
            mse_vit_row.append(mse_vit.item())
            # print("Masked MSE via VIT:", mse_vit.item())
            
            #2. Evaluating MAFT attention difference
            #pdb.set_trace()
            rec_img_vit = read_image("reconstructed_vit.jpg", format="BGR")
            predictions_rec_vit, _ = demo.run_on_image(rec_img_vit, user_classes)
            predictions_rec_vit = predictions_rec_vit['sem_seg'].unsqueeze(0)
            Diff_attn_vit = torch.mean((predictions - predictions_rec_vit) ** 2)
            attn_vit_row.append(Diff_attn_vit.item())
            # print("Diff_attn via VIT:", Diff_attn_vit.item())
            
            #3. Evaluating CLIP score
            #pdb.set_trace()
            inputs_img_rec_vit = processor(images=pil_image_vit, return_tensors="pt", padding=True)
            image_features_rec_vit = model.get_image_features(**inputs_img_rec_vit)
            
            # inputs_img_org = processor(images=img_resized, return_tensors="pt", padding=True)
            # image_features_org = model.get_image_features(**inputs_img_org)

            similarity_score_vit = torch.nn.functional.cosine_similarity(image_features_rec_vit, text_features).item()
            # similarity_score_UP = torch.nn.functional.cosine_similarity(image_features_org, text_features).item()
            clip_vit_row.append(similarity_score_vit)
            # print(f"similarity_score_vit: {similarity_score_vit}")
            
        mse_all.append(mse_row)
        attn_all.append(attn_row)
        clip_all.append(clip_row)
        mse_vit_all.append(mse_vit_row)
        attn_vit_all.append(attn_vit_row)
        clip_vit_all.append(clip_vit_row)
        
        # Save .pt for each image to avoid losing progress
        torch.save(mse_all, "MSE_100.pt")
        torch.save(attn_all, "Diff_attn_100.pt")
        torch.save(clip_all, "CLIP_score_100.pt")
        torch.save(mse_vit_all, "MSE_ViT_100.pt")
        torch.save(attn_vit_all, "Diff_attn_ViT_100.pt")
        torch.save(clip_vit_all, "CLIP_score_ViT_100.pt")
        print(f"Saved a result for {file_name} (image {count+1}).")  
        count += 1
        
    print("Saved all results.")
    print("--- %s seconds ---" % (time.time() - start_time))