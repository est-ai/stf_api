import numpy as np
import cv2
import torchvision.transforms as T
from PIL import Image
import random
import torch


def manual_seed(rng):
    if rng:
        torch.manual_seed(rng)
        random.seed(rng)


#------------------------------------------------------------------------------
mask_img_trsf_ver_00_jitter = T.ColorJitter(brightness=0.3, contrast=0.2, saturation=0.4, hue=0.0)

def mask_img_trsf_ver_00(im, rng = None):
    
    manual_seed(rng)
    
    if random.randint(0, 1) == 0:
        im = cv2.GaussianBlur(im, (random.randint(0,1)*2+1,random.randint(0,1)*2+1), 0, 0)
        
    if random.randint(0, 1) == 0:
        im = Image.fromarray(im)
        im = mask_img_trsf_ver_00_jitter(im)
        im = np.array(im)
    return im

#------------------------------------------------------------------------------
mask_img_trsf_ver_01_jitter = T.ColorJitter(brightness=0.15, contrast=0.1, saturation=0.2, hue=0.0)

def mask_img_trsf_ver_01(im, rng = None):
    
    manual_seed(rng)
    
    #if random.randint(0, 1) == 0:
    #    im = cv2.GaussianBlur(im, (random.randint(0,1)*2+1,random.randint(0,1)*2+1), 0, 0)
        
    if random.randint(0, 1) == 0:
        im = Image.fromarray(im)
        im = mask_img_trsf_ver_01_jitter(im)
        im = np.array(im)
    return im

mask_img_trsfs= {
    0: mask_img_trsf_ver_00,
    1: mask_img_trsf_ver_01
}