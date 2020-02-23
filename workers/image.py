import numpy as np
import scipy.io as io

import torch
from torch.nn import functional as F

def get_new_size(img_size, max_dim=800):
    w, h = img_size
    ratio = 1.0
    if w > max_dim:
        ratio = max_dim / w
    if h > max_dim:
        ratio = min(max_dim / h, ratio)
    nw = int(w * ratio + 0.5)
    nh = int(h * ratio + 0.5)
    return nw, nh

def find_nearest_power_of_2(dim):
    return 1 << (dim.bit_length() - 1)

class Denormalize():
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
        return tensor
    
class Normalize():
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        for t, m, s in zip(tensor, self.mean, self.std):
            t.sub_(m).div_(s)
        return tensor
    
class RandomCrop():
    def __init__(self, size):
        self.size = size
        
    def find_nearest_power_of_2(self, dim):
        return 1 << (dim.bit_length() - 1)
    
    def __call__(self, img, seg):
        _, h, w = img.size()
        size = find_nearest_power_of_2(min(w, h))
        c = np.random.randint(w - size + 1)
        r = np.random.randint(h - size + 1)
        img = F.interpolate(img[None, :, r:r+size, c:c+size], 
                            size=(self.size, self.size),
                            mode="bilinear", 
                            align_corners=True)[0]
        seg = F.interpolate(seg[None, None, r:r+size, c:c+size].float(), 
                            size=(self.size, self.size),
                            mode="nearest")[0][0].long()
        return img, seg