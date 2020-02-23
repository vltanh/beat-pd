import numpy as np
import scipy.io as spio
import io
from PIL import Image
import matplotlib.pyplot as plt

from workers.image import Denormalize

import torch
from torch.nn import functional as F

def get_img_from_fig(fig, dpi=180):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi)
    buf.seek(0)
    img = Image.open(buf)
    return img

class Visualizer:
    def __init__(self, **kwargs):
        pass
    
    def visualize(self, item):
        pass
    
class Im2ImVisualizer(Visualizer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
    def process_input(self, inp):
        pass
    
    def process_label(self, lbl):
        pass
    
    def process_output(self, pred):
        pass
    
    def visualize(self, item, title=None, output_path=None):
        fig = plt.figure(figsize=(5, 10))

        inp = self.process_input(item[0])
        lbl = self.process_label(item[1])
        out = self.process_output(item[2])
        
        for j, k in enumerate((inp, lbl, out)):
            plt.subplot(3, 1, j + 1)
            plt.imshow(k)
            plt.xticks([])
            plt.yticks([])
        
        if title is not None:
            fig.suptitle(title)
            
        img = get_img_from_fig(fig)
        if output_path is not None:
            img.save(output_path)
        plt.close(fig)
        
        return img
    
class SegmentationVisualizer(Im2ImVisualizer):
    def __init__(self, cmap, **kwargs):
        super().__init__(**kwargs)
        self.cmap = cmap
        
    def visualize_segmentation(self, seg):
        seg_ = np.zeros((*seg.shape, 3), dtype=np.uint8)
        for l in range(len(self.cmap)):
            seg_[seg == l] = self.cmap[l]
        return seg_
    
class ADE20KVisualizer(SegmentationVisualizer):
    def __init__(self, cmap_path, **kwargs):
        super().__init__(spio.loadmat(cmap_path)['colors'], **kwargs)
    
    def process_input(self, inp):
        denormalize = Denormalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        inp = denormalize(inp)
        return inp.numpy().transpose(1, 2, 0)
    
    def process_label(self, lbl):
        return self.visualize_segmentation(lbl.numpy().astype(int))
    
    def process_output(self, out):
        _, pred = torch.max(out, dim=0)
        return self.visualize_segmentation(pred.numpy().astype(int))