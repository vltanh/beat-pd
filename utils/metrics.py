import numpy as np
import torch
from torch.nn import functional as F

class Metrics():
    def __init__(self):
        self.reset()
    
    def calculate(self, output, target, is_track):
        # Calculate on single batch
        per_batch_result = self._calculate(output, target)
        
        # If is tracking, update
        if is_track: self.update()
        
        # Return result on single batch
        return per_batch_result
    
    def _calculate(self, output, target):
        pass
    
    def update(self):
        pass
    
    def value(self):
        pass
    
    def reset(self):
        pass
    
    def summary(self):
        pass

class PixelAccuracy(Metrics):
    def __init__(self):
        super().__init__()
    
    def _calculate(self, output, target):
        _, prediction = torch.max(output, dim=1)
        self.per_batch_val = torch.sum(torch.eq(prediction, target)) 
        self.per_batch_n = np.prod(prediction.shape)
        return float(self.per_batch_val / self.per_batch_n)
    
    def update(self):
        self.val += self.per_batch_val
        self.n += self.per_batch_n
    
    def value(self):
        return float(self.val / self.n)
    
    def reset(self):        
        self.per_batch_val = 0
        self.per_batch_n = 0
        
        self.val = 0
        self.n = 0
        
    def summary(self):
        return { 
                'Correct': int(self.val),
                'Total': int(self.n)
               }
        
class IoU(Metrics):
    def __init__(self, num_classes=151):
        self.num_classes = num_classes
        super().__init__()
    
    def _calculate(self, output, target):
        _, prediction = torch.max(output, dim=1)
        unique_classes =  np.unique(prediction)
        self.per_batch_class_iou = dict()
        for c in range(self.num_classes):
            pred_c = prediction == c
            seg_c = target == c
            intersection = torch.sum(seg_c & pred_c)
            union = torch.sum(seg_c | pred_c)
            self.per_batch_class_iou.setdefault(c, {'i': 0, 'u': 0})
            self.per_batch_class_iou[c]['i'] += torch.sum(intersection)
            self.per_batch_class_iou[c]['u'] += torch.sum(union)
        miou = np.mean([x['i'] * 1.0 / x['u'] for x in self.per_batch_class_iou.values()])
        return miou
    
    def update(self):
        for c, x in self.per_batch_class_iou.items():
            self.class_iou.setdefault(c, {'i': 0, 'u': 0})
            self.class_iou[c]['i'] += x['i']
            self.class_iou[c]['u'] += x['u']
    
    def value(self):
        return float(np.mean([x['i'] * 1.0 / x['u'] for x in self.class_iou.values()]))
    
    def reset(self):        
        self.per_batch_class_iou = { c: {'i': 0, 'u': 0} for c in range(self.num_classes) }
        self.class_iou = { c: {'i': 0, 'u': 0} for c in range(self.num_classes) }
        
    def summary(self):
        return { c: float(x['i'] * 1.0 / x['u']) for c, x in self.class_iou.items() }