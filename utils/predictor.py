import time
import os
from PIL import Image

from utils.visualizer import ADE20KVisualizer

import torch
from torchvision import transforms

class Predictor():
    def __init__(self, net, device, visualizer):
        self.net = net
        self.device = device
        self.visualizer = visualizer
        self.net.eval()
        
    def load_image(self, input_path):
        img = Image.open(input_path)
        img = transforms.ToTensor()(img)
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        img = normalize(img)
        img = img[None,:,:,:]
        return img

    @torch.no_grad()
    def predict(self, input_image):
        # Start predicting
        start = time.time()

        # Load input
        inp = input_image.to(self.device)

        # Get network outputs
        out = self.net(inp)[0].detach().cpu()

        # Post processing outputs
        pred = Image.fromarray(self.visualizer.process_output(out))
        
        print('Prediction time: %f' % (time.time()-start))
        
        return pred