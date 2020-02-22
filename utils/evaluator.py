from tqdm import tqdm
from torchnet import meter
import time
import os
import matplotlib.pyplot as plt

from utils.metrics import PixelAccuracy
from utils.visualizer import ADE20KVisualizer

import torch

class Evaluator():
    def __init__(self, output_path, 
                 net, criterion, metrics, 
                 visualizer, device):
        self.output_path = output_path
        self.net = net
        self.criterion = criterion
        self.metrics = metrics
        self.device = device
        self.visualizer = visualizer

    @torch.no_grad()
    def eval(self, dataloader):
        # Recore inference time
        _time = meter.AverageValueMeter()

        # Record loss and metrics
        _loss = meter.AverageValueMeter()
        _metrics = self.metrics
        for metric in _metrics.values(): 
            metric.reset()

        # Start validating
        self.net.eval()
        progress_bar = tqdm(dataloader)
        progress_bar.set_description_str('[ VAL ]')
        for i, (inps, lbls) in enumerate(progress_bar):
            # Begin stopwatch
            start = time.time()

            # Load inputs and labels
            inps = inps.to(self.device)
            lbls = lbls.to(self.device)

            # Get network outputs
            outs = self.net(inps)

            # Stop stopwatch
            _time.add(time.time() - start)

            # Calculate the loss
            loss = self.criterion(outs, lbls)

            # Update loss
            _loss.add(loss.item())
            
            # Detach and transfer to CPU
            inps = inps.detach().cpu()
            lbls = lbls.detach().cpu()
            outs = outs.detach().cpu()

            # Update metrics
            for metric in _metrics.values():
                metric.calculate(outs, lbls, is_track=True)
            
            # Visualize
            for idx, item in enumerate(zip(inps, lbls, outs)):
                fig = self.visualizer.visualize(item, output_path=os.path.join(self.output_path, 
                                                                               'Image_%04d.png' % (i*len(inps) + idx)))

        # Calculate evaluation result
        avg_pred_time = _time.value()[0]
        avg_loss = _loss.value()[0]
        avg_metrics = { mid: metric.value() for mid, metric in _metrics.items() }

        # Print results
        print('Average prediction time: {} (s)'.format(avg_pred_time))
        print('loss:', avg_loss)
        for mid, metric in _metrics.items():
            print(metric.summary())
            print('{}: {}'.format(mid, metric.value()))