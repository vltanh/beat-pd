from tqdm import tqdm
from torchnet import meter
import time
import os
import matplotlib.pyplot as plt

from utils.metrics import PixelAccuracy, IoU
from utils.logger import Logger, TensorboardHelper
from utils.visualizer import ADE20KVisualizer

import torch

class Trainer():
    def __init__(self, train_id, config, device,
                 net, optimizer, criterion, metrics, scheduler):
        self.train_id = train_id
        
        # TODO: passing full config defeats the purpose, any better way?
        self.full_config = config        
        self.config = config['trainer']
        
        self.net = net
        self.optimizer = optimizer
        self.criterion = criterion
        self.scheduler = scheduler
        self.metrics = metrics
        
        self.device = device
        
        self.output_path = self.init_output_dir()
        self.logger = self.init_logger()
        self.tboard = self.init_tboard()
        
    def init_output_dir(self):
        # Create output directories
        path = dict()
        for subdir in {'weight', 'intermediate_output'}:
            _sub = os.path.join(self.config['output_path'], self.train_id, subdir)
            path[subdir] = _sub
            os.system('mkdir -p %s' % _sub)
        return path
    
    def init_logger(self):
        return Logger(metrics=self.metrics.keys())
    
    def init_tboard(self):
        return TensorboardHelper(log_path=os.path.join(self.config['output_path'], self.train_id))
        
    def train_phase(self, dataloader, epoch):
        # Record loss and metrics during training
        running_loss = meter.AverageValueMeter()
        running_metrics = self.metrics

        # Start training
        self.net.train()
        progress_bar = tqdm(dataloader)
        for i, batch in enumerate(progress_bar):
            # Load inputs and labels
            inps = batch['input'][0].to(self.device)
            lbls = batch['target'][0].to(self.device)

            # Clear out gradients from previous iteration
            self.optimizer.zero_grad()
            # Get network outputs
            outs = self.net(inps)
            # Calculate the loss
            loss = self.criterion(outs, lbls)
            # Calculate the gradients
            loss.backward()
            # Performing backpropagation
            self.optimizer.step()
            
            # Update loss
            running_loss.add(loss.item())

            # Log loss
            self.logger.update_loss('train', loss.item())
            self.tboard.update_loss('train', loss.item(), epoch * len(dataloader) + i)

            # Update metrics
            lbls = lbls.detach().cpu()
            outs = outs.detach().cpu()
            for mid, metric in running_metrics.items():
                val = metric.calculate(outs, lbls, is_track=True)
                self.logger.update_metrics('train', mid, val)

            # Log in interval
            if i % self.config['log_step'] == 0:
                # Update progress bar
                progress_bar.set_description_str(
                    '[TRAIN] loss {:.5f}'.format(running_loss.value()[0]))
                
                # Reset metrics meter
                running_loss.reset()
                for metric in running_metrics.values():
                    metric.reset()

    @torch.no_grad()
    def val_phase(self, dataloader, epoch):
        # Record loss and metrics
        _loss = meter.AverageValueMeter()
        _metrics = self.metrics
        for metric in _metrics.values(): 
            metric.reset()

        # Start validating
        self.net.eval()
        progress_bar = tqdm(dataloader)
        progress_bar.set_description_str('[ VAL ]')
        for i, batch in enumerate(progress_bar):
            # Load inputs and labels
            inps = batch['input'][0].to(self.device)
            lbls = batch['target'][0].to(self.device)

            # Get network outputs
            outs = self.net(inps)

            # Calculate the loss
            loss = self.criterion(outs, lbls)

            # Update loss
            _loss.add(loss.item())

            # Update metrics
            lbls = lbls.detach().cpu()
            outs = outs.detach().cpu()
            for metric in _metrics.values():
                metric.calculate(outs, lbls, is_track=True)

        # Calculate evaluation result
        avg_loss = _loss.value()[0]
        avg_metrics = { mid: metric.value() for mid, metric in _metrics.items() }

        # Print results
        print('loss:', avg_loss)
        for mid, metric in _metrics.items():
            print(metric.summary())
            print('{}: {}'.format(mid, avg_metrics[mid]))

        # Log loss
        self.logger.update_loss('val', avg_loss)
        self.tboard.update_loss('val', avg_loss, epoch)
        
        # Log metrics
        for mid, metric in _metrics.items():
            self.logger.update_metrics('val', mid, avg_metrics[mid])
            self.tboard.update_metrics('val', mid, avg_metrics[mid], epoch)
        
    def save_checkpoint(self, cfg, epoch):
        save_data = {
            'epoch': epoch,
            'model_state_dict': self.net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'log': self.logger.get_logs(),
            'config': self.full_config
        }
        
        weight_output_path = self.output_path['weight']

        if cfg.get('best_loss', False):
            # Get validation loss
            val_loss = self.logger.get_latest_loss('val')
            best_loss = self.logger.get_best_loss()
            
            # Save model with lowest val_loss
            if val_loss < best_loss:
                print('loss improved from %.6f to %.6f. Saving weights...' % (best_loss, val_loss))
                torch.save(save_data, os.path.join(weight_output_path, '{}_best_loss.pth'.format(self.train_id)))
                self.logger.update_best_loss(val_loss)
            else:
                print('loss did not improve from %.6f.' % (best_loss))

        if cfg.get('best_metrics', False):
            for metric in self.metrics.keys():
                if cfg['best_metrics'].get(metric, False):
                    # Get validation metric
                    val_metric = self.logger.get_latest_metrics('val', metric)
                    best_metric = self.logger.get_best_metrics(metric)

                    # Save model with highest validation metric
                    if val_metric > best_metric:
                        print('%s improved from %.6f to %.6f. Saving weights...' % (metric, best_metric, val_metric))
                        torch.save(save_data, os.path.join(weight_output_path, '{}_best_{}.pth'.format(self.train_id, metric)))
                        self.logger.update_best_metrics(metric, val_metric)
                    else:
                        print('%s did not improve from %.6f.' % (metric, best_metric))
                
        if cfg.get('current', False):
            # Save current model
            torch.save(save_data, os.path.join(weight_output_path, '{}.pth'.format(self.train_id)))
            print('Saving current model...')

    def train(self, train_dataloader, val_dataloader):
        for epoch in range(self.config['nepochs']):        
            print('Epoch {:>3d}'.format(epoch))
            print('-'*20)

            # 1. Training phase
            # --------------------------------------------------------------

            self.train_phase(dataloader=train_dataloader,
                             epoch=epoch)

            print('-'*20)

            # 2. Validation phase
            # --------------------------------------------------------------

            # TODO: think about validation and save step, because it affects
            #       each other and also affects scheduler in some cases
            
            if epoch % self.config['val_step'] == 0:
                self.val_phase(dataloader=val_dataloader,
                               epoch=epoch)

                print('-'*20)

            # 3. Learning rate scheduling
            # --------------------------------------------------------------
            
            # Based on validation loss
            val_loss = self.logger.get_latest_loss('val')
            self.scheduler.step(val_loss)
       
            # 4. Saving checkpoints
            # --------------------------------------------------------------

            if epoch % self.config['save_step'] == 0:
                self.save_checkpoint(cfg=self.config['save'], 
                                     epoch=epoch)
            
            # 5. Visualizing some examples
            # --------------------------------------------------------------
            
            print('=' * 30)