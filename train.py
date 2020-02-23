import os
import argparse
import yaml
import time

from workers.getter import NetworkGetter, LossGetter, OptimizerGetter, SchedulerGetter, DatasetGetter, VisualizerGetter, MetricGetter
from workers.trainer import Trainer
from datasets.cispd import CISPDTrain

import torch
import numpy as np
import random

manualSeed = 3698

def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)

def get_device(config):
    dev_id = 'cuda:{}'.format(config['gpus']) \
             if torch.cuda.is_available() and config.get('gpus', None) is not None \
             else 'cpu'
    return torch.device(dev_id), dev_id

def get_train_id(_id, current_time):
    return '%s_%s' % (_id, current_time)

def train(config):
    '''
        The training workflow consists of:
            1. Specify the device to train on (no parallelism yet);
            2. Specify train id, which is the id in config + timestamp;
            3. Load configuration from checkpoint (if specified);
            4. Get network, criterion, optimizer, and callbacks (learning rate scheduler).
               Load pretrained weights if necessary;
            5. Create trainer using all the above;
            6. Get train/val datasets;
            7. Perform training on train/val datasets.
    '''
    
    # TODO: parallelize training
    
    # Specify device
    device, dev_id = get_device(config)
    
    # -----------------------------------------------------------------
    
    # Training start time
    current_time = time.strftime('%b%d_%H-%M-%S', time.gmtime())
    print('Training starts at', current_time)
     
    # Get train id
    train_id = get_train_id(config['id'], current_time)
            
    # -----------------------------------------------------------------

    # TODO: think about continue training on a different datasets
    
    # Load checkpoint configuration (if specified)
    checkpoint = None
    if config.get('checkpoint', None) is not None:
        print('Continue from checkpoint at %s' % config['checkpoint'])
        checkpoint = torch.load(config['checkpoint'], map_location=dev_id)
        # Override config
        # TODO: what to load (arch is a must, what about the rest)?
        for cfg_item in ['arch', 'loss', 'optimizer', 'scheduler']:
            config[cfg_item] = checkpoint['config'][cfg_item]

    # -----------------------------------------------------------------

    set_seed(manualSeed)

    # Define network
    net = NetworkGetter().get(config=config['arch']).to(device)

    # Define loss
    criterion = LossGetter().get(config=config['loss']).to(device)

    # Define optim
    optimizer = OptimizerGetter().get(params=net.parameters(), 
                                      config=config['optimizer'])
    
    # Define learning rate scheduler
    scheduler = SchedulerGetter().get(optimizer=optimizer,
                                      config=config['scheduler'])
    
    metrics = { metric: MetricGetter().get(config=cfg) 
                for metric, cfg in config['metrics'].items() }
    
    # -----------------------------------------------------------------
    
    # TODO: Summarizer network
    
    # Print network
    print(net)
    print('=' * 30)
    
    # -----------------------------------------------------------------

    # Load pretrained weights (if specified)
    if checkpoint is not None:
        net.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
    # ------------------------------------------------------------------
    
    # Create trainer
    trainer = Trainer(train_id=train_id,
                      config=config,
                      net=net,
                      optimizer=optimizer,
                      criterion=criterion,
                      scheduler=scheduler,
                      metrics=metrics,
                      device=device)

    # -----------------------------------------------------------------
    
    set_seed(manualSeed)

    # Load datasets
    train_dataset = CISPDTrain(data_path='data/cis-pd/training_data', 
                               label_path='data/cis-pd/data_labels/CIS-PD_Training_Data_IDs_Labels.csv')
    train_dataset, val_dataset = torch.utils.data.random_split(train_dataset, 
                                                               [len(train_dataset) - len(train_dataset) // 5, len(train_dataset) // 5])
    train_dataloader = torch.utils.data.DataLoader(train_dataset, num_workers=6, batch_size=1, shuffle=True)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, num_workers=6, batch_size=1)
    
    # -----------------------------------------------------------------
    
    # Training
    trainer.train(train_dataloader=train_dataloader,
                  val_dataloader=val_dataloader)
    
    # -----------------------------------------------------------------
            
    current_time = time.strftime('%b%d_%H-%M-%S', time.gmtime())
    print('Training finishes at', current_time)

def main():
    # Parse argument
    parser = argparse.ArgumentParser()
    parser.add_argument('--config')
    args = parser.parse_args()

    # Load configurations
    assert args.config, 'Config file not found'
    cfg = yaml.load(open(args.config), Loader=yaml.Loader)
    
    # Train on loaded config
    train(config=cfg)
    
if __name__ == "__main__":
    main()
