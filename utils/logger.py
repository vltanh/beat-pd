import numpy as np
from torch.utils.tensorboard import SummaryWriter

class Logger():
    def __init__(self, metrics=['acc']):
        self.metrics = metrics
        self.logger = self.init_logger_dict(metrics)
        
    def init_logger_dict(self, metrics):
        logger = dict()
        for phase in ['train', 'val']:
            logger[phase] = dict()
            logger[phase]['loss'] = []
            logger[phase]['metrics'] = dict()
            for metric in metrics:
                logger[phase]['metrics'].setdefault(metric, [])
        logger['best'] = dict()
        logger['best']['loss'] = np.inf
        logger['best']['metrics'] = dict()
        for metric in metrics:
            logger['best']['metrics'].setdefault(metric, 0.0)
        return logger
    
    def update_loss(self, phase, value):
        self.logger[phase]['loss'].append(value)
    
    def update_metrics(self, phase, metric, value):
        self.logger[phase]['metrics'][metric].append(value)
    
    def get_logs(self):
        return self.logger
    
    def get_best_loss(self):
        return self.logger['best']['loss']
    
    def update_best_loss(self, value):
        self.logger['best']['loss'] = value
    
    def get_best_metrics(self, metric):
        assert metric in self.metrics
        return self.logger['best']['metrics'][metric]
    
    def update_best_metrics(self, metric, value):
        self.logger['best']['metrics'][metric] = value
    
    def get_latest_loss(self, phase):
        return self.logger[phase]['loss'][-1]
    
    def get_latest_metrics(self, phase, metric):
        return self.logger[phase]['metrics'][metric][-1]
    
class TensorboardHelper():
    def __init__(self, log_path):
        self.writer = SummaryWriter(log_dir=log_path)
    
    def update_loss(self, phase, value, step):
        self.writer.add_scalar('{}/loss'.format(phase), value, step)
    
    def update_metrics(self, phase, metric, value, step):
        self.writer.add_scalar('{}/{}'.format(phase, metric), value, step)