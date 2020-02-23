class Getter():
    def __init__(self):
        self.elems = dict()
        
    def add(self, key, value):
        assert key not in self.elems, '{} already exists.'.format(key)
        self.elems[key] = value
    
    def check(self, type):
        assert type in self.elems, '{} not found.'.format(type)
    
    def get(self, config, **kwargs):
        self.check(config['type'])
        return self.elems[config['type']](**config['config'], **kwargs)
    
# ---------------------------------------------------------------------------------

from models.toymodel import ToyModel

class NetworkGetter(Getter):
    def __init__(self):
        super().__init__()
        self.add('ToyModel', ToyModel)
        
# ---------------------------------------------------------------------------------

from workers.losses import CrossEntropyLoss
from torch.nn import MSELoss
        
class LossGetter(Getter):
    def __init__(self):
        super().__init__()
        self.add('CrossEntropy', CrossEntropyLoss)
        self.add('MSE', MSELoss)
        
# ---------------------------------------------------------------------------------
    
from torch import optim
    
class OptimizerGetter(Getter):
    def __init__(self):
        super().__init__()
        self.add('SGD', optim.SGD)
        self.add('Adam', optim.Adam)
        
# ---------------------------------------------------------------------------------
    
from torch import optim
    
class SchedulerGetter(Getter):
    def __init__(self):
        super().__init__()
        self.add('ReduceLROnPlateau', optim.lr_scheduler.ReduceLROnPlateau)
        
# ---------------------------------------------------------------------------------
        
from torch.utils import data

from datasets.cispd import CISPDTrain

class DatasetGetter(Getter):
    def __init__(self):
        super().__init__()
        self.add('CISPDTrain', CISPDTrain)
    
    def get(self, config, **kwargs):
        dataset = super().get(config, **kwargs)
        dataloader = data.DataLoader(dataset, **config['loader_config'])
        return dataloader
    
# ---------------------------------------------------------------------------------

class VisualizerGetter(Getter):
    def __init__(self):
        super().__init__()
        
# ---------------------------------------------------------------------------------

class MetricGetter(Getter):
    def __init__(self):
        super().__init__()