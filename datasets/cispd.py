import torch
import torch.utils.data as data

import pandas as pd
import matplotlib.pyplot as plt

import os

class CISPDTrain(data.Dataset):
    def __init__(self, data_path: str, label_path: str):
        super().__init__()
        self.ids = [os.path.splitext(x)[0] for x in os.listdir(data_path)]
        self.labels = pd.read_csv(label_path)
        self.root = data_path

        self.preprocess()

    def __getitem__(self, i: int):
        # Get ID
        _id = self.ids[i]
        
        # Get data
        data = pd.read_csv(os.path.join(self.root, f'{_id}.csv'))
        timestamp = torch.Tensor(data['Timestamp'].values)
        accels = torch.Tensor(data[['X', 'Y', 'Z']].values)
        _input = self.process_input(accels, timestamp)

        # Get label
        _, _, on_off, dyskinesia, tremor = self.labels.loc[self.labels['measurement_id'] == _id].values[0]
        _target = self.process_label(on_off, dyskinesia, tremor)


        # os.system(f'mkdir -p vis/on_off/{on_off}/')
        # os.system(f'mkdir -p vis/dyskinesia/{dyskinesia}/')
        # os.system(f'mkdir -p vis/tremor/{tremor}/')

        # plt.plot(timestamp, accels[:,0])
        # plt.plot(timestamp, accels[:,1])
        # plt.plot(timestamp, accels[:,2])
        # plt.legend(['X', 'Y', 'Z'])
        # plt.tight_layout()
        # # plt.show()
        # plt.savefig(f'vis/on_off/{on_off}/{_id}')
        # plt.savefig(f'vis/dyskinesia/{dyskinesia}/{_id}')
        # plt.savefig(f'vis/tremor/{tremor}/{_id}')
        # plt.close()

        return {
            'input': _input,
            'target': _target
        }

    def __len__(self):
        return len(self.ids)

    def preprocess(self):
        # self.labels[['on_off', 'dyskinesia', 'tremor']] = self.labels[['on_off', 'dyskinesia', 'tremor']].fillna(-4)
        self.labels = self.labels.dropna()
        self.ids = self.labels['measurement_id'].values

    def process_input(self, accels, timestamp):
        _timestamp = timestamp
        _accels = accels.unsqueeze(0)
        return _accels, _timestamp

    def process_label(self, on_off, dyskinesia, tremor):
        # _on_off = torch.FloatTensor([on_off]) / 4.0
        # _dyskinesia = torch.FloatTensor([dyskinesia]) / 4.0
        # _tremor = torch.FloatTensor([tremor]) / 4.0
        return torch.FloatTensor([on_off, dyskinesia, tremor]).unsqueeze(0) / 4.0