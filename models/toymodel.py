import torch
import torch.nn as nn

from models.modules.MultiHeadedAttention import MultiHeadedAttention
from models.modules.utils import clones

class AxisBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 64, (10, 1), stride=(10, 1)),
            nn.ReLU(),
            nn.Conv2d(64, 128, (5, 1), stride=(5, 1)),
            nn.ReLU()
        )
        self.attn = MultiHeadedAttention(4, 128, 0.0)
        self.lstm = nn.LSTM(128, 64, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(64, 128),
            nn.ReLU()
        )
    
    def forward(self, x):
        # x => [B, 1, L, 1]
        x = self.cnn(x)
        # x => [B, C, L', 1]
        x = x.squeeze(-1).transpose(1, 2)
        # x => [B, L', C]
        # x = self.attn(x, x, x)
        _, (x, _) = self.lstm(x)
        x = x.reshape(-1, 64)
        # x => [B, 64]
        x = self.ffn(x)
        # x => [B, 128]
        return x

class ToyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.axis = clones(AxisBlock(), 3)
        self.cls = nn.Linear(384, 3)
    
    def forward(self, inp):
        x = self.axis[0](inp[:, :, :, 0].unsqueeze(-1))
        y = self.axis[1](inp[:, :, :, 1].unsqueeze(-1))
        z = self.axis[2](inp[:, :, :, 2].unsqueeze(-1))
        
        out = torch.cat([x, y, z], dim=1)
        out = self.cls(out)

        return out