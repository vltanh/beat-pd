import torch
import torch.nn as nn

from models.modules.MultiHeadedAttention import MultiHeadedAttention
from models.modules.PositionwiseFeedForward import PositionwiseFeedForward
from models.modules.Encoder import Encoder, EncoderLayer
from models.modules.utils import clones

# from torch.nn import MultiHeadedAttention

# Exp 1.1: MultiHeadedAttention(4, 128, 0.0), x[:, -1]
# Exp 1.2: ................................., torch.mean(x, dim=1)                                                          => =1.1
# Exp 2  : LSTM(128, 128)                                                                                                   => <1.1
# Exp 3.1: Exp1 -> Exp2                                                                                                     => <1.1
# Exp 3.2: Exp2 -> Exp1                                                                                                     => ~3.1

# Exp 4.1: Encoder(EncoderLayer(128, Exp1.Attn, PositionwiseFF(128, 256, 0.0), 0.0), 1), x + f(self.norm(x)), x[:, -1]      => <1.1
# Exp 4.2: ................................................................................................., torch.mean(x) => <4.1
# Exp 5.1: ............................................................................, self.norm(x + f(x)), x[:, -1]      => <4.2
# Exp 5.2: ................................................................................................., torch.mean(x) => ~4.2

# Exp 6  : Concat, Mean then Attention

class AxisBlock(nn.Module):
    def __init__(self, method):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 64, (10, 1), stride=(10, 1)),
            nn.ReLU(),
            nn.Conv2d(64, 128, (5, 1), stride=(5, 1)),
            nn.ReLU()
        )

        # self.attn = MultiHeadedAttention(1, 128, 0.0)

        nheads = 4
        nfeatures = 128
        dropout = 0.0
        attn = MultiHeadedAttention(h=nheads, d_model=nfeatures, dropout=dropout)
        ff = PositionwiseFeedForward(nfeatures, d_ff=4*nfeatures, dropout=dropout)
        self.attn = Encoder(EncoderLayer(nfeatures, attn, ff, dropout), 1)

        self.lstm = nn.LSTM(128, 128, batch_first=True)

        self.method = method
    
    def forward(self, x):
        # x => [B, 1, L, 1]

        x = self.cnn(x)
        # x => [B, C, L', 1]

        x = x.squeeze(-1).transpose(1, 2)
        # x => [B, L', C]

        # x, _ = self.lstm(x)
        # if self.method == 'attention':
            # x = self.attn(x, None)
            # x = x[:, -1]
            # x = torch.mean(x, dim=1)
        # elif self.method == 'lstm':
            # _, (x, _) = self.lstm(x)
            # x = x.reshape(-1, 128)
        # x => [B, 128]

        return x.unsqueeze(-1)

class ToyModel(nn.Module):
    def __init__(self, method):
        super().__init__()
        self.axis = clones(AxisBlock(method), 3)
        self.cls = nn.Linear(128, 3)

        nheads = 4
        nfeatures = 128
        dropout = 0.1
        attn = MultiHeadedAttention(h=nheads, d_model=nfeatures, dropout=dropout)
        ff = PositionwiseFeedForward(nfeatures, d_ff=4*nfeatures, dropout=dropout)
        self.attn = Encoder(EncoderLayer(nfeatures, attn, ff, dropout), 1)
    
    def forward(self, inp):
        x = self.axis[0](inp[:, :, :, 0].unsqueeze(-1))
        y = self.axis[1](inp[:, :, :, 1].unsqueeze(-1))
        z = self.axis[2](inp[:, :, :, 2].unsqueeze(-1))
        
        out = torch.cat([x, y, z], dim=-1)
        # out = out.reshape(out.size(0), out.size(1), -1, 3)
        out = torch.mean(out, dim=-1)
        # out => [B, L', 3 x C]
        out = self.attn(out, None)
        # out => [B, L', 3 x C]
        out = torch.mean(out, dim=1)
        # out => [B, 1, 3 x C]

        out = self.cls(out)

        return out