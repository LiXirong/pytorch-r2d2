import torch
import torch.nn as nn
import numpy as np


class QNet(nn.Module):
    def __init__(self, device='cpu'):
        super(QNet, self).__init__()
        self.device = device
        self.hs, self.cs = None, None

        self.vis_layers = nn.Sequential(
            # (84, 84, *) -> (20, 20, 16)
            nn.Conv2d(4, 32, kernel_size=8, stride=4),
            nn.ReLU(True),
            # (20, 20, 16) -> (9, 9, 32)
            nn.Conv2d(32, 32, kernel_size=4, stride=2),
            nn.ReLU(True),
            # (9, 9, 32) -> (7, 7, 64)
            nn.Conv2d(32, 32, kernel_size=3, stride=1),
            nn.ReLU(True)
        )

        self.lstm = nn.LSTMCell(7*7*32, 256)

        self.val = nn.Sequential(
            nn.Linear(256, 256),
            nn.ReLU(True),
            nn.Linear(256, 1)
        )

        self.adv = nn.Sequential(
            nn.Linear(256, 256),
            nn.ReLU(True),
            nn.Linear(256, 6)
        )

    def forward(self, state, return_hs_cs=False):
        if len(state.size()) == 5:
            seq_size, batch_size, _, _, _ = state.size()
        else:
            seq_size = 1
            batch_size, c, h, w = state.size()
        state = state.view(-1, 4, 84, 84)

        hs = self.vis_layers(state).view(seq_size, batch_size, 7*7*32)

        if self.hs is None:
            self.hs = torch.zeros(batch_size, 256).to(self.device)
            self.cs = torch.zeros(batch_size, 256).to(self.device)

        hs_seq = []
        cs_seq = []
        for h in hs:
            self.hs, self.cs = self.lstm(h, (self.hs, self.cs))
            hs_seq.append(self.hs)
            cs_seq.append(self.cs)

        hs_seq = torch.cat(hs_seq, dim=0)
        cs_seq = torch.cat(cs_seq, dim=0)

        val = self.val(hs_seq)
        adv = self.adv(hs_seq)
        q_val = val + adv - adv.mean(1, keepdim=True)
        if seq_size > 1:
            q_val = q_val.view(-1, 6)
        
        hs_seq = hs_seq.detach().cpu().numpy()
        cs_seq = cs_seq.detach().cpu().numpy()

        if return_hs_cs:
            return q_val, hs_seq, cs_seq
        else:
            return q_val
    
    def reset(self):
        self.hs, self.cs = None, None
    
    def set_state(self, hs, cs):
        self.hs, self.cs = hs, cs
    
    def get_state(self):
        return self.hs.detach().cpu().numpy(), self.cs.detach().cpu().numpy()