import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np

import fasteners
import os
from time import sleep
from copy import deepcopy


class QNet(nn.Module):
    def __init__(self, path, device='cpu'):
        super(QNet, self).__init__()
        self.path = path
        self.device = device

        self.vis_layers = nn.Sequential(
            # (84, 84, *) -> (20, 20, 16)
            nn.Conv2d(4, 32, kernel_size=8, stride=4),
            nn.ReLU(True),
            # (20, 20, 16) -> (9, 9, 32)
            nn.Conv2d(32, 32, kernel_size=4, stride=2),
            nn.ReLU(True),
            # (9, 9, 32) -> (7, 7, 64)
            nn.Conv2d(32, 64, kernel_size=3, stride=1),
            nn.ReLU(True)
            #Flatten(),
        )

        self.l1 = nn.Sequential(
            nn.Linear(7 * 7 * 64, 512),
            nn.ReLU(True)
        )

        self.val = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(True),
            nn.Linear(256, 1)
        )

        self.adv = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(True),
            nn.Linear(256, 6)
        )

    def forward(self, state):
        if len(state.size()) == 5:
            seq_size, batch_size, _, _, _ = state.size()
        else:
            seq_size = 1
            batch_size, c, h, w = state.size()
        state = state.view(-1, 4, 84, 84)

        h = self.vis_layers(state).view(-1, 7*7*64)
        h = self.l1(h)

        val = self.val(h)
        adv = self.adv(h)
        q_val = val + adv - adv.mean(1, keepdim=True)
        if seq_size > 1:
            q_val = q_val.view(-1, 6)

        return q_val
    
    def save(self):
        lock = fasteners.ReaderWriterLock()
        while True:
            try:
                with lock.write_lock():
                    torch.save(deepcopy(self).cpu().state_dict(), self.path)
                    sleep(0.1)
                return
            except:
                sleep(np.random.random()+1)
    
    def load(self):
        lock = fasteners.ReaderWriterLock()
        while True:
            try:
                with lock.read_lock():
                    state_dict = torch.load(self.path)
                self.load_state_dict(state_dict)
                return
            except:
                sleep(np.random.random()+1)
    