import torch.nn as nn
import torch
import numpy as np
import torch.nn.functional as F

class SEBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super(SEBlock, self).__init__()
        self.fc1 = nn.Linear(channels, channels // reduction, bias=False)
        self.fc2 = nn.Linear(channels // reduction, channels, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        batch_size, channels, _, _ = x.size()
        # Global Average Pooling
        y = torch.mean(x, dim=[2, 3])
        # Fully connected layers
        y = self.fc1(y)
        y = nn.ReLU()(y)
        y = self.fc2(y)
        y = self.sigmoid(y).view(batch_size, channels, 1, 1)
        return x * y



class CFCE(nn.Module):
    def __init__(self, in_channel=3):
        super(CFCE, self).__init__()
        self.relu = nn.ReLU()
        self.bn = nn.BatchNorm2d(in_channel)
        self.stage1 = nn.Sequential(
            nn.Conv2d(in_channel, in_channel, 3, 1, bias=False),
            nn.BatchNorm2d(in_channel),
            nn.ReLU()
        )
        self.stage2 = nn.Sequential(
            nn.Conv2d(in_channel, in_channel, 3, 1, bias=False),
            nn.BatchNorm2d(in_channel),
            nn.ReLU()
        )

    def forward(self, fa, fb):
        (b1, c1, h1, w1), (b2, c2, h2, w2) = fa.size(), fb.size()
        assert c1 == c2
        cos_sim = F.cosine_similarity(fa, fb, dim=1)
        cos_sim = cos_sim.unsqueeze(1)
        fa = fa + fb * cos_sim
        fb = fb + fa * cos_sim
        fa = self.relu(fa)
        fb = self.relu(fb)

        return fa, fb




