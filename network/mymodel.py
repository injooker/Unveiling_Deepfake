import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import torchvision
import torch.utils.model_zoo as model_zoo
from torch.nn import init
from components.linear_fusion import HdmProdBilinearFusion
from network.xception import TransferModel, Xception
from components.srm_conv import SRMConv2d_simple, SRMConv2d_Separate
from network.modules import *
import time
class Xception_Net(nn.Module):
    def __init__(self,num_classes=1000, inc=3):
        super().__init__()
        self.params = {
            'location': {
                'size': 19,
                'channels': [64, 128, 256, 728, 728, 728],
                'mid_channel': 512
            },
            'cls_size': 10,
            'HBFusion': {
                'hidden_dim': 2048,
                'output_dim': 4096,
            }
        }
        self.xception = TransferModel('xception',dropout=0.5, inc=3, return_fea=True)
      
    def features(self, input):
        # input = input.permute(0, 3, 1, 2)  # 将输入从 [N, H, W, C] 调整为 [N, C, H, W]
      
        input = input.float().to('cuda')  # 转换为 FloatTensor 并移动到 GPU
        self.xception.model = self.xception.model.to('cuda')  # 确保模型也在同一设备  
        x6 = self.xception.model.features(input)
        return x6
        
    def classifier(self, features):
        x = self.xception.model.relu(features)

        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = x.view(x.size(0), -1)
        out = self.xception.model.last_linear(x)
        return out

    def forward(self, input):
        x = self.features(input)
        out = self.classifier(x)
        return out
