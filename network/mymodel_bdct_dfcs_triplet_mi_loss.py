import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import numpy as np
import math
import torchvision
import torch.utils.model_zoo as model_zoo
from torchjpeg import dct
from torch.nn import init
from components.linear_fusion import HdmProdBilinearFusion
from network.xception import TransferModel, Xception
from network.modules import *
class DynamicChannelPruner(nn.Module):
    def __init__(self, num_channels=192, num_keep=64):

        super().__init__()
        self.num_channels = num_channels
        self.num_keep = num_keep

        self.pool = nn.AdaptiveAvgPool2d(1) 
        self.conv = nn.Conv2d(num_channels, num_channels, kernel_size=1)
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=1, stride=1, padding=0)
        self.conv_r = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=1, stride=1, padding=0)
        self.conv_l = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(3)
        self.fc = nn.Linear(num_channels, num_channels)
        self.row_pool = nn.AdaptiveAvgPool2d((8, 1)) 
        self.col_pool = nn.AdaptiveAvgPool2d((1, 8))  
        self.a = nn.Parameter(torch.tensor(0.5), requires_grad=True)
    def forward(self, x_freq):
        b, c, f, h, w = x_freq.shape 


        x_pooled = self.pool(x_freq.reshape(b * c, f, h, w))
        x_pooled = x_pooled.reshape(b, c, f)  

        x_conv = self.conv(x_freq.reshape(b * c, f, h, w)) 
        x_conv = self.pool(x_conv.reshape(b * c, f, h, w))

        x_conv = x_conv.reshape(b, c, f) 
        scores = torch.sigmoid(self.fc(x_conv))  
        x_pooled_2=x_pooled.reshape(b,c,8,8)
        
        x_row_pooled = self.row_pool(x_pooled_2) 
        x_row_pooled = x_row_pooled.permute(0,1,3,2) 
        x_col_pooled = self.col_pool(x_pooled_2)  
        x_row_and_col_pooled = torch.cat([x_row_pooled,x_col_pooled],dim = 2)
        x_row_and_col_pooled = self.conv1(x_row_and_col_pooled)
        x_row_and_col_pooled = self.bn1(x_row_and_col_pooled)
        x_row_and_col_pooled = torch.sigmoid(x_row_and_col_pooled)
        x_r, x_l = torch.split(x_row_and_col_pooled, [1, 1], dim=2)
        x_r = x_r.permute(0,1,3,2)
        a_r = self.conv_r(x_r)
        a_r = torch.sigmoid(a_r)
        a_l = self.conv_l(x_l)
        a_l = torch.sigmoid(a_l)
        x_att = torch.matmul(a_r, a_l) 
        x_att = x_att.view(b, c, -1)  
        final_scores = self.a * x_att + (1 - self.a) * scores 
        top_scores, top_indices = torch.topk(final_scores, self.num_keep, dim=2, largest=True, sorted=True) 
        
        
        mask = torch.zeros_like(scores)

        for i in range(b):
            for j in range(c):  
                mask[i, j, top_indices[i, j]] = 1 

        mask = mask.unsqueeze(-1).unsqueeze(-1)  

        x_pruned = x_freq * mask  # [B, C, F, H, W]
        mask_2k = torch.zeros_like(scores)
        for i in range(b):
            for j in range(c): 
                mask_2k[i, j, top_indices[i, j][self.num_keep:]] = 1 

        mask_2k = mask_2k.unsqueeze(-1).unsqueeze(-1)  # [B, C, F, 1, 1]

        x_pruned_2k = x_freq * mask_2k  # [B, C, F, H, W]

        return x_pruned, x_pruned_2k
        



class Xception_Net(nn.Module):
    def __init__(self,num_classes=2, inc=3):
        super().__init__()
        self.xception = TransferModel('xception',num_out_classes=2)
        self.xception_fre = TransferModel('xception',num_out_classes=2)
        self.xception_fre_2 = TransferModel('xception',num_out_classes=2)
        self.cmc = CFCE(in_channel=2048)
        self.HBFusion_y = HdmProdBilinearFusion(dim1=(64+128+256+728+728), dim2=2048, 
                        hidden_dim=2048, output_dim=2048)
        self.HBFusion_x = HdmProdBilinearFusion(dim1=(64+128+256+728+728), dim2=2048, 
                        hidden_dim=2048, output_dim=2048)
        self.relu_all = nn.ReLU(inplace=True)
        self.relu_x = nn.ReLU(inplace=True)
        self.relu_y = nn.ReLU(inplace=True)
        self.relu_z = nn.ReLU(inplace=True)
        self.pruner = DynamicChannelPruner(num_channels=64, num_keep=2) 
        self.channel_compress = ChannelCompress(in_ch=4096, out_ch=2048, dropout=0.5)
        self.base_line=nn.Linear(6144,2)
        self.base_line_x=nn.Linear(4096,2)
        self.base_line_y=nn.Linear(4096,2)
        self.base_line_z=nn.Linear(4096,2)
    def pad_max_pool(self, x):
        b, c, h, w = x.size()
        padding = abs(h % 10 - 10) % 10
        pad = nn.ReplicationPad2d(padding=(padding // 2, (padding + 1) // 2, padding // 2, (padding + 1) // 2)).to(x.device)
        x = pad(x)
        b, c, h, w = x.size()
        
        max_pool = nn.MaxPool2d(kernel_size=h // 10, stride=h // 10, padding=0)
        return max_pool(x)
    def dct_transform(self,x, chs_remove=[0,1,2], chs_pad=True,
                  size=8, stride=8, pad=0, dilation=1, ratio=8):
        """
            Transform a spatial image into its frequency channels.
            Prune low-frequency channels if necessary.
        """
        assert isinstance(x, torch.Tensor), f"Expected x to be a tensor, but got {type(x)}"
        assert x.dim() == 4, f"Expected input to have 4 dimensions (B, C, H, W), but got {x.dim()}"
        # assert x is a (3, H, W) RGB image
        assert x.shape[1] == 3
        # up-sample
        x = F.interpolate(x, scale_factor=ratio, mode='bilinear', align_corners=True)
        x = x * 255
        x = dct.to_ycbcr(x)
        x = x - 128
        b, c, h, w = x.shape
        n_block = h // stride
        x = x.view(b * c, 1, h, w)
        x = F.unfold(x, kernel_size=(size, size), dilation=dilation, padding=pad, stride=(stride, stride))
        x = x.transpose(1, 2)
        x = x.view(b, c, -1, size, size)
        x_freq = dct.block_dct(x)
        x_freq = x_freq.view(b, c, n_block, n_block, size * size).permute(0, 1, 4, 2, 3)

        x_pruned,x_pruned_2 = self.pruner(x_freq)  # Select the most important channels

        x_pruned = x_pruned.reshape(b, -1, n_block, n_block)
        x_pruned_2 = x_pruned_2.reshape(b, -1, n_block, n_block)
        return x_pruned,x_pruned_2


    def idct_transform(self,x, size=8, stride=8, pad=0, dilation=1, ratio=8):
        """
        The inverse of DCT transform.
        Transform frequency channels (must be 192 channels, can be padded with 0) back to the spatial image.
        """

        b, _, h, w = x.shape

        x = x.view(b, 3, 64, h, w)
        x = x.permute(0, 1, 3, 4, 2)
        x = x.view(b, 3, h * w, 8, 8)
        x = dct.block_idct(x)
        x = x.view(b * 3, h * w, 64)
        x = x.transpose(1, 2)
        x = F.fold(x, output_size=(299 * ratio, 299 * ratio),
                kernel_size=(size, size), dilation=dilation, padding=pad, stride=(stride, stride))
        x = x.view(b, 3, 299 * ratio, 299 * ratio)
        x = x + 128
        x = dct.to_rgb(x)
        x = x / 255
        
        x = F.interpolate(x, scale_factor=1 / ratio, mode='bilinear', align_corners=True)
        x = x.clamp(min=0.0, max=1.0)
        return x
    def features(self, input):
        input = input.float().to('cuda')  # 转换为 FloatTensor 并移动到 GPU


        x_freq,x_freq_2 = self.dct_transform(input, chs_remove=[20], chs_pad=True,size=8, stride=8, pad=0, dilation=1, ratio=8)    

        x_reconstructed = self.idct_transform(x_freq, size=8, stride=8, pad=0, dilation=1, ratio=8)

        x_reconstructed_2 = self.idct_transform(x_freq_2, size=8, stride=8, pad=0, dilation=1, ratio=8)
      
        z6 = self.xception_fre_2.model.features(x_reconstructed_2)

        x0 = self.xception.model.fea_part1_0(input)
        y0 = self.xception_fre.model.fea_part1_0(x_reconstructed)
      
        x1 = self.xception.model.fea_part1_1(x0) 
        y1 = self.xception_fre.model.fea_part1_1(y0) 
        
        x2 = self.xception.model.fea_part1_2(x1)
        y2 = self.xception_fre.model.fea_part1_2(y1)
     
        x3 = self.xception.model.fea_part1_3(x2)   
        y3 = self.xception_fre.model.fea_part1_3(y2)
       
        x4 = self.xception.model.fea_part2_0(x3) 
        y4 = self.xception_fre.model.fea_part2_0(y3)   
      
        x5 = self.xception.model.fea_part2_1(x4)   
        y5 = self.xception_fre.model.fea_part2_1(y4)  
     
        x6 = self.xception.model.fea_part3(x5)
        y6 = self.xception_fre.model.fea_part3(y5)
        
        x0m = self.pad_max_pool(x0)
        x1m = self.pad_max_pool(x1)
        x2m = self.pad_max_pool(x2)
        x3m = self.pad_max_pool(x3)
        x5m = self.pad_max_pool(x5)
        
        y0m = self.pad_max_pool(y0)
        y1m = self.pad_max_pool(y1)
        y2m = self.pad_max_pool(y2)
        y3m = self.pad_max_pool(y3)
        y5m = self.pad_max_pool(y5)
        
        mul_feas_y = torch.cat((y0m, y1m, y2m, y3m, y5m), dim=1)
        mul_feas_x = torch.cat((x0m, x1m, x2m, x3m, x5m), dim=1)
        y6cmc,z6cmc = self.cmc(y6,z6)
        y6_fusion = self.HBFusion_y(mul_feas_y,y6cmc)
        x6_fusion = self.HBFusion_x(mul_feas_x,x6)
        combined_features = torch.cat([x6_fusion, y6_fusion], dim=1)

        compressed_features = self.channel_compress(combined_features)  # [batch_size, 2048, H, W]
        return compressed_features,x6,y6,z6
    
    
    def classifier(self, features):
        x = self.xception.model.relu(features)

        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = x.view(x.size(0), -1)
        out = self.xception.model.last_linear(x)
        return out
    def classifier_all(self, features):
        x = self.relu_all(features)

        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = x.view(x.size(0), -1)
        out = self.base_line(x)
        return out
    def classifier_x6(self, features):
        x = self.relu_x(features)

        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = x.view(x.size(0), -1)
        out = self.base_line_x(x)
        return out
    def classifier_y6(self, features):
        x = self.relu_y(features)
        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = x.view(x.size(0), -1)
        out = self.base_line_y(x)
        return out
    def classifier_z6(self, features):
        x = self.relu_z(features)

        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = x.view(x.size(0), -1)
        out = self.base_line_z(x)
        return out
    def forward(self, input):
        compressed_features,x6,y6,z6 = self.features(input)
        
        out = self.classifier(compressed_features)
        combined_features = torch.cat([x6, y6,z6], dim=1)
        combined_features_x= torch.cat([y6,z6],dim=1)
        combined_features_y= torch.cat([x6,z6],dim=1)
        combined_features_z= torch.cat([y6,x6],dim=1)
        out_all = self.classifier_all(combined_features)
        out_x = self.classifier_x6(combined_features_x)
        out_y = self.classifier_y6(combined_features_y)
        out_z = self.classifier_z6(combined_features_z)
        out_list = []
        out_list.append(out_z)
        out_list.append(out_y)
        out_list.append(out_x)
        # return {'p_y_given_z': p_y_given_z, 'p_y_given_f_all': p_y_given_f1_f2_f3_f4,
        #         'p_y_given_f1_fn_list': p_y_given_f1_fn_list}
        return {'out': out, 'out_all': out_all,
                 'out_list': out_list}

class ChannelCompress(nn.Module):
    def __init__(self, in_ch=4096, out_ch=2048, dropout=0.5):
        """
        Reduce the amount of channels to prevent final embeddings overwhelming shallow feature maps.
        out_ch could be 512, 256, 128, or other smaller values.
        """
        super(ChannelCompress, self).__init__()

        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout2d(p=dropout)

    def forward(self, x):

        x = self.conv1(x) 
        x = self.bn1(x)   
        x = self.relu(x)   
        x = self.dropout(x) 

        return x