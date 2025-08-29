import logging

import torch
from losses import distance
import torch.nn as nn
from losses.AutomaticWeightedLoss import AutomaticWeightedLoss
import numpy as np
class loss_functions():
    def __init__(self, method='distance', mi_calculator='kl', temperature=1.5, bml_method='auto', scales=[1, 1, 1],
                 gia_loss=False,dec_loss=False,device='cuda:0'):
        self.dec_loss=dec_loss
        self.gia_loss=gia_loss

        loss_num=1
        if gia_loss:
            loss_num+=1
            logging.info("With Global Information Loss")
        if dec_loss:
            loss_num+=1
            logging.info("With Local Information Loss")
        self.balance_loss = AutomaticWeightedLoss(loss_num)  # we have 3 losses

        self.softmax = torch.nn.Softmax(dim=1)
        self.method=method
        self.bml_method =bml_method
        self.scales=scales
        print(f"Mutual Information Calculator is :{mi_calculator}")
        if mi_calculator == "kl":
            self.mi_calculator = torch.nn.KLDivLoss()
        self.temperature =temperature
    def criterion(self,out_dict,y):

        return_losses=[]

        p_y_given_z=out_dict['out']
        p_y_given_f1_f2_f3_f4=out_dict['out_all']
        p_y_given_f1_fn_list=out_dict['out_list']

        # CE loss
        loss_fn = nn.CrossEntropyLoss()
        ce_loss = loss_fn(out_dict['out'], y)

        #Global Information Alignment Loss
        if self.gia_loss:
            ce_loss+=loss_fn(p_y_given_f1_fn_list[0], y)
            global_mi_loss = self.mi_calculator(self.softmax(p_y_given_f1_fn_list[0].detach() / self.temperature).log(),
                                                self.softmax(p_y_given_z / self.temperature))


        # Decoupling Loss
        if self.dec_loss:
            local_loss = 0
            if self.method == 'mi':
                # local MI loss
                p_y_given_f1_f2_f3_f4_soft = self.softmax(p_y_given_f1_f2_f3_f4.detach() / self.temperature)

                for i in range(len(p_y_given_f1_fn_list)):
                    out_v = p_y_given_f1_fn_list[i]
                    loss_1 = self.mi_calculator(p_y_given_f1_f2_f3_f4_soft.log(),
                                                     self.softmax(out_v / self.temperature))
                    if i == 2:
                        loss_1 = torch.min(loss_1, torch.tensor(0.5, device=loss_1.device))                          
                    local_loss = local_loss + loss_1
                    ce_loss = ce_loss + loss_fn(out_v, y)
                    
                local_loss = torch.exp(-local_loss)
        return_losses.append(ce_loss)
        if self.gia_loss:
            return_losses.append(global_mi_loss)
        if self.dec_loss:
            return_losses.append(local_loss)
        return return_losses
    def balance_mult_loss(self,losses):
        if self.bml_method == 'auto':
            # Automatic Weighted Loss
            loss =self.balance_loss(losses)

        elif self.bml_method == 'hyper':
            # hyper-parameter
            loss = 0
            for i, l in enumerate(losses):
                loss = loss+l*self.scales[i]
        else:
            loss=sum(losses)
        return loss
