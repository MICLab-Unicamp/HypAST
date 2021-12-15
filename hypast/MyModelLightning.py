import os
import sys
import torch
from .Correct2Seg import Correct2Seg
import numpy as np
import torch.nn as nn
import pytorch_lightning as pl
if torch.cuda.is_available():
    os.environ["CUDA_VISIBLE_DEVICES"] = "0" 
    device = torch.device("cuda:0")
else:
    print('No CUDA device Found')


class MyModelLightning(pl.LightningModule):

    def __init__(self, train_dataloader, val_dataloader, weights, lr):
                 
        super(MyModelLightning, self).__init__()

        self._train_dataloader = train_dataloader
        self._val_dataloader = val_dataloader
        self.model = Correct2Seg()
        self.class_weights = torch.FloatTensor(weights).cuda()
        self.criteria_seg = nn.CrossEntropyLoss(self.class_weights)
        self.criteria_conf = nn.BCEWithLogitsLoss()
        self.lr = lr
        
    def dice_(self, y_pred_in, y_true, turn):
        soft = nn.Softmax2d()
        if turn == 'val':
            bs = y_pred_in.shape[0]
            dice_vol = 0
            for i in range(bs):
                y_pred = ((y_pred_in[i,1])>0.8).long()
                dice_vol += torch.sum(y_pred*y_true[i])*2.0 / ((torch.sum(y_pred) + torch.sum(y_true[i]))+0.00001)
            return dice_vol/bs
        else:
            y_pred = (soft(y_pred_in)>0.8).long()
            return torch.sum(y_pred[:,1]*y_true)*2.0 / ((torch.sum(y_pred[:,1]) + torch.sum(y_true))+0.00001)

    def forward(self,x):
        return self.model(x)

    def training_step(self, batch, batch_nb):        
        x,y = batch
        
        logit_seg, logit_conf = self(x)
        loss_seg = self.criteria_seg(logit_seg, y)
        soft = nn.Softmax()
        loss_conf = self.criteria_conf(logit_conf[:,0], y.float())
        loss = loss_seg + loss_conf
        dice = self.dice_(logit_seg,y, 'train')
        
        return loss

    def validation_step(self, batch, batch_nb):               
        x,y = batch        
        bsize = x.shape[0]
        x_dice = torch.zeros([bsize,2,90,112,112], dtype=torch.float).to(device)
        loss = 0
        for i in range(87):
            x_batch = x[:,i:i+3]
            logit_seg, logit_conf = self(x_batch)
            soft = nn.Softmax()
            x_dice[:,:,i+1] = soft(logit_seg)
            loss_seg = self.criteria_seg(x_dice, y)
            loss_conf = self.criteria_conf(x_dice[:,0], y.float())     
            loss_slice = loss_seg + loss_conf 
            loss += loss_slice
        dice = self.dice_(x_dice,y, 'val')               
        self.log("val_dice", dice)

    def configure_optimizers(self):
        return torch.optim.Adam(
            [p for p in self.parameters() if p.requires_grad],
            lr=self.lr)

    def train_dataloader(self):
        return self._train_dataloader

    def val_dataloader(self):
        return self._val_dataloader
