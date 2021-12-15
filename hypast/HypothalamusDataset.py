import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader

class HypothalamusDataset(Dataset):
    
    def __init__(self, list_img, list_seg, turn, transform):
        self.list_img = list_img
        self.list_seg = list_seg
        self.turn = turn
        self.transform = transform
    
    def __len__(self):
        if self.turn == 'train':
            return self.list_img.shape[-1]-1
        else:
            return int(self.list_img.shape[-1]/90)-1
    
    def __getitem__(self, idx):
                
        if self.turn == 'train':
            t = 20        
        
            if idx%t<(t-2):            
                img = self.list_img[...,idx:idx+3]
                seg = self.list_seg[...,idx+1]
            elif idx%t == (t-2):
                img = np.zeros((130,130,3))
                img[...,0] = img[...,1] = self.list_img[...,idx]
                img[...,2] = self.list_img[...,idx+1]
                seg = self.list_seg[...,idx]

            elif idx%t == (t-1):
                img = np.zeros((130,130,3))
                img[...,0] = img[...,1] = img[...,2] = self.list_img[...,idx]
                seg = self.list_seg[...,idx]

            sample = self.transform(image=img, mask = seg.astype(np.uint8))
            img_final = sample['image'].copy()
            seg_final = sample['mask'].copy()
            img_final = (img_final - img_final.min())/(img_final.max()-img_final.min()+0.00001) 
            img_final = img_final*2 - 1
            img_final = img_final.astype(np.float16)
            sample = (torch.from_numpy(img_final).permute(2,0,1).float(), torch.from_numpy(seg_final).long())
            
        
        if self.turn == 'val':
                
            img_final = self.list_img[9:-9, 9:-9,idx*90:idx*90+90]
            seg_final = self.list_seg[9:-9, 9:-9,idx*90:idx*90+90]

            img_final = (img_final - img_final.min())/(img_final.max()-img_final.min()+0.00001) 
            img_final = img_final*2 - 1
            img_final = img_final.astype(np.float16)
            sample = (torch.from_numpy(img_final).permute(2,0,1).float(), torch.from_numpy(seg_final).permute(2,0,1).long())

        return sample
            
