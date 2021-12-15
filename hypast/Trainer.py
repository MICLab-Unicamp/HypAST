import warnings
from .HypothalamusDataset import HypothalamusDataset
from .MyModelLightning import MyModelLightning
import os
import h5py
import torch
import numpy as np
import argparse
import pytorch_lightning as pl
import albumentations as A
from skimage.measure import regionprops
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader
if torch.cuda.is_available():
    os.environ["CUDA_VISIBLE_DEVICES"] = "0" 
    device = torch.device("cuda:0")
else:
    print('No CUDA device Found')

warnings.filterwarnings("ignore")     


class Trainer():
    '''
    Hypothalamus Segmentation Trainer. Return trained models on given output path.
    train_path: path to h5py train set
    chkp_path: checkpoint path
    val_path: path to h5py val set
    maxep: Maximum # of epochs in training (defaul = 200)
    accum: Batch accumulation (defaul = 16)
    weight: Cross Entropy Weight (defaul = [1,4])
    lr: Learning Rate (defaul = 5e-3)
    bs: Batch Size (defaul = 8)
    '''
    
    def __init__(self, train_path, chkp_path, val_path, maxep=200, accum=16, weight=[1,4], lr=5e-3, bs=8):
        
        self.train_path = train_path
        self.val_path = val_path
        self.maxep = maxep
        self.accum = accum
        self.chkp_path = chkp_path
        self.weight = weight
        self.lr = lr
        self.bs = bs
    
    def data(self):    
        data_train = h5py.File(self.train_path, 'r')
        data_val = h5py.File(self.val_path, 'r')
        Xtrain = data_train['data'][:]
        Ytrain = data_train['label'][:].astype(np.uint8)
        Xval = data_val['data'][:]
        Yval = data_val['label'][:].astype(np.uint8)
        train_dataset = HypothalamusDataset(list_img = Xtrain, list_seg = Ytrain, turn = 'train', transform = A.Compose([
                                          A.Rotate(limit = (-10,10), p = 0.6),
                                          A.RandomCrop(width=112, height=112),
                                          A.ElasticTransform(p=0.3, alpha=40, sigma=120 * 0.05, alpha_affine=30 * 0.03)
                                                                                                              ]))
        train_dataloader = DataLoader(train_dataset, batch_size = self.bs, shuffle = True)
        val_dataset = HypothalamusDataset(list_img = Xval, list_seg = Yval,  turn = 'val',transform = False)
        val_dataloader = DataLoader(val_dataset, batch_size = self.bs, shuffle = True)
        return train_dataloader, val_dataloader

    def trainer(self):
        max_epochs = self.maxep
        accumulate_grad_batches = self.accum
        checkpoint_path = self.chkp_path
        checkpoint_dir = os.path.dirname(os.path.abspath(checkpoint_path))
        print(f'Files in {checkpoint_dir}: {os.listdir(checkpoint_dir)}')
        print(f'Saving checkpoints to {checkpoint_dir}')
        checkpoint_callback = ModelCheckpoint(dirpath=checkpoint_dir,
                                              filename = "exp2-{epoch}-{val_dice:.2f}",
                                              monitor="val_dice",
                                                   mode="max",
                                                   save_top_k=3)

        early_stop_callback = pl.callbacks.EarlyStopping(monitor="val_dice", min_delta=0.001, patience=20, verbose=True, mode="max")

        resume_from_checkpoint = None
        if os.path.exists(checkpoint_path):
            print(f'Restoring checkpoint: {checkpoint_path}')
            resume_from_checkpoint = checkpoint_path

        trainer = pl.Trainer(gpus=1,
                             max_epochs=max_epochs,
                             accumulate_grad_batches=accumulate_grad_batches,
                             resume_from_checkpoint = resume_from_checkpoint,
                             callbacks=[checkpoint_callback,early_stop_callback]
                            )

        train_dataloader, val_dataloader = self.data()
        model = MyModelLightning(train_dataloader=train_dataloader,
                            val_dataloader=val_dataloader, weights = self.weight, lr = self.lr)

        trainer.fit(model)
        
