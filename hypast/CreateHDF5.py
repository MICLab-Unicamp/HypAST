import os
import glob
import h5py
import argparse
import scipy.ndimage
import numpy as np
import nibabel as nib
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from nibabel.orientations import axcodes2ornt, ornt_transform, inv_ornt_aff

class CreateHDF5():

    '''
    Creates input for Trainer. Return the .hdf5 file on defined path.
    list_data: List containing paths of .nii or .nii.gz images
    list_labels: List containing paths of labels (.nii, .nii.gz or .npy)
    out_path: Path were .hdf5 files will be saved        
    '''
    
    def __init__(self, list_data, list_labels, out_path):
        self.list_data = list_data
        self.list_labels = list_labels
        self.out_path = out_path

    def orientation(self, img):
        orig_ornt = nib.io_orientation(img.affine)
        targ_ornt = axcodes2ornt('LPS')
        transform = ornt_transform(orig_ornt, targ_ornt)
        affine_xfm = inv_ornt_aff(transform, img.shape)
        return img.as_reoriented(transform)

    def find_max(self, img):
        if img.sum() < 50:
            return 0
        else:
            return np.argmax(img.sum(axis=0).sum(axis=0))


    def create_hdf5(self, xlist, ylist, depth, turn, hdf5_path): 
        lenght = len(xlist)
        idx=0
        with h5py.File(hdf5_path, 'w') as hdf5_file:
            hdf5_file.create_dataset("data", (130,130,depth*lenght),maxshape = (130,130,None) , dtype='uint16',chunks=True)
            hdf5_file.create_dataset("label", (130,130,depth*lenght),maxshape = (130,130, None),dtype='float16',chunks=True)

            for d in range(lenght):    
                img_nib = self.orientation(nib.load(xlist[d]))
                img_orig = np.asarray(img_nib.dataobj) 

                if ylist[d].split('/')[-1].split('.')[-1] == 'npy':
                    seg_orig = np.load(ylist[d])

                else:
                    seg_nib = self.orientation(nib.load(ylist[d]))
                    seg_orig = np.asarray(seg_nib.dataobj)           

                (L2,C2,W) = img_orig.shape
                if turn == 'train':
                    index = self.find_max(seg_orig)
                    if index+10>=W:
                        index = int(W/2)
                    else:
                        depth1 = index-10
                        depth2 = index+10
                    
                else:
                    index = 1
                    depth1 = int(W/2)-45
                    depth2 = int(W/2)+45

                if index != 0:
                    img_final = img_orig[int(L2/2)-65:int(L2/2)+65, int(C2/2)-65:int(C2/2)+65, depth1:depth2]
                    seg_final = seg_orig[int(L2/2)-65:int(L2/2)+65, int(C2/2)-65:int(C2/2)+65, depth1:depth2]
                    hdf5_file['data'][...,idx:idx+depth] = img_final
                    hdf5_file['label'][...,idx:idx+depth] = seg_final
                    idx+=depth

    def create_links(self):
        
        val_path = os.path.join(str(self.out_path), 'val.hd5f')
        train_path = os.path.join(str(self.out_path), 'train.hd5f')
        
        xtrain, xval, ytrain, yval = train_test_split(self.list_data, self.list_labels, test_size=0.15, random_state=4321)
        self.create_hdf5(xtrain, ytrain, 20, 'train', train_path)
        self.create_hdf5(xval, yval, 90, 'val', val_path)
