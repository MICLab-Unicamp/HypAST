import sys
import numpy as np
import nibabel as nib
from nibabel.orientations import axcodes2ornt, ornt_transform, inv_ornt_aff

class Preprocessing():    
    
    def orientation(self, img):
        orig_ornt = nib.io_orientation(img.affine)
        targ_ornt = axcodes2ornt('LPS')
        transform = ornt_transform(orig_ornt, targ_ornt)
        affine_xfm = inv_ornt_aff(transform, img.shape)
        return img.as_reoriented(transform), orig_ornt

    def adjust_vol(self, vol):
        (L,C,W) = vol.shape
        img_final = vol[int(L/2)-56:int(L/2)+56, int(C/2)-56:int(C/2)+56, int(W/2)-45:int(W/2)+45]
        img_final = (img_final - img_final.min())/(img_final.max()-img_final.min()) 
        img_final = img_final*2 - 1
        return img_final
    
    def preproc(self, vol_path):    
        img_nib = nib.load(vol_path)
        img_oriented, orig_ornt = self.orientation(img_nib)
        header = img_oriented.header
        shape = img_oriented.shape
        affine = header.get_sform()  
        pixdim = header['pixdim']
        img_npy = np.asarray(img_oriented.dataobj)
        return self.adjust_vol(img_npy), affine, pixdim, shape, img_npy, orig_ornt
        
