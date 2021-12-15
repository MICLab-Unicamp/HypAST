import os
import sys
import cc3d
import time
import torch
import warnings
import numpy as np
import torch.nn as nn
import nibabel as nib
from .Preprocessing import Preprocessing
from .model import SegmentationModel
from skimage.measure import regionprops
from nibabel.orientations import axcodes2ornt, ornt_transform, inv_ornt_aff
warnings.filterwarnings("ignore")     
if torch.cuda.is_available():
    os.environ["CUDA_VISIBLE_DEVICES"] = "0" 
    device = torch.device("cuda:0")
else:
    print('No CUDA device Found, using CPU')
    device = torch.device("cpu")


class Predictor():  
    
    '''
    Prediction class. Return the segmentation on defined path.
    input_list: List containing paths of .nii or .nii.gz files to be segmented
    out: Path were segmentation will be saved
    '''
    def __init__(self, input_list, out):
        self.input_list = input_list
        self.out = out

    def prediction(self, vol_path):    
        modelseg = SegmentationModel()
        preprocessing = Preprocessing()
        new_state_dict_seg = torch.utils.model_zoo.load_url('https://github.com/MICLab-Unicamp/HypAST/releases/download/v1.0.0/weights.pth')
        modelseg.load_state_dict(new_state_dict_seg)
        modelseg.to(device)
        modelseg.eval()
        soft = nn.Softmax()
        input_volume, affine, pixdim, shape, img_oriented, orig_ornt = preprocessing.preproc(vol_path)
        final_seg = np.zeros((112,112,90))
        with torch.no_grad():
            for j in range(87):
                inputt = torch.from_numpy(input_volume[...,j:j+3]).permute(2,0,1).float()
                img = inputt.view(1,3,112,112).to(device)
                outseg = soft(modelseg(img))
                final_seg[...,j+1] = outseg[0,1].detach().cpu().numpy() 

            labels_out = cc3d.connected_components((final_seg>0.8).astype(np.uint8))
            reg = regionprops(labels_out)
            for blob in reg:
                if blob.area<50:
                    labels_out[labels_out==blob.label] = 0
            out_final = (labels_out>0).astype(int)

            if labels_out.sum() == 0:
                labels_out = cc3d.connected_components((final_seg>0.2).astype(np.uint8))
                reg = regionprops(labels_out)
                for blob in reg:
                    if blob.area<50:
                        labels_out[labels_out==blob.label] =0
                out_final = (labels_out>0).astype(int)

        return out_final, affine, pixdim, shape, img_oriented, orig_ornt

    def array2nii(self, out_final, affine, pixdim, shape):

        (L,C,W) = shape
        img_arr = np.zeros((L,C,W))
        img_arr[int(L/2)-56:int(L/2)+56, int(C/2)-56:int(C/2)+56, int(W/2)-45:int(W/2)+45] = out_final
        return nib.Nifti1Image(img_arr, affine)
           

    def save_file(self, out_path, vol_path, seg_nii, orig_ornt):
        name_file = (vol_path.split('/')[-1]).split('.')[0]+"_seg.nii"
        path_save = os.path.join(out_path, name_file)
        segm_ornt = self.reorientation(seg_nii, orig_ornt)
        nib.save(segm_ornt, path_save) 
        print('saved at ',path_save)

    def reorientation(self, img, orig_ornt):
        lps_ornt = axcodes2ornt('LPS')
        transform = ornt_transform(lps_ornt, orig_ornt)
        affine_xfm = inv_ornt_aff(transform, img.shape)
        return img.as_reoriented(transform)  
   
    def predictor (self):
        for index, vol_path in enumerate(self.input_list):
            t1 = time.time()
            print('processing....')
            print(vol_path)
            out_final, affine, pixdim, shape, img_oriented, orig_ornt = self.prediction(vol_path)
            seg_nii = self.array2nii(out_final, affine, pixdim, shape)
            self.save_file(str(self.out), vol_path, seg_nii, orig_ornt)
            print(f'Done in:{np.round(time.time()-t1,3)} seconds')
            
        print('PROCESSING DONE!')
        print('SEGMENTATIONS SAVED AT ', self.out)
        
