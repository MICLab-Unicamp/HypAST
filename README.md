# HypAST
Hypothalamus Automatic Segmentation Tool

HypAST - Hypothalamus Automatic Segmentation Tool

HypAST is a hypothalamus segmentation tool created to be used by physicians and researchers. It takes as input an T1 MRI file of extension .nii and outputs the hypothalamus mask (extensions .nii and .npy), its volume, and texture attributes. 


# GETTING STARTED

# Prerequisites

There are no necessary prerequisites to run HypAST

# Requirements

HypAST needs approximately 6.5Gb of memory RAM to operate. We suggest to use a computer or virtual machine containing at least 8Gb. In case your computer does not have this space available, you can increase your swap memory.

#Download

Please, download HypAST at the following link: https://drive.google.com/drive/folders/1JJSgTeX-NlmWs0zmp9lOeiODZnOrD7F5?usp=sharing

# Installing

HypAST works only on Linux operational system. To run it, you just need to download the .zip file "HypASTExe.zip" and run the following commands on your terminal, on the directory the zip file is located:

unzip HypASTExe.zip
cd HypASTExe/dist
./hypast

You only need to unzip the folder once. After the first time, you only need to access the "dist" folder inside "HypASTExe" folder and run:

./hypast

After approximately 30s a window will pop up and you are ready to go


# USING THE TOOL

HypAST has, at first, three main buttons: "Open file", "Run Code" and "Save Segmentation".

- Click "Open File" button to choose your .nii or .nii.gz files. A label will appear showing how many files were opened.
- Click "Run Code". This will generate the segmentations. Using a CPU (i5 8th gen) each volume will take about 6.5s to be done(the first segmentation may take a little longer). A "Done!!" label will appear when segmentation is complete
- Finally, save your segmentation and attribute files using the "Save Segmentation" button. At this step, you will be saving one .nii mask with the segmentation for each T1 image opened and two .csv files, one with volume information and other with texture information.

All files will receive the same root name. For example, if you open 3 T1 images, run the code and save as "test", you will have 5 final files:
- test1.nii, test2.nii, test3.nii, test_TextureInfo.csv, test_volumeInfo.csv

After saving, three new buttons will appear:

- Volume Information:  This window shows a spreadsheet with volume information of all hypothalamus segmented. Besides, it has the dispersion graph and bloxpot, where you will be able to analyze possible outliers.
- Texture Information: This window shows a spreadsheet with texture information of all hypothalamus segmented.
- Visualization Tool: This window shows all T1 images and masks generated.

# CITATION

In case you use this tool for research purposes, please, cite our segmentation method:

Rodrigues,L., Rezende,T., Zanesco,A., Hernandez,A.L., França,M., Rittner, L. Hypothalamus fully automatic segmentation from MR images using a U-Net based architecture. SIPAIM, 2019, Colombia.

# CONTACT

For more information or suggestions, please contact liviamarodrigues@gmail.com

