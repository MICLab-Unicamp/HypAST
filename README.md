# HypAST - Hypothalamus Automatic Segmentation Tool

On this package you will find a trained model for hypothalamus segmentation on T1 MRI images and a trainable class, in case you wish to use your own data.

This tool is not suitable for clinical purposes.

## INSTALLATION

        pip install hypast

**HypAST** requires **Python 3.7**.

## GETTING STARTED

**HypAST** works on .nii or .nii.gz input files for images and .nii, .nii.gz or .npy for annotations. 
Using **HypAST** you will be able to predict hypothalamus segmentation using our model or to train with your own data.

See more bellow:

#### Trainer

With **Trainer** you can train the model using your own data. 

Example:

        import hypast as hyp
        train = hyp.Trainer(train_path, chkp_path, val_path, maxep=200, accum=16, weight=[1,4], lr=5e-3, bs=8)
        train.trainer()

- Input:

    - train_path: path to h5py train set
    - chkp_path: checkpoint path
    - val_path: path to h5py val set
    - maxep: Maximum # of epochs in training (defaul = 200)
    - accum: Batch accumulation (defaul = 16)
    - weight: Cross Entropy Weight (defaul = [1,4])
    - lr: Learning Rate (defaul = 5e-3)
    - bs: Batch Size (defaul = 8)

- Output:
    - Checkpoint file on defined path


#### CreateHDF5


To facilitate training using **HypAST**, CreateHDF5 will adjust your data for you.

Example:

        import hypast as hyp
        hyp.CreateHDF5(list_data, list_labels, out_path)
        create.create_links() 

- Input:

    - list_data: List containing paths of .nii or .nii.gz images
    - list_labels: List containing paths of labels (.nii, .nii.gz or .npy)
    - out_path: Path were .hdf5 files will be saved

- Output:

    - Return train.hdf5 and val.hdf5 on defined path 

#### Predictor

With **HypAST** you can also generate hypothalamus segmentation using our trained model.

Example:
        
        import hypast as hyp
        pred = hyp.Predictor(list_data, out)
        pred.predictor()

- Input:

    - list_data: List containing paths of .nii or .nii.gz files to be segmented
    - out: Path were segmentation will be saved

- Output:

    - Segmentation files on defined path

## GRAPHICAL USER INTERFACE

If you wish, you may use our HypAST graphical user interface for prediction, available for Linux.

#### DOWNLOAD

You can download HypAST-GUI at: https://drive.google.com/file/d/1CZrApOsyickRQsPlo97s0P8EnmuDUCfT/view?usp=sharing

After downloading, unzip the file and type on the terminal (inside $PATH/HypAST/):
           
            cd dist/HypAST/
            ./HypAST       

After a few seconds, a window will pop-up and you are ready to go!

#### GETTING STARTED

HypAST-GUI has, at first, three main buttons: "Open file", "Run Code," and "Save Segmentation".

- Click the "Open File" button to choose your .nii or .nii.gz files. A label will appear showing how many files were opened.
- Click the "Run" button. This will generate the segmentations. Using a CPU (i5 8th gen), each volume will take about 6.5s to be done(the first segmentation may take a little longer). 

After running, a new menu will appear with new options:

- Volume Information: This window shows a spreadsheet with the volume (in cm3) of all hypothalamus segmented. Besides, it has the dispersion graph and boxplot, where you will analyze possible outliers.
- Texture Information: This window shows a spreadsheet with texture information of all hypothalamus segmented (***ON TRIAL***).
- Visualization Tool: This window shows all T1 images and masks generated.
- Save Files: Save your segmentation and attribute files using the "Save Files" button. At this step, you will be saving one .nii mask containing the segmentation for each T1 image opened and two .csv files, one with volume information and the other with texture information. The segmentation files will receive the same name from the original image plus a "_seg" suffix. 
- Run another analysis: This button resets the menu, making it possible to run a different analysis.


## CONTACT

For more information or suggestions, please contact liviamarodrigues@gmail.com

See more on https://github.com/MICLab-Unicamp/HypAST

