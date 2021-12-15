# HypAST - Hypothalamus Automatic Segmentation Tool

On this package you will find a trained model for hypothalamus segmentation on T1 MRI images and a trainable class, in case you wish to use your own data.

This tool is not suitable for clinical purposes.

## INSTALLATION

        pip3 install hypast

**HypAST** requires **Python 3**.

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

## CONTACT

For more information or suggestions, please contact liviamarodrigues@gmail.com

See more on https://github.com/MICLab-Unicamp/HypAST

## CITATION

In case you use this tool, please, cite our segmentation method:

        @inproceedings{rodrigues2020hypothalamus,
            title={Hypothalamus fully automatic segmentation from MR images using a U-Net based architecture},
            author={Rodrigues, Livia and Rezende, Thiago and Zanesco, Ariane and Hernandez, Ana Luisa and Franca, Marcondes and Rittner, Leticia},
            booktitle={15th International Symposium on Medical Information Processing and Analysis},
            volume={11330},
            pages={113300J},
            year={2020},
            organization={International Society for Optics and Photonics}
            }

