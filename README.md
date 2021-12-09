# HypAST

HypAST - Hypothalamus Automatic Segmentation Tool
On this package you will find a trained model for hypothalamus segmentation on T1 MRI images and a trainable class, in case you wish to use your own data.
This tool must be used for research only. Not suitable for clinical clinical purposes.

## GETTING STARTED

- CreateHDF5(list_data, list_labels, out_path): Creates input for Trainer. Return the .hdf5 file on defined path.

        list_data: List containing paths of .nii or .nii.gz images
        list_labels: List containing paths of labels (.nii, .nii.gz or .npy)
        out_path: Path were .hdf5 files will be saved

- Predictor(input_list, out): Return the segmentation on defined path.

        input_list: List containing paths of .nii or .nii.gz files to be segmented
        out: Path were segmentation will be saved

- Trainer(train_path, chkp_path, val_path, maxep=200, accum=16, weight=[1,4], lr=5e-3, bs=8):   Hypothalamus Segmentation Trainer. Return trained models on given output path.
	
        train_path: path to h5py train set
        chkp_path: checkpoint path
        val_path: path to h5py val set
        maxep: Maximum # of epochs in training (defaul = 200)
        accum: Batch accumulation (defaul = 16)
        weight: Cross Entropy Weight (defaul = [1,4])
        lr: Learning Rate (defaul = 5e-3)
        bs: Batch Size (defaul = 8)

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

## CONTACT

For more information or suggestions, please contact liviamarodrigues@gmail.com

