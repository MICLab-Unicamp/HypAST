from setuptools import setup, find_packages

with open("README.md","r") as fh:
    long_description = fh.read()

setup(
	name = 'hypast',
	version = '0.0.7',
	author = 'Livia Rodrigues',
	author_email = 'l180545@dac.unicamp.br',
	description = 'Hypothalamus Automatic Segmentation Tool',
	long_description = long_description,
	long_description_content_type = "text/markdown",
	url = 'https://github.com/MICLab-Unicamp/HypAST',
        packages = find_packages(),
	classifiers = [
	    'Programming Language :: Python :: 3',
	    'Operating System :: OS Independent',
	    'License :: OSI Approved :: MIT License'
            ] ,
        
        python_requires='>3.7',

	install_requires=['numpy==1.18.1',
		          'pytorch-lightning==1.4.7',
		          'h5py==2.10.0',
		          'scikit-image==0.17.2',
		          'nibabel==3.1.1',
		          'connected-components-3d',
		          'albumentations==0.4.6',
		          'scikit-learn==0.22.2.post1',
                          'efficientnet_pytorch'
	],

)
