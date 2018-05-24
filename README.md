## Project title
A convolutional neural network for estimating multiple face attributes from images.

## Motivation
This project was developed as assignment for master thesis. The main goal is to develop and evaluate capabilities of deep learning, 
specifically convolution neural network, for estimating multiple attributes from faces while learning on incomplete data. The main motivation
behind this project is to evaluate possibility of learning such a model from different data sources which have different anotation labels and consequently
be able to predict all attributes on new examples even though the model was not provide with any fully anotated examples. Final goal is to use this learned model for real time features estimation from video stream.

## Code style

[python-standard-style](https://www.python.org/dev/peps/pep-0008/)

## Software dependencies:

<b>Software versions:</b>
- [Python v 3.5.2] (https://www.python.org/downloads/release/python-352/)
- [Keras 2.0.8] (https://faroit.github.io/keras-docs/2.0.8/)
- [Keras backend : Tensorflow 1.2.1] (https://www.tensorflow.org/)
- [GraphViz 2.38.0] (https://www.graphviz.org/)


## Features
<b>Implemented features:</b>
 - Training, validation and testing of CNN model with:
 	- Ploting aggregate values for specific metrices (Accuracy, MSE, MAE)
 	- Visualization of results with diffusion matrices
 	- Computing average evaluation error (per metric) from batch inter-results 
 - Model specification
 	- Linear architecture model 
 	- Model with BN
 	- Branching architecture model
 - Single output model for:
 	- CelebA dataset
 	- Wiki dataset
 - Multi-output model for:
 	- CelebA dataset
 	- Wiki dataset
 	- Imdb dataset
 - Multi-output model with hiding labels from CelebA dataset
 - Multi-output model for merged datasets CelebA and Wiki
 - Real-time demo on video stream with estimation of learned attributes

## Installation
After installing all required libararies mentioned in previous sections and cloning the project you should valiadate folder structure against this description. 
Moreover in the git project there are not any databases with pictures from obvious reasons (size and compability). Therefore you should download image database of your desire, add corresponding configuration files. Before the first run of training model validate that you have required software installed by running:

    python requirements.py    

This script will validate required libraries and configuration. If everything will be checked and requirements met, you are done with instalation. 

### Folder structure

    .   						# CNN - Main directory
    ├── data_proc               # Scripts responsible for pre processing of data, loading data, etc..
    │    ├── config_files       # Configuration files e.g. attributes labels, description files, etc...
    │    └── data               # Parent directory for databases
    │         └── celebA        # Database with celebA dataset
    │         └── wiki_crop     # Database with wiki dataset
    ├── figures				    # Parent folder for all graphs and plots e.g. history plots
    │	   └── confusions 		# Folder for confusion matrices with error rates
    └── models					# Saved model from the last epoch and the best achieved validation error model



## How to use?
The scripts should be run in following order.

	$ python requirements.py 					# chcek if required libraries are installed and properly configured
	$ python main_training.py 					# train model
	$ python main_evaluation.py 				# evaluate trained model
	$ python main_plots.py 						# produce training history plots and confusion matrices
	$ python stream_cap.py 						# run live demo on trained model

In case you would like to skip some steps, see documentation of each script, which lists its prerequisites.


## Credits


README.md template thanks to [Akash Nimare](https://medium.com/@meakaakka/a-beginners-guide-to-writing-a-kickass-readme-7ac01da88ab3) 

Rainer Lienhart, haarcascade_frontalface_default.xml
