## Project title
A convolutional neural network for estimating multiple face attributes from images of faces which is learning from incomplete data.

## Motivation
This project was developed as assignment for master thesis. The main goal is to develop and evaluate capabilities of deep learning, 
specifically convolution neural network, for estimating multiple attributes from faces while learning on incomplete data. The main motivation
behind this project is to evaluate possibility of learning such a model from different data sources which have different anotation labels and consequently
be able to predict all attributes on new examples even though the model was not provide with any fully anotated examples.

## Code style

[python-standard-style](https://www.python.org/dev/peps/pep-0008/)
 
## Screenshots
Include logo/demo screenshot etc.

## Software dependencies:

<b>Software versions:</b>
- [Python v 3.5.2] (https://www.python.org/downloads/release/python-352/)
- [Keras 2.0.8] (https://faroit.github.io/keras-docs/2.0.8/)
- [Keras backend : Tensorflow 1.2.1] (https://www.tensorflow.org/)
- [GraphViz 2.38.0] (https://www.graphviz.org/)


## Features
<b>Implemented features:</b>
 - Single output model for CelebA dataset
 - Single output model for Wiki dataset
 - Multi-output model for CelebA dataset
 - Multi-output model for Wiki dataset
 - Multi-output model with hiding labels from CelebA dataset
 - Multi-output model for merged datasets CelebA and Wiki
 - Real-time demo on video stream with estimation of learned attributes
 
## Code Example
Show what the library does as concisely as possible, developers should be able to figure out **how** your project solves their problem by looking at the code example. Make sure the API you are showing off is obvious, and that your code is short and concise.

## Installation
After installing all required libararies mentioned in previous sections and cloning the project you should valiadate folder structure against this description. 
Moreover in the git project there are not any databases with pictures from obvious reasons (size and compability). Therefore you should download image database of your desire, add corresponding configuration files. Before the first run of training model validate that you have required software installed by running:

    python requirements.py    

This script will validate required libraries and configuration. If everything will be checked and requirements met, you are done with instalation. 

### Folder structure

    .   						# CNN - Main directory
    ├── data_proc               # Scripts responsible for pre processing of data, loading data, etc..
    │    ├── config_files       # Configuration files like attributes labels, description files, etc...
    │    └── data               # Parent directory for databases
    │         └── celebA        # Database with celebA dataset
    ├── figures				    # Parent folder for all graphs and plots like history plots
    │	   └── confusions 		# Folder for confusion matrices with error rates
    └── model					# Saved model from the last epoch and the best achieved validation error model


## Tests
Describe and show how to run the tests with code examples.

## How to use?
If people like your project they’ll want to learn how they can use it. To do so include step by step guide to use your project.

## Contribute

Let people know how they can contribute into your project. A [contributing guideline](https://github.com/zulip/zulip-electron/blob/master/CONTRIBUTING.md) will be a big plus.

## Credits


README.md template thanks to [Akash Nimare](https://medium.com/@meakaakka/a-beginners-guide-to-writing-a-kickass-readme-7ac01da88ab3) 

Rainer Lienhart, haarcascade_frontalface_default.xml

#### Anything else that seems useful

## License
A short snippet describing the license (MIT, Apache etc)

MIT © [Yourname]()