## Project title
A convolutional neural network for estimating multiple face attributes from images of faces which is learning from incomplete data.

## Motivation
This project was developed as assignment for master thesis. The main goal is to develop and evaluate capabilities of deep learning, 
specifically convolution neural network, for estimating multiple attributes from faces while learning on incomplete data. The main motivation
behind this project is to evaluate possibility of learning such a model from different data sources which have different anotation labels and consequently
be able to predict all attributes on new examples even though the model was not provide with any fully anotated examples.

## Build status
Build status of continus integration i.e. travis, appveyor etc. Ex. - 

[![Build Status](https://travis-ci.org/akashnimare/foco.svg?branch=master)](https://travis-ci.org/akashnimare/foco)
[![Windows Build Status](https://ci.appveyor.com/api/projects/status/github/akashnimare/foco?branch=master&svg=true)](https://ci.appveyor.com/project/akashnimare/foco/branch/master)

## Code style
If you're using any code style like xo, standard etc. That will help others while contributing to your project. Ex. -

[![js-standard-style](https://img.shields.io/badge/code%20style-standard-brightgreen.svg?style=flat)](https://github.com/feross/standard)
 
## Screenshots
Include logo/demo screenshot etc.

## Tech/framework used

<b>Software versions:</b>
- [Python v 3.5.2] (https://www.python.org/downloads/release/python-352/)
- [Keras 2.0.8] (https://faroit.github.io/keras-docs/2.0.8/)
- [Keras backend : Tensorflow 1.2.1] (https://www.tensorflow.org/)
- [GraphViz 2.38.0] (https://www.graphviz.org/)


## Features
What makes your project stand out?

## Code Example
Show what the library does as concisely as possible, developers should be able to figure out **how** your project solves their problem by looking at the code example. Make sure the API you are showing off is obvious, and that your code is short and concise.

## Installation
Provide step by step series of examples and explanations about how to get a development env running.

### Folder structure

    .   						# CNN - Main directory
    ├── data_proc               # Scripts responsible for pre processing of data, loading data, etc..
    │    ├── config_files       # Configuration files like attributes labels, description files, 
    │    └── data               # Parent directory for databases
    │         └── celebA        # Database with celebA dataset
    ├── figures				    # Parent folder for all graphs and plots like history plots
    │	   └── confusions 		# Folder for confusion matrices with error rates
    └── model					# Saved model from the last epoch and the best     achieved validation error model


## API Reference

Depending on the size of the project, if it is small and simple enough the reference docs can be added to the README. For medium size to larger projects it is important to at least provide a link to where the API reference docs live.

## Tests
Describe and show how to run the tests with code examples.

## How to use?
If people like your project they’ll want to learn how they can use it. To do so include step by step guide to use your project.

## Contribute

Let people know how they can contribute into your project. A [contributing guideline](https://github.com/zulip/zulip-electron/blob/master/CONTRIBUTING.md) will be a big plus.

## Credits


README.md template thanks to [Akash Nimare](https://medium.com/@meakaakka/a-beginners-guide-to-writing-a-kickass-readme-7ac01da88ab3) 

#### Anything else that seems useful

## License
A short snippet describing the license (MIT, Apache etc)

MIT © [Yourname]()