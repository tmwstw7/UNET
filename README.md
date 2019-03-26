# UNET CNN

The main purpose of this project is to build convolutional neural network for semantic nuclei segmentation. This the first time I&#39;m working with UNET architecture so don&#39;t judge too strictly.

Model was built using Keras library (plot of model is attached in repository).

## Recommendations

- The best performance of CNN is shown when using GPU so better install CUDA (if using Nvidia GPUs) or other appropriate software.
- You are free to change activations, loss and optimizer.
- PyCharm is probably the most convenient IDE for projects like this.

## Requirements

All required libs and frameworks indicated in _requirements.txt._

Also you need Python 3.5 interpreter.

## Overview

- **Exploratory\_analysis.ipynb**

Jupyter notebook with exploratory data analysis, visualization and preliminary predictions. Just run it in Jupiter notebook. (If using PyCharm JN, as I did, just import jupyter). I used  Otsu threshold method to indicate nuclei and show them on plot. Also k-means was used to determine whether object is foreground or background and collect corresponding data into csv files.

- **pre\_processes.py**

Script to execute files from catalogue and pre-process images and save them as NumPy arrays.

- **unet\_architecture.py**

File contains UNET model and visualize it (only if you got GraphViz installed).

- **predict.py**

Main file that collects all the previous ones together and perform predicting. Outputs csv file with RLE of each nuclei mask. Simply copy all code above and run predict.py

|   |   |
| --- | --- |

## Afterword

As I said before, it&#39;s my first time working with UNET architecture. Partly it was hard to understand but a lot of new material learnt. That was a great experience and even greater practice. Hope you will understand my (and not only mine, some code was copied from different [kaggle](https://www.kaggle.com/) kernels from [Data Science Bowl 2018](https://www.kaggle.com/c/data-science-bowl-2018)) code.

##
