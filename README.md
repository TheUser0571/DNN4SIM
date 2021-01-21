# DNN4SIM
MA3 Semester Project

## Abstract
In this project an alternative to the classical 9-image SR-SIM reconstruction method was proposed using 4-image SR-SIM paired with a DNN. It allows to compensate for the typical reconstruction artefacts and provide a noise resistant, clean super-resolution image. The implemented system requires only 3 raw structured images and a widefield image, reducing acquisition time and photobleaching effects. The appropriate DNN architecture for this task was found to be RCAN. It was trained on simulated SIM images using the DIV2K dataset. A detailed evaluation, using several metrics on three simulated SIM test images with different noise levels confirmed that the proposed system could indeed provide super-resolution images from a reduced number of SIM images.

## File structure:

Direct4SIM/
	- Direct4SIM/ --> contains the necessary source code of the 4-image reconstruction
	- ExampleSimu.m --> used to generate the test image reconstruction and widefield
	- generate_reconstructions.m --> used to create the training data (reconstructions) from the DIV2K images (saved as .mat files)
	- Real_recons.m --> used to reconstruct real SIM images using the 4-image algorithm (experimental)

lib/
	- iplabs.py --> contains the IPLabsViewer class used to display images in jupyter notebooks

pytorch_ssim/ --> contains the source code for the structural similarity index measure used in the loss function and to compare the images

source/
	- nn.py --> contains the DNN structures and utility functions used in training
	- run_training.sh --> a shell script that performs the automated training on a remote server
	- train_NN.py --> contains all the necessary functions used to train and evaluate the DNNs (can also be run as a script)

DNN_test.ipynb --> jupyter notebook used to test and compare the different DNNs

Exploration.ipynb --> jupyter notebook used to explore the W2S dataset, which in the end was not used, but could maybe be useful in future work

Explore_DIV2K.ipynb --> jupyter notebook (SoS kernel) used to explore the DIV2K dataset, generate the .mat used for reconstruction and the .npy files used for training

requirements.txt --> text file containing the required modules needed to run the notebooks