#!/bin/bash

# python3 train_nn.py features_path labels_path out_folder batch_size epochs 
python3 train_NN.py ../DNN4SIM_data/features_2.npy ../DNN4SIM_data/labels_2.npy ../train_out_customloss_2 2 50 ../train_out_customloss/trained_model.pt