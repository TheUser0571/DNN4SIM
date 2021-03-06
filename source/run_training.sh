#!/bin/bash
# ---------------------------------------------------------------
# This script performs the automated training on a remote server
# ---------------------------------------------------------------

# Specify the DNN and the SNR used
SNR=MIX3
LOSS_ID=RCAN_custom_16_84

# Perform 4 training steps with 20 epochs each, switching between the two halfes of the dataset
# Call template: python3 train_NN.py features_path labels_path out_folder batch_size epochs (pretrained_path)
python3 train_NN.py ../DNN4SIM_data/features_snr${SNR}_1.npy ../DNN4SIM_data/labels_1.npy ../train_out_snr${SNR}_${LOSS_ID} 2 20

python3 train_NN.py ../DNN4SIM_data/features_snr${SNR}_2.npy ../DNN4SIM_data/labels_2.npy ../train_out_snr${SNR}_${LOSS_ID}_2 2 20 ../train_out_snr${SNR}_${LOSS_ID}/trained_model.pt

python3 train_NN.py ../DNN4SIM_data/features_snr${SNR}_1.npy ../DNN4SIM_data/labels_1.npy ../train_out_snr${SNR}_${LOSS_ID}_3 2 20 ../train_out_snr${SNR}_${LOSS_ID}_2/trained_model.pt

python3 train_NN.py ../DNN4SIM_data/features_snr${SNR}_2.npy ../DNN4SIM_data/labels_2.npy ../train_out_snr${SNR}_${LOSS_ID}_4 2 20 ../train_out_snr${SNR}_${LOSS_ID}_3/trained_model.pt