#!/bin/bash 

source /home/psrivastava/new_env/bin/activate
export PYTHONPATH=/home/psrivastava/pyroomacoustics_latest/pyroomacoustics_DIRPAT/:$PYTHONPATH
python /home/psrivastava/source_localization/doa_estimation/dcase_mic_arr/train_scripts/D5_1101/he_cnn_resnet.py
