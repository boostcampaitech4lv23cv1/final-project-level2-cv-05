#!bin/bash
name='mosaic_mix_25000_training_scale'
epochs=40
weights=./pretrained/yolov7x_training.pt
python train.py --weights ${weights} --epochs ${epochs} --name ${name} --cache-images
