#!bin/bash

name=mosaic_25000_training_base
weights=./runs/train/${name}/weights/best.pt
data=./data/4th_run_124.yaml
task='test'
device=0


python test.py --weights ${weights} --data ${data} --task ${task} --name ${name} --verbose --save-txt
