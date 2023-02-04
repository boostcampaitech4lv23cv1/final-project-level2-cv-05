#!bin/bash

weights=./pretrained/yolov5x.pt
cfg=./models/yolov5x.yaml
data=./data/4th_run_124.yaml
hyp=./data/hyps/hyp.custom.yaml
device=0
entity='entity'
epochs=40
name=exp

python train_custom.py --weights ${weights} --cfg ${cfg} --data ${data} --hyp ${hyp} --device ${device} --entity ${entity} --epochs ${epochs} --name ${name}
