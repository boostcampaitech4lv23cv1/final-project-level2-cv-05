#!bin/bash

task=$1
name=$2
data=data/4th_run_124.yaml
device=0
weights=./runs/train/${name}/weights/best.pt
project=runs/${task}

if [ ${task} == "val" ]; then
    thres="--conf-thres 0.25 --iou-thres 0.45"
elif [ ${task} == "test" ]; then
    thres=""
else
    echo "Unknown task [${task}]"
    exit 1
fi

python val.py --weights ${weights} --data ${data} --task ${task} --device ${device} --project ${project} --name ${name} --verbose --save-txt ${thres}