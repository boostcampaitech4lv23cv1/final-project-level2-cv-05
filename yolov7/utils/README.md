train.py 파일에서 28번째 줄

from utils.datasets import create_dataloader

을

from utils.custom_datasets import create_dataloader

으로 수정하고

data 파일 안에 있는 사용하시는 hyp.~.yaml 파일 맨 밑에

crop_paste: 0.5 # image crop paste (probability)

추가하시면 됩니다.
