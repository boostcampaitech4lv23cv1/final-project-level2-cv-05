# final-project-level2-cv-05
final-project-level2-cv-05 created by GitHub Classroom

# 파인더스에이아이

# ⚖️프로젝트 개요

---

## 목표

⭐ 데이터내의 spurious correlation 최소화
⭐ 모델 성능 향상

## 문제 정의

<aside>
❗             모든 class를 동시에 모아 놓고 데이터 수집하는 것은 현실적으로 어려움
                                                             ⬇️
 class를 여러 세션으로 나누어 데이터 수집함으로 인해 발생하는 spurious correlation을 
                                                 딥러닝 모델이 학습

</aside>

## 👨🏻‍💻팀 소개

| 김도윤 | Crop paste 고도화, Crop paste Dataloader 구현, Fit paste 구현, Image cut mix 실험, Second classifier 실험 |
| --- | --- |
| 김형석 | Crop paste 구현, Yolov5 Crop paste 실험, B-box cut out 실험, Crop mix paste 구현 |
| 박근태 | 팀장, EDA, CV strategy, Crop paste 구현, 모델 구조 수정 |
| 양윤석 | BBox segmentation 작업 후 labeling, Segmentation 활용한 pastein Augmentation실험, MMDetection의 다양한 모델 실험 |
| 정선규 | Yolov7 base 설정, Image Background 제거, Upsampling 조정을 통해 small box Detection |

## 프로젝트 진행방안

 Data correlation 문제를 Augmentation, Modeling등의 다양한 방식으로 접근하고 해결방안 모색

# 🏪프로젝트 소개

---

### 무인매장이란?

![image](https://user-images.githubusercontent.com/62612606/217691978-da9ebd55-8fb6-491c-8e1d-ff68b83fcbed.png)

 무인 매장이란 매장내 직원 없이 자동으로 결제가 되는 매장을 의미합니다. 고객이 어떤 물건을 구매하였는지 카메라로 추적하고 자동으로 결제까지 제공하는 매장을 의미합니다

## 문제점 파악

![image](https://user-images.githubusercontent.com/62612606/217692059-02550f0b-63ff-4efb-a77f-0283dfbde58e.png)

 매장에서는 수많은 상품들이 있어 데이터을 수집하고 라벨링하는 데에 현실적인 어려움이 존재합니다. 이럴 경우에 생각해 볼 수 있는 방법 중 하나는 상품을 여러 세션으로 나누어 세션 별로 데이터를 수집하는 것입니다. 그러나 현실의 데이터는 session 별로 있지 않고 다양한 상품이 섞여 있으므로, session 단위로 수집된 데이터로 모델이 학습할 경우 Spurious correlation까지 학습하게 되는 위험이 있습니다.

▽Sprious correlation에 의해 잘못 탐색한 예시

![image](https://user-images.githubusercontent.com/62612606/217692125-fe98fd3f-2345-4c9a-bb1c-803066558f4b.png)

- 데이터 수집 과정에서 Session별로 분류한 object들을 균등하게 수집하지 못함
- 모델이 한 이미지에 같은 Session의 Object만 존재한다고 등장할 확률이 높다고 판단하는 문제 발생

## 🗓 일정

[프로젝트 일정](https://www.notion.so/d5176c670eb440439086a7adf0add847)

# 📚 프로젝트 세부사항

---

## CV strategy

---

- EDA
    
    ![image](https://user-images.githubusercontent.com/62612606/217692203-04716a13-413e-49cc-8b77-2cec6ba90552.png)
    
    - 분포 상이
- CV strategy
    - EDA 결과를 활용해, Multi stratified k-fold를 통해 session, object 개수, BBox 크기 분포가 
    유지되도록 dataset size를 1/2 로 축소
    → 기존 dataset을 가장 잘 대변하는 subset 추출
    → 학습 시간 감소로 인해 다양한 실험 가능

## Baseline Modeling

---

Inference 속도가 중요한 무인 매장의 특성 상, Real Time Detection에서 뛰어난 성능을 보여주는 Yolov7의 Yolov7-X 모델을 사용, 실험 모델의 안정성을 위해 Yolov5에서의 실험도 병행함

## ✂️Crop paste

---

![image](https://user-images.githubusercontent.com/62612606/217692267-e83b292b-313b-4442-b37d-39524d2515c5.png)

- 한 이미지의 bbox별로 다른 이미지의 다른 세션의 오브젝트를 가져와 붙여오는 방식

| Augmentation(Prob) | mAP0.5 | SameFP | Real_SameFP | DiffFP | Correlation Metric |
| --- | --- | --- | --- | --- | --- |
| No Aug | 0.260 | 1377 | 954 | 30 | 0.9695 |
| Cutout(0.5) | 0.303 | 1525 | 1040 | 24 | 0.9774 |
| Mosaic(1.0) | 0.376 | 1188 | 813 | 26 | 0.9690 |
| Mosaic(1.0), Mixup(0.5) | 0.489 | 981 | 630 | 65 | 0.9065 |
| Crop Paste(0.5) | 0.583 | 945 | 624 | 104 | 0.8571 |
| Crop Paste(1.0) | 0.615 | 883 | 588 | 138 | 0.8099 |

Crop Paste의 경우 타 Augmentation보다 뛰어난 성능을 보임

## Crop Paste 고도화

---

### Resize paste

![Animation2](https://user-images.githubusercontent.com/62612606/217692394-5e79f5cd-e532-412e-97e2-7a6ff8e11658.gif)

### Fit paste

![Animation](https://user-images.githubusercontent.com/62612606/217692359-26895e13-d729-4c09-8e9f-8118b04e6ccf.gif)

### **Bbox Probability**

![image](https://user-images.githubusercontent.com/62612606/217692467-ec5ed076-b321-454c-a06d-0800acfebd83.png)

기존의 방법은 image 단위로 확률에 따라 적용 여부를 정합니다. 만약 적용한다면 image 내 모든 bbox에 대해 crop paste를 진행합니다. 

개선된 방안은 image내 bbox 별로 확률에 따라 개별적으로 crop하는 방식입니다.

| Settings | test mAP0.5 | Correlation Metric |
| --- | --- | --- |
| Base | 0.508 | 0.805 |
| +Crop Paste | 0.557 | 0.471 |
| Resize Paste -> Fit Paste | 0.608 | 0.543 |
| Image prob -> Bbox prob | 0.732 | 0.75 |

## 🧬모델 구조 수정

---

**상황 재정의**

- 무인 매장 object detection -> medium ~ small object가 대부분
- object와 다른 object, 배경 간의 spurious correlation 존재
    - receptive field가 너무 넓으면 관련 없는 주변 정보와 spurious correlation 발생 가능

<aside>
💡 Idea : Detection시, receptive field 작은 low level feature map 추가
                                          ⬇️

1. small object detection 성능 향상
2. object 자체에 집중하여, spurious correlation 감소
</aside>

 **→** p2-Layer를 Detection에 활용하도록 YOLOv7 모델 수정

### 모델 구조 비교

**기존 Yolov7x**                                                   

**Yolov7x-p2**

![image](https://user-images.githubusercontent.com/62612606/217692577-fc09ad23-71b4-4166-9828-a2cb7ca7e0bc.png)

![image](https://user-images.githubusercontent.com/62612606/217692649-8192e414-4090-4f15-84f1-6739aaca64e4.png)

- p2 layer를 detection head에 추가하여 receptive field가 작은 feature map 추가 활용
- top-down path에서 low level에 high level의 feature를 추가하고, bottom-up path에서는 high level에 low level의 feature를 추가하는 기존 구조를 반복

### 성능 비교

- 정확한 비교를 위해 train from scratch로 비교

|  | mAP0.5 | Correlation Metric |
| --- | --- | --- |
| YOLOv7x(base) | 0.474 | 0.7755 |
| YOLOv7x-p2 | 0.488 (+ 1.4%) | 0.4628 (- 31.27%) |

[시도한 기법들](https://www.notion.so/7a774b2379ca4c4a842ba60e7fa9d435)

# 🏆 프로젝트 결과

---

### 최종 모델 결과 비교
![image](https://user-images.githubusercontent.com/62612606/217692720-d06d1d42-3ba9-45b4-ba35-0e5e319436b4.png)

![image](https://user-images.githubusercontent.com/62612606/217692763-2c67f1ec-bf5d-4ee5-adee-85295e3a7ea7.png)
