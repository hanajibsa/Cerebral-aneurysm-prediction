# Prediction-of-cerebral-aneurysm-and-location

## 0. Goal
익명화된 뇌혈관조영술 영상을 기반으로 **뇌동맥류 여부, 위치를 진단**하는 소프트웨어 개발한다.  

## 1. Detect Aneurysm’s existence

### a. Task type: Regression

한 가지 label에 대해서 0~1 사이의 확률을 추정하는 task이므로, image input에 대해서 Aneurysm의 존재 확률을 예측하기 위한 regression model을 구성했다.

### b. Data pre-processing

train_set에 있는 images 중에는 검은 테두리가 존재하는 것과 그렇지 않은 것이 섞여있었기 때문에 이를 인식하여 테두리가 감지될 경우, 해당 부분을 crop하는 과정을 거쳤다. 제공된 이미지는 한 index에 해당하는 image가 8장이고 8장 중 일부에서만 Aneurysm이 육안으로 확인되었다. 그러나, 특정 위치에서 찍은 뇌 혈관 조영술 사진의 feature에 다른 위치에 존재하는 Aneurysm으로 인한 영향이 있을 수 있음을 고려해야한다고 판단했다. 따라서, 한 번의 prediction training에 multi-inputs을 받는 model을 구성하고자 8가지 위치 별로 9016장의 images를 각각 1127장의 8개 폴더로 분류하여 이용했다. 또한 학습을 진행하는데 있어 train set이이 부족하다고 판단하여 데이터 증강 기법을 사용하였고, 위치 정보 및 방향을 유지시키기 위하여 flip만 적용하였다.

### c. Model architecture

pre-trained ResNet50 model을 이용해 8개의 multi-inputs에 따라, feature extraction을 수행한 뒤 output을 Flatten하여 FC layer와 sigmoid 함수를 통해 1개의 label에 대한 regression 결과를 출력하도록 구성했다.

하이퍼 파라미터 튜닝을 위해 초기에 optimizer는 Adam과 NAdam을, 손실 함수는 CrossEntropy를 사용했었지만 train accuracy와 validation accuracy이 50%대로 낮았고 SGD와 BCE 손실 함수를 사용했을 때 정확도가 70%대로 눈에 띄게 향상했다. 또 learning rate를 0.0001로 작게 설정했을 때는 정확도가 낮아, 이를 증가시키기 위한 시도 끝에 learning rate=0.0015, momentum = 0.9, weight decay = 5e-4로 설정했으며 학습을 안정화시키기 위해 epoch을 150으로 증가시켰다. 이에 따라 최종적으로 train accuracy: 99.55%, validation accuracy: 99.56%를 얻었다.

https://lh4.googleusercontent.com/4pi-eFhQzYKTahcui7BmfkQ72HKbaAcqFjsqXRhzlCDvq6y-Fswd7tIJTgaJTHPUmDJ94g1p8pE1jKR_GmrHodCUzYm9n-QQ7kp7GOF0h6NamUdLoEacf8rbf_L4xrMn8eh0K56cQ0E-m0YHRAm_hkw  


## 2. Classify Aneurysm’s locations

### a. Task type: Multi-Label classification

단순히 image가 하나의 class에 속하는 것이 아니라 여러 위치의 class에 해당하는 Aneurysm이 있을 수 있으므로, 이 문제는 Multi-Label classification model을 이용해 해결하고자 했다.

### b. Data pre-processing

train_set에는 한 환자의 index 당 8개의 촬영 위치 별 images가 있다. Aneurysm의 위치 label은 총 21개로 촬영 위치 label과 상이하기 때문에 8가지의 촬영 위치에 따른 상대성을 기반으로 위치 class를 예측할 수 있도록 해야 했다. 따라서, 왼쪽 방향에서 촬영한 4장의 사진을 왼쪽 열에, 오른쪽 방향에서 촬영한 4장의 사진을 오른쪽 열에 나열하여 2열 4행의 한 image를 만들어 학습용 데이터를 준비하였다.

### c. Model architecture

pre-trained ResNet50 model을 이용해 pre-processed image input에서 feature extraction을 수행한 뒤 output을 Flatten하여 FC layer와 sigmoid 함수를 통해 21개의 label에 대한 multi-label classification 결과를 출력하도록 구성했다. optimizer SGD와 learning rate = 0.001를 사용하여 최종 training accuracy: 90.9%, validation accuracy: 91.3%를 얻었다.

https://lh4.googleusercontent.com/9vUYdj85EcfwaAX8EGm8MEHkGLHleNjQtRZVpb3PVDcbl9FKUfFWvHv-WB1XitV2nyLtBKtgYkXCQ_Vu3J75ugmEbJ1Ys5xzciP2ETonT24PVs_E1y8KSYxTV7HztUz58ne88tbvoV7acKBc1vaY2ns

convNext_tiny pretrained model을 사용했을 때는 training accuracy와 validation accuracy가 미세하게 낮았고, convNext_tiny 이상의 확장된 model을 사용할 경우 과적합 양상을 보여, 최종 model은 resnet50을 선택하게 되었다. 이후, learning rate 조정을 통해 가장 accuracy가 빠르게 최대 값으로 도달하는 것을 선정하였다.  


## 3. C-statistics

Aneurysm existence: Regression / Aneurysm locations: Multi-Label Classification

https://lh3.googleusercontent.com/HsLEPW2EIwnTjiE8oiqLe-uAoCB-51UXiA-FS98bI1FuIFqq4RPhpwKtXSZk3KV8Pr_y1yyfy4gc7Y6auHGOENEfQ7k_sZTmbc4P0w-OvvXZ0G7VAROqTY95M3vQGF8yMTuV5jYgo96VkcVvFxBZRIg

- AUROC of the provided model 0.995197725805022
- Accuracy for locations 0.9711412515316685
