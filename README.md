# 프로젝트 개요
본 프로젝트는 딥러닝의 대표적인 모델 구조인 MLP, CNN, RNN, LSTM, Attention, Transformer 등을 학습하고, 이를 활용하여 다양한 데이터를 분석하는 것을 목표로 합니다. PyTorch와 Google Colab 환경에서 실습을 수행하며, 이론과 구현을 병행하여 딥러닝 모델에 대한 이해를 깊이 있게 다집니다

# 학습목표
딥러닝의 주요 아키텍처(Multi-Layer Perceptron, CNN, RNN, LSTM, Attention 등)를 구조적 측면에서 이해

각 모델의 수학적 원리와 시각적 구조 학습

다양한 데이터셋을 활용한 모델 구현 및 실습 (MNIST, FashionMNIST, IMDB 등)

다양한 Optimizer와 Loss function을 활용한 성능 비교

Pretrained 모델(BERT, GloVe)을 활용한 고도화된 자연어 처리 적용

# 🛠️사용기술
Python 3.x
PyTorch
TorchText, Torchvision
Scikit-learn, Matplotlib
Google Colab
Pretrained Models: GloVe, BERT (HuggingFace Transformers)

# 📁 프로젝트 구조
<pre>
├── mnlist데이학습_LR모델.py               # Logistic Regression with MNIST
├── FashionMNIST와 여러 Optimizer.py      # MLP 구현 + 다양한 Optimizer 실험
├── XOR문제.py                           # XOR 분류 문제 및 Perceptron 구현
├── 영화평감정분석_LSTM&BERT.py          # IMDB 영화리뷰 감정 분석 with LSTM, BERT
└── README.md
</pre>

# 🧪 실습 내용 요약
🔹 Logistic Regression (MNIST)
mnlist데이터학습_LR모델.py

MNIST 데이터셋을 활용한 로지스틱 회귀 모델 구현

손실 및 정확도 시각화 포함

🔹 MLP & Optimizer 실험 (FashionMNIST)
FashionMNIST와 여러 Optimizer.py

MLP 모델을 구성하여 다양한 Optimizer(SGD, Adam 등) 성능 비교

정규화(Normalization) 실험 포함

🔹 Perceptron (XOR 문제)
XOR문제.py

단층 퍼셉트론을 활용한 선형 분류기 학습

데이터 시각화 및 결정 경계 시각화 포함

🔹 감정 분석 (IMDB)
영화평감정분석_LSTM&BERT.py

LSTM 기반 모델 및 HuggingFace BERT를 사용한 감정 분석

GloVe 임베딩 적용 실험으로 성능 개선
