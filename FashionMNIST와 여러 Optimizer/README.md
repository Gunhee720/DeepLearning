# FashionMNIST Classification with Multiple Optimizers

본 프로젝트는 PyTorch를 활용하여 **FashionMNIST** 데이터셋에 대해 다양한 모델 개선 기법(Mini-Batch, Normalization, Dropout, Optimizer 비교, BatchNorm 등)을 적용하며 분류 정확도를 향상시키는 과정을 다루고 있습니다.

---

## 📌 프로젝트 목표

- FashionMNIST 데이터셋을 활용하여 다층 퍼셉트론(MLP) 기반 분류기 구현
- 다양한 최적화 기법과 regularization 방법 적용 및 비교
- 모델의 성능 향상과 일반화 성능 개선 확인

---

## 📁 프로젝트 구성

- `FashionMNIST와 여러 Optimizer.ipynb`: 전체 실험 코드 (문제 1~7 포함)
- `FashionMNIST와 여러 Optimizer.py`: 동일 실험의 Python 스크립트 버전
- `결과보고서_FashionMNIST와 여러 Optimizer.pdf`: 분석 및 성능 요약 보고서

---

## ✅ 실험 요약

### 🔹 문제 1: 기본 학습 (No Minibatch)
- 전체 70,000 데이터를 8:1:1 비율로 분할 후, minibatch 없이 학습.
- Epoch 10 동안 전체 데이터를 한 번에 모델에 전달.
- Accuracy:
  - Train: 약 89.6%
  - Validation: 약 87%
  - Test: 약 87%

### 🔹 문제 2: Minibatch 적용
- Batch Size = 64
- GPU 병렬처리 성능 향상 → 학습 시간 단축, 성능도 증가
- Accuracy:
  - Train: 90%+
  - Validation/Test: 성능 향상 확인

### 🔹 문제 3: Normalization 적용
- 정규화를 통해 입력 픽셀 범위를 `[-1, 1]`로 조정
- Loss 감소 및 안정적인 수렴 곡선 확인
- 모델의 전반적인 성능 향상

### 🔹 문제 4: Optimizer 비교
- 비교 대상: `SGD`, `SGD+Momentum`, `SGD+Nesterov`, `AdaGrad`, `RMSProp`, `Adam`
- 동일한 조건(epoch=10, lr=0.001) 하에 loss 감소 속도 비교
- 결과:
  - **Adam, RMSProp, AdaGrad**가 빠른 수렴을 보임
  - `SGD`는 느린 수렴을 보이며 초기 학습에는 부적합

### 🔹 문제 5: Dropout 적용
- MLP 내 은닉층 후에 Dropout(0.5) 추가
- 특정 뉴런에 과도한 의존 방지 → 일반화 성능 증가
- Test Accuracy가 약 **85% 이상**으로 향상됨

### 🔹 문제 6: BatchNorm 추가
- 각 은닉층에 `BatchNorm1d` 삽입 → Internal Covariate Shift 완화
- Accuracy:
  - Test Accuracy **88.76%** 기록 (최고 성능)

### 🔹 문제 7: Hyperparameter Tuning
- Learning Rate 유지 (`0.001`)
- Epoch 수를 20으로 증가
- 과적합 없이 안정적으로 학습 진행됨
- Train Accuracy 90% 이상, Test Accuracy 역시 이에 준함

---

## 📊 성능 요약

| 실험 항목               | Test Accuracy |
|------------------------|----------------|
| 기본 학습 (No minibatch) | ~87%           |
| Minibatch 적용          | ↑ (~88%+)      |
| Normalization 적용      | ↑              |
| Dropout 적용            | ↑ (~85%)       |
| BatchNorm 적용          | **88.76%**     |
| Optimizer 비교 (Adam)   | 빠른 수렴      |
| Hyperparameter 튜닝     | 안정적 90% 도달 |

---

## 🧠 결론

- 다양한 모델 개선 기법을 차례로 적용하며 MLP 모델의 정확도와 일반화 성능을 꾸준히 향상시킬 수 있었음.
- 특히, **BatchNorm + Dropout + Adam Optimizer** 조합이 가장 우수한 성능을 보여줌.
- 실험 결과는 교과서적인 Deep Learning 이론과 부합하며, 실용적인 성능 튜닝 과정을 잘 보여줌.

---

## 🛠️ 사용 기술

- Python (3.9)
- PyTorch
- torchvision
- matplotlib

---

## 📎 참고

- 데이터셋: FashionMNIST (torchvision 제공)
- 보고서: [`결과보고서_FashionMNIST와 여러 Optimizer.pdf`](./결과보고서_FashionMNIST와%20여러%20Optimizer.pdf)

---

