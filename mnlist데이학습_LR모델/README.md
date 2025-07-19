# 🧮 MNIST 데이터 분류 - Logistic Regression 기반 모델 실험

본 프로젝트는 **PyTorch 기반 Logistic Regression 모델**을 사용하여  
MNIST 숫자 이미지 데이터셋 분류기를 학습하고, **loss 및 accuracy 변화**를 시각화하며  
성능 개선을 위한 실험도 진행한 과제입니다.

---

## 🧾 과제 개요

| 항목 | 내용 |
|------|------|
| 과목 | 딥러닝 |
| 학과 | 데이터사이언스 전공 |
| 교수님 | 오민식 교수님 |
| 학번 | 60192328 |
| 이름 | 강건희 |

---

## 📁 파일 구성

| 파일명 | 설명 |
|--------|------|
| `mnlist데이학습_LR모델.ipynb` | MNIST 분류 실험 Jupyter 노트북 |
| `결과보고서_mnlist데이학습_LR모델.pdf` | 실험 결과 보고서 및 분석 내용 포함 |

---

## ✅ 실험 목표

- MNIST 숫자 이미지 (28x28) → 0~9까지 분류하는 문제 해결
- 모델: **Logistic Regression (1-layer MLP)**
- 학습은 총 10,000 epoch 진행
- 매 10 epoch마다 loss 기록, 매 1,000 epoch마다 accuracy 기록
- **loss 그래프 시각화**, 성능 변화 분석

---

## 🔧 모델 및 파라미터

- Optimizer: `SGD`  
- Learning Rate: `0.01`  
- Criterion: `CrossEntropyLoss`  
- Model 구조: `Linear(input=784, output=10)`  
- (초기에는 sigmoid 포함 → 추후 제거하여 개선)

---

## 📊 시각화 및 주요 결과

- Loss는 학습이 진행될수록 꾸준히 감소함
- Accuracy는 다음과 같은 패턴을 보임:
  - 초기 1000 epoch: 낮은 정확도 (50~60%)
  - 6000~10000 epoch: 점진적 향상 → 최고 90% 근접

<p align="center"><img src="https://your-image-link.com/loss_plot.png" alt="Loss Plot" width="500"/></p>

---

## 🔁 코드 개선 실험

### 🔸 문제점:
- CrossEntropyLoss는 **logit (정규화되지 않은 값)** 을 입력받아야 하나  
초기 모델에서 `sigmoid()` 함수를 마지막에 사용하여 정규화된 값이 들어감

### 🔸 해결:
<img width="610" height="449" alt="image" src="https://github.com/user-attachments/assets/4a64f5bf-f48b-4049-baf9-e727ec9d60d4" />

- `sigmoid()` 제거하여 `logit` 그대로 출력  
- `outputs = self.linear(x)` 로 수정

### 🔸 개선 결과:
<img width="620" height="589" alt="image" src="https://github.com/user-attachments/assets/a67a02f4-eece-4cb3-99fd-7a18e3729f90" />

- Accuracy 향상: 0.01 수준에서 정확도 개선 확인됨
- 정확한 손실 계산 → **학습 안정성과 수렴 속도 개선**

---

## 📌 추가 질문에 대한 응답

- **Q: Appendix에 sigmoid가 없는 모델이 올바른가?**  
- **A: 맞음.**  
  - CrossEntropyLoss는 softmax와 log 연산을 내부적으로 처리함
  - 따라서 모델에서 `sigmoid` 또는 `softmax` 를 별도로 쓰면 안 됨
  - → Appendix 코드가 더 적절한 구조임

---

## 📈 최종 요약

| 항목 | 내용 |
|------|------|
| 최종 Accuracy | 약 90% 도달 |
| Loss 감소 경향 | 명확함 |
| 성능 개선 포인트 | sigmoid 제거 후 정확도 향상 확인 |

---

## 🧠 한 줄 의견

> “손실 함수의 정의와 입력값의 의미(logit vs 확률)를 정확히 이해해야  
모델을 올바르게 설계할 수 있다는 걸 다시 한번 느꼈습니다.”

---

## 🛠️ 사용 기술

- Python (3.x)
- PyTorch
- matplotlib
- torchvision

---


