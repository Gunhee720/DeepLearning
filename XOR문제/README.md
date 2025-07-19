# 🧠 XOR 문제와 Perceptron, Logistic Regression, MLP 분류 실험

본 프로젝트는 **선형(linear)** 및 **비선형(non-linear)** 데이터셋에 대해  
Perceptron, Logistic Regression, MLP 모델의 성능을 비교하고 시각화하는 과제입니다.

---

## 📁 파일 구성

| 파일명 | 설명 |
|--------|------|
| `XOR문제.ipynb` | 전체 실험 코드 (선형/비선형 데이터 생성 및 모델 학습 포함) |
| `결과보고서_XOR문제.pdf` | 실험 분석 내용, 시각화 및 결론 정리된 보고서 |

---

## 🧪 실험 구성

### 🔸 선형 분류 가능한 데이터
<img width="645" height="565" alt="image" src="https://github.com/user-attachments/assets/ccd3ac40-8bae-47ca-ac82-3987122ca0e2" />

- 두 클래스 각각 100개 데이터 생성 → 70% 학습 / 30% 테스트로 분리
- GPT를 활용하여 랜덤 데이터 생성 및 2D 시각화 진행
- Perceptron으로 학습 → 높은 정확도 달성
- Decision Boundary 시각화:
  - 직선 경계로 명확히 구분되는 scatter plot 생성

---

### 🔸 XOR: 비선형 분류 문제
<img width="632" height="465" alt="image" src="https://github.com/user-attachments/assets/30b8d1c2-d72e-4488-8291-635dfd04ac2d" />

- XOR 형태로 2D binary data 생성 (100개씩, train/test 분할)
- 시각화 결과: **단일 직선으로는 구분 불가**
- Perceptron 학습 결과: 정확도 50% 수준 → 학습 실패
- **비선형 모델 필요성 확인**

---

## 🧪 모델별 실험 요약
<img width="664" height="539" alt="image" src="https://github.com/user-attachments/assets/0101b93d-85a5-4c33-9753-03a141054209" />

| 모델 | Train Accuracy | Test Accuracy | Decision Boundary |
|------|----------------|---------------|--------------------|
| **Perceptron (XOR)** | 약 50% | 약 50% | 선형 (구분 불가) |
| **Logistic Regression** | 향상 없음 | 50%대 | 선형 |
| **MLP (2-hidden layers)** | 높음 (~100%) | 높음 (~100%) | 비선형 경계 생성 |

---

### 🧠 Logistic Regression이 "선형 분류기"인 이유?

- Sigmoid는 비선형 함수지만,  
  분류 결정 경계는 **`wᵀx + b = 0`**이라는 **선형 방정식**으로 정의됨
- Sigmoid는 출력값을 [0,1]로 압축하지만,  
  클래스가 나뉘는 기준선은 여전히 **직선**임
- 따라서 **Logistic Regression은 결정 경계 관점에서 선형 분류기**

---

## 🔍 시각화 예시

- 모든 모델별 Decision Boundary를 2D로 시각화
- Perceptron & Logistic → 직선 경계  
- MLP → 복잡한 곡선 형태의 분류 경계 (XOR 문제 해결 가능)

---

## 🛠️ 사용 기술

- Python
- PyTorch
- matplotlib, numpy
- GPT 활용 데이터 생성

---

## 📌 주요 인사이트

> XOR 문제처럼 **선형 분류기로는 해결할 수 없는 데이터 구조**에 대해,  
MLP와 같은 **다층 신경망**이 어떤 방식으로 효과적인 분류를 수행하는지  
직접 비교 및 시각화를 통해 확인할 수 있었습니다.

---

