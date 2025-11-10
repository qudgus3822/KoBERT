# Fine-tuning 시 레이어별 학습 전략
2025-11-07, 김병현 작성

## 왜 깊은 레이어는 덜 학습시키나?

### BERT의 레이어 구조
```
Layer 0 (입력층)    → 일반적인 단어 표현 (범용적)
Layer 1-3          → 형태소, 구문 정보
Layer 4-8          → 문맥, 의미 정보
Layer 9-11 (출력층) → 태스크 특화 정보
```

### 핵심 개념
1. **하위 레이어**: 이미 좋은 일반적 표현을 학습했음 → 크게 바꿀 필요 없음
2. **상위 레이어**: 우리의 태스크에 맞게 조정 필요 → 많이 학습

## 🎯 3가지 전략

### 전략 1: **레이어 Freeze** (가장 간단)

하위 레이어를 완전히 고정 (학습 안 함)

```python
# BERT의 처음 6개 레이어 freeze
for param in model.bert.encoder.layer[:6].parameters():
    param.requires_grad = False
```

**장점**:
- 학습 속도 빠름 (파라미터 줄어듦)
- 과적합 방지
- 메모리 절약

**단점**:
- 표현력 제한될 수 있음


### 전략 2: **Discriminative Learning Rate** (추천!)

레이어별로 다른 learning rate 적용

```python
# 하위 레이어: 작은 lr (거의 안 바뀜)
# 상위 레이어: 큰 lr (많이 바뀜)
# 분류기: 가장 큰 lr (새로 학습)

optimizer = AdamW([
    {'params': model.bert.embeddings.parameters(), 'lr': 1e-6},
    {'params': model.bert.encoder.layer[:6].parameters(), 'lr': 1e-6},
    {'params': model.bert.encoder.layer[6:].parameters(), 'lr': 1e-5},
    {'params': model.classifier.parameters(), 'lr': 2e-5}
])
```

**장점**:
- 미세 조정 가능
- 과적합 방지하면서 표현력 유지

**단점**:
- 하이퍼파라미터 튜닝 필요


### 전략 3: **Gradual Unfreezing** (고급)

처음에는 freeze하고 점차 풀어줌

```python
# Epoch 1-2: 분류기만 학습
# Epoch 3-4: 상위 레이어 학습
# Epoch 5+: 전체 학습
```

**장점**:
- 안정적 학습
- 좋은 성능

**단점**:
- 구현 복잡
- 학습 시간 김


## 📊 데이터 크기별 추천

| 데이터 크기 | 추천 전략 |
|------------|----------|
| < 100개 | 전략 1: 하위 9개 레이어 freeze |
| 100-1000개 | 전략 2: Discriminative LR (현재 상황) |
| > 1000개 | 전략 3 또는 전체 학습 |


## 🔥 현재 코드 문제점

```python
# 현재 코드
optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)
```

**문제**:
- BERT의 모든 레이어가 동일한 learning rate (2e-5)
- 하위 레이어도 많이 변경될 수 있음
- 과적합 위험 증가


## ✅ 개선된 코드

```python
# 개선안 1: 레이어 Freeze
for param in model.bert.encoder.layer[:8].parameters():
    param.requires_grad = False

optimizer = AdamW(model.parameters(), lr=2e-5)
```

```python
# 개선안 2: Discriminative Learning Rate (추천!)
optimizer = AdamW([
    # BERT 임베딩 & 하위 레이어: 매우 작은 lr
    {'params': model.bert.embeddings.parameters(), 'lr': 1e-6},
    {'params': model.bert.encoder.layer[:6].parameters(), 'lr': 5e-6},

    # BERT 상위 레이어: 중간 lr
    {'params': model.bert.encoder.layer[6:].parameters(), 'lr': 1e-5},

    # Pooler: 중간 lr
    {'params': model.bert.pooler.parameters(), 'lr': 1e-5},

    # Attention & 분류기: 큰 lr (새로 추가된 레이어)
    {'params': model.sentence_attention.parameters(), 'lr': 2e-5},
    {'params': model.classifier.parameters(), 'lr': 2e-5}
], weight_decay=0.01)
```


## 🎓 학습 팁

1. **처음에는 전략 1로 시작** (빠르게 테스트)
2. **성능 부족하면 전략 2 적용** (더 좋은 성능)
3. **Validation accuracy 모니터링** (과적합 체크)
