# Scripts 사용법
2025-11-07, 김병현 작성

## 📝 스크립트 실행 방법

모든 스크립트는 **kobert_hf 루트 디렉토리**에서 실행하세요.

### 1. 데이터 생성
```bash
cd /home/bhkim/Source/pytorch/KoBERT/kobert_hf
python3 scripts/generate_data.py
```

### 2. 모델 학습
```bash
python3 scripts/train.py
```

### 3. 추론 (예측)
```bash
python3 scripts/predict.py
```

### 4. 이어서 학습
```bash
python3 scripts/continue_training.py
```

## 📁 데이터 및 모델 위치

- **데이터**: `data/sentence_order_dataset.json`
- **학습된 모델**: `models/sentence_order_model_best.pt`
- **최종 모델**: `models/sentence_order_model_final.pt`

## ⚙️  설정 변경

학습 설정을 변경하려면 `scripts/train.py`의 하이퍼파라미터를 수정하세요:

```python
BATCH_SIZE = 2
LEARNING_RATE = 2e-5
EPOCHS = 20
MAX_LENGTH = 64
```
