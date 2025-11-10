# 📁 KoBERT 문장 순서 예측 프로젝트 구조
2025-11-07, 김병현 정리

## 디렉토리 구조

```
kobert_hf/
│
├── README.md                      # 프로젝트 소개
├── setup.py                       # 패키지 설치
├── requirements.txt               # 의존성
├── PROJECT_STRUCTURE.md           # 이 파일
├── RESTRUCTURE_PLAN.md            # 재구조화 계획
│
├── kobert_tokenizer/              # 토크나이저 패키지
│   ├── __init__.py
│   └── kobert_tokenizer.py
│
├── src/                           # 소스 코드
│   ├── __init__.py
│   ├── models/                    # 모델 정의
│   │   ├── __init__.py
│   │   └── sentence_order_model.py
│   ├── data/                      # 데이터 처리
│   │   └── __init__.py
│   └── utils/                     # 유틸리티
│       └── __init__.py
│
├── scripts/                       # 실행 스크립트
│   ├── README.md                  # 스크립트 사용법
│   ├── train.py                   # 학습
│   ├── predict.py                 # 추론
│   ├── generate_data.py           # 데이터 생성
│   └── continue_training.py       # 이어서 학습
│
├── data/                          # 데이터 파일
│   └── sentence_order_dataset.json
│
├── models/                        # 학습된 모델
│   ├── sentence_order_model_best.pt
│   └── sentence_order_model_final.pt
│
├── docs/                          # 문서
│   ├── classifier_explanation.md
│   └── layer_freezing_explanation.md
│
└── examples/                      # 예제 코드
    └── basic_usage.py
```

## 🚀 사용 방법

### 1. 데이터 생성
```bash
cd /home/bhkim/Source/pytorch/KoBERT/kobert_hf
python3 scripts/generate_data.py
```

### 2. 모델 학습
```bash
python3 scripts/train.py
```

### 3. 추론
```bash
python3 scripts/predict.py
```

## 📂 각 폴더 설명

| 폴더 | 설명 |
|------|------|
| `src/models/` | 모델 아키텍처 정의 |
| `scripts/` | 실행 스크립트 (학습, 추론 등) |
| `data/` | 데이터셋 저장 |
| `models/` | 학습된 모델 체크포인트 |
| `docs/` | 프로젝트 문서 |
| `examples/` | 사용 예제 |

## 🔧 개발 환경 설정

```bash
# 의존성 설치
pip install -r requirements.txt
pip install -e .
```

## 📝 코드 구조

### 모델
- `src/models/sentence_order_model.py`: SentenceOrderPredictor 클래스

### 스크립트
- `scripts/train.py`: 학습 루프, 데이터 로더, Optimizer 설정
- `scripts/predict.py`: 추론 및 대화형 모드
- `scripts/generate_data.py`: 템플릿 기반 데이터 생성

## 🎯 주요 특징

- **가변 길이 문장 처리**: 4-6개 문장 지원
- **Discriminative Learning Rate**: 레이어별 차등 학습률
- **Gradient Accumulation**: 메모리 효율적 학습
- **자동 체크포인트**: 최고 성능 모델 자동 저장
