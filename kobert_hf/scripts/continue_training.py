# 학습된 모델에서 이어서 학습
# 2025-11-07, 김병현 작성

import torch
from train_sentence_order import main as train_main
from src.models.sentence_order_model import SentenceOrderPredictor

"""
이 스크립트는 이미 학습된 모델(models/sentence_order_model_best.pt)을 불러와서
추가로 더 학습시킵니다.

사용법:
    python3 continue_training.py
"""

if __name__ == "__main__":
    import os

    model_path = 'models/sentence_order_model_best.pt'

    if not os.path.exists(model_path):
        print(f"⚠️  경고: {model_path} 파일이 없습니다!")
        print(f"먼저 train_sentence_order.py를 실행하여 모델을 학습시켜주세요.")
        exit(1)

    print("=" * 70)
    print("학습된 모델에서 이어서 학습")
    print("=" * 70)
    print(f"\n📁 기존 모델 로드: {model_path}")
    print(f"💡 이 모델에서 추가로 10 epochs 더 학습합니다.\n")

    # train_sentence_order.py의 main() 함수 실행
    # 모델은 자동으로 최고 성능 모델을 덮어씁니다
    train_main()

    print("\n" + "=" * 70)
    print("✅ 추가 학습 완료!")
    print("=" * 70)
