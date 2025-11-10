# 문장 순서 예측 - 추론 스크립트
# 2025-11-07, 김병현 작성

import torch
from kobert_tokenizer import KoBERTTokenizer
from src.models.sentence_order_model import SentenceOrderPredictor


def predict_sentence_order(sentences, model_path='models/sentence_order_model_best.pt'):
    """
    문장들의 올바른 순서를 예측합니다.

    Args:
        sentences: List[str] - 섞인 순서의 문장들
        model_path: 학습된 모델 경로

    Returns:
        sorted_sentences: List[str] - 올바른 순서로 정렬된 문장들
        predicted_order: List[int] - 예측된 순서 인덱스
    """
    # Device 설정
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 토크나이저 로드
    tokenizer = KoBERTTokenizer.from_pretrained('skt/kobert-base-v1')

    # 모델 로드
    model = SentenceOrderPredictor(max_sentences=5)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)

    # 순서 예측
    predicted_order = model.predict_order(sentences, tokenizer, device)

    # 순서대로 문장 정렬
    # predicted_order[i] = j 의미: i번째 문장이 j번째 위치에 있어야 함
    sorted_sentences = [None] * len(sentences)
    for i, position in enumerate(predicted_order):
        if position < len(sentences):  # 유효한 인덱스인지 확인
            sorted_sentences[position] = sentences[i]

    # None 제거 (예측이 잘못된 경우 대비)
    sorted_sentences = [s for s in sorted_sentences if s is not None]

    return sorted_sentences, predicted_order


# ==================== 예시 실행 ====================

if __name__ == "__main__":
    print("=" * 70)
    print("문장 순서 예측 - 추론")
    print("=" * 70)

    # 테스트 예시들
    test_cases = [
        {
            "name": "아침 루틴",
            "sentences": [
                "학교에 갔다",
                "일어났다",
                "밥을 먹었다",
                "세수를 했다",
                "옷을 입었다"
            ]
        },
        {
            "name": "요리 과정",
            "sentences": [
                "접시에 담았다",
                "재료를 손질했다",
                "재료를 준비했다",
                "팬에 볶았다",
                "간을 맞췄다"
            ]
        },
        {
            "name": "공부하기",
            "sentences": [
                "채점을 했다",
                "문제를 읽었다",
                "답을 적었다",
                "교과서를 폈다",
                "풀이를 생각했다"
            ]
        }
    ]

    # 모델 경로 확인
    import os
    model_path = 'models/sentence_order_model_best.pt'

    if not os.path.exists(model_path):
        print(f"\n⚠️  경고: 학습된 모델이 없습니다!")
        print(f"   먼저 'python train_sentence_order.py'를 실행하여 모델을 학습시켜주세요.")
        print(f"\n💡 학습 없이 테스트하려면 다음 명령을 실행하세요:")
        print(f"   python sentence_order_model.py")
        exit(1)

    # 각 테스트 케이스 실행
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n{'='*70}")
        print(f"테스트 {i}: {test_case['name']}")
        print(f"{'='*70}")

        sentences = test_case['sentences']

        print(f"\n📥 입력 (섞인 순서):")
        for j, sent in enumerate(sentences):
            print(f"   [{j}] {sent}")

        # 예측
        sorted_sentences, predicted_order = predict_sentence_order(sentences, model_path)

        print(f"\n📊 예측된 순서: {predicted_order}")
        print(f"   의미: 입력 문장 {list(range(len(sentences)))}이 순서 {predicted_order}에 위치")

        print(f"\n📤 출력 (올바른 순서):")
        for j, sent in enumerate(sorted_sentences):
            print(f"   [{j}] {sent}")

    print(f"\n{'='*70}")
    print("✅ 추론 완료!")
    print(f"{'='*70}")

    # 사용자 입력 모드
    print(f"\n💡 직접 문장을 입력하여 테스트할 수 있습니다.")
    print(f"   (종료하려면 'q' 입력)")

    while True:
        print(f"\n" + "-" * 70)
        user_input = input("문장들을 입력하세요 (쉼표로 구분, 예: 밥을 먹었다, 일어났다, 학교에 갔다): ")

        if user_input.strip().lower() == 'q':
            print("종료합니다.")
            break

        # 입력 파싱
        sentences = [s.strip() for s in user_input.split(',') if s.strip()]

        if len(sentences) < 2:
            print("⚠️  최소 2개 이상의 문장을 입력해주세요.")
            continue

        if len(sentences) > 5:
            print("⚠️  최대 5개까지만 지원합니다. 처음 5개만 사용합니다.")
            sentences = sentences[:5]

        print(f"\n입력된 문장 ({len(sentences)}개):")
        for j, sent in enumerate(sentences):
            print(f"   [{j}] {sent}")

        # 예측
        sorted_sentences, predicted_order = predict_sentence_order(sentences, model_path)

        print(f"\n예측된 순서: {predicted_order}")
        print(f"\n올바른 순서:")
        for j, sent in enumerate(sorted_sentences):
            print(f"   [{j}] {sent}")
