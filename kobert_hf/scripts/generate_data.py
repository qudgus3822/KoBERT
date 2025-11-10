# 문장 순서 예측을 위한 자동 학습 데이터 생성 예시
# 2025-11-07, 김병현 작성

import random
import json
from typing import List, Dict

# ==================== 방법 1: 위키피디아/뉴스 문단 활용 ====================
# 연속된 문장들은 자연스러운 시간 순서를 가지고 있음

def generate_from_paragraph(paragraph: str) -> Dict:
    """
    연속된 문단에서 학습 데이터 생성

    예시 문단:
    "아침에 일어났다. 세수를 했다. 아침을 먹었다. 학교에 갔다."
    """
    # 문장으로 분리
    sentences = [s.strip() + '.' for s in paragraph.strip().split('.') if s.strip()]

    if len(sentences) < 2:
        return None

    # 원본 순서 저장 (정답)
    original_order = list(range(len(sentences)))

    # 문장 섞기
    shuffled_sentences = sentences.copy()
    shuffled_order = list(range(len(sentences)))
    random.shuffle(shuffled_order)
    shuffled_sentences = [sentences[i] for i in shuffled_order]

    return {
        'shuffled_sentences': shuffled_sentences,  # 섞인 문장들
        'original_indices': shuffled_order,        # 각 문장의 원래 위치
        'correct_order': [shuffled_order.index(i) for i in range(len(sentences))]  # 정답 순서
    }


# ==================== 방법 2: 시간 템플릿 기반 생성 ====================
# 미리 정의된 템플릿으로 데이터 생성

TIME_TEMPLATES = {
    'morning_routine': [
        "알람이 울렸다",
        "침대에서 일어났다",
        "세수를 했다",
        "옷을 입었다",
        "아침을 먹었다"
    ],
    'cooking': [
        "재료를 준비했다",
        "재료를 손질했다",
        "팬에 기름을 두르고 볶았다",
        "간을 맞췄다",
        "접시에 담아 완성했다"
    ],
    'shopping': [
        "쇼핑 목록을 작성했다",
        "마트에 갔다",
        "장바구니에 물건을 담았다",
        "계산대에서 결제했다",
        "집으로 돌아왔다"
    ],
    'study': [
        "교과서를 폈다",
        "문제를 읽었다",
        "풀이를 생각했다",
        "답을 적었다",
        "채점을 했다"
    ]
}

def generate_from_template(template_name: str = None) -> Dict:
    """템플릿 기반 데이터 생성"""
    if template_name is None:
        template_name = random.choice(list(TIME_TEMPLATES.keys()))

    sentences = TIME_TEMPLATES[template_name].copy()

    # 일부만 샘플링 (3-5개 문장)
    if len(sentences) > 5:
        start_idx = random.randint(0, len(sentences) - 5)
        sentences = sentences[start_idx:start_idx + random.randint(3, 5)]

    # 섞기
    original_order = list(range(len(sentences)))
    shuffled_indices = original_order.copy()
    random.shuffle(shuffled_indices)

    return {
        'template': template_name,
        'shuffled_sentences': [sentences[i] for i in shuffled_indices],
        'correct_order': [shuffled_indices.index(i) for i in range(len(sentences))]
    }


# ==================== 방법 3: 웹 크롤링 (뉴스/위키) ====================

def crawl_news_or_wiki():
    """
    실제 구현 시 필요한 것:
    - requests, beautifulsoup4 설치
    - 한국어 위키피디아: https://ko.wikipedia.org
    - 뉴스 사이트 크롤링
    """
    # 예시: 위키피디아 한 문단 가져오기
    import requests
    from bs4 import BeautifulSoup

    # 실제 코드 (예시)
    # url = "https://ko.wikipedia.org/wiki/김치"
    # response = requests.get(url)
    # soup = BeautifulSoup(response.text, 'html.parser')
    # paragraphs = soup.find_all('p')

    # 각 문단에 대해 generate_from_paragraph() 호출
    pass


# ==================== 방법 4: GPT/LLM API로 생성 ====================

def generate_with_llm_api():
    """
    OpenAI API나 Claude API를 사용하여
    시간 순서가 있는 문장들을 대량 생성
    """
    prompt = """
    다음 주제에 대해 시간 순서대로 일어나는 5개의 짧은 문장을 생성해주세요:

    주제: 아침에 출근 준비하기

    출력 형식:
    1. 첫 번째 문장
    2. 두 번째 문장
    ...
    """

    # API 호출 코드 (예시)
    # import openai
    # response = openai.ChatCompletion.create(...)
    pass


# ==================== 실제 데이터셋 생성 ====================

def create_dataset(num_samples: int = 1000) -> List[Dict]:
    """
    전체 데이터셋 생성
    """
    dataset = []

    print(f"🔄 {num_samples}개의 학습 데이터 생성 중...")

    for i in range(num_samples):
        # 50% 템플릿 기반, 50% 실제 텍스트 (만약 있다면)
        if random.random() < 0.5:
            data = generate_from_template()
        else:
            # 실제로는 위키피디아나 뉴스에서 가져온 텍스트 사용
            data = generate_from_template()  # 임시로 템플릿 사용

        if data:
            dataset.append(data)

        if (i + 1) % 100 == 0:
            print(f"✅ {i + 1}/{num_samples} 완료")

    return dataset


# ==================== 예시 실행 ====================

if __name__ == "__main__":
    print("=" * 70)
    print("문장 순서 예측 학습 데이터 자동 생성 예시")
    print("=" * 70)

    # 예시 1: 문단에서 생성
    print("\n[예시 1] 문단에서 데이터 생성")
    paragraph = "아침에 일어났다. 세수를 했다. 밥을 먹었다. 학교에 갔다"
    data1 = generate_from_paragraph(paragraph)
    print(f"원본 문장: {paragraph}")
    print(f"섞인 문장: {data1['shuffled_sentences']}")
    print(f"정답 순서: {data1['correct_order']}")

    # 예시 2: 템플릿에서 생성
    print("\n" + "=" * 70)
    print("[예시 2] 템플릿 기반 데이터 생성")
    data2 = generate_from_template('morning_routine')
    print(f"템플릿: {data2['template']}")
    print(f"섞인 문장: {data2['shuffled_sentences']}")
    print(f"정답 순서: {data2['correct_order']}")

    # 예시 3: 대량 생성
    print("\n" + "=" * 70)
    print("[예시 3] 데이터셋 대량 생성 (10개만)")
    dataset = create_dataset(10)

    # JSON으로 저장
    output_file = 'data/sentence_order_dataset.json'
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(dataset, f, ensure_ascii=False, indent=2)

    print(f"\n✅ 데이터셋 저장 완료: {output_file}")
    print(f"총 {len(dataset)}개 생성")

    # 샘플 데이터 출력
    print("\n" + "=" * 70)
    print("[샘플 데이터 3개]")
    for i, sample in enumerate(dataset[:3], 1):
        print(f"\n{i}. 섞인 문장: {sample['shuffled_sentences']}")
        print(f"   정답 순서: {sample['correct_order']}")
