# HuggingFace 버전 KoBERT 사용 예시
# 2025-11-07, 김병현 수정

import torch
from kobert_tokenizer import KoBERTTokenizer
from transformers import BertModel

# 토크나이저와 모델 로드
tokenizer = KoBERTTokenizer.from_pretrained('skt/kobert-base-v1')
model = BertModel.from_pretrained('skt/kobert-base-v1')

# 텍스트 입력
text = "한국어 모델을 공유합니다."
inputs = tokenizer(text, return_tensors='pt')

# 모델 실행
outputs = model(**inputs)

# ==================== Output 해석 ====================
# outputs는 BaseModelOutputWithPoolingAndCrossAttentions 객체입니다.

# 1. last_hidden_state: 모든 토큰의 마지막 레이어 출력
#    shape: [batch_size, sequence_length, hidden_size]
#    용도: 각 토큰의 문맥 정보가 담긴 벡터
print("=" * 60)
print("1. last_hidden_state (모든 토큰의 임베딩)")
print("   Shape:", outputs.last_hidden_state.shape)  # [1, seq_len, 768]
print("   의미: 각 토큰이 문장 내에서 어떤 의미를 가지는지 표현")
print("   사용 예: 개체명 인식(NER), 품사 태깅 등 토큰 단위 태스크")

# 2. pooler_output: [CLS] 토큰의 출력을 Dense layer에 통과시킨 결과
#    shape: [batch_size, hidden_size]
#    용도: 문장 전체를 대표하는 벡터
print("\n" + "=" * 60)
print("2. pooler_output (문장 전체 임베딩)")
print("   Shape:", outputs.pooler_output.shape)  # [1, 768]
print("   의미: 문장 전체의 의미를 하나의 벡터로 압축")
print("   사용 예: 감정 분석, 문장 분류, 문장 유사도 계산")

# 실제 활용 예시
print("\n" + "=" * 60)
print("3. 실제 활용 예시")
print("=" * 60)

# 예시 1: 문장 분류 (감정 분석 등)
print("\n[예시 1] 문장 분류용")
sentence_embedding = outputs.pooler_output  # [1, 768]
print(f"문장 임베딩: {sentence_embedding.shape}")
print("→ 이 벡터를 분류 레이어(Linear)에 입력하여 긍정/부정 등 분류")

# 예시 2: 토큰별 태깅 (NER 등)
print("\n[예시 2] 토큰별 태깅용 (개체명 인식)")
token_embeddings = outputs.last_hidden_state  # [1, seq_len, 768]
print(f"토큰 임베딩: {token_embeddings.shape}")
print("→ 각 토큰 벡터를 태깅 레이어에 입력하여 B-PER, I-LOC 등 태깅")

# 예시 3: 문장 유사도 계산
print("\n[예시 3] 문장 유사도 계산")
print("1) 두 문장의 pooler_output 추출")
print("2) 코사인 유사도 계산: cosine_similarity(embed1, embed2)")
print("→ 유사한 문장일수록 1에 가까운 값")

# 토큰별 정보 확인
print("\n" + "=" * 60)
print("4. 입력 토큰 분석")
print("=" * 60)
tokens = tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
print(f"입력 문장: {text}")
print(f"토큰화 결과: {tokens}")
print(f"토큰 개수: {len(tokens)}")
print(f"각 토큰의 임베딩 shape: {outputs.last_hidden_state[0].shape}")

# 첫 번째 토큰([CLS])의 벡터 일부 출력
print("\n" + "=" * 60)
print("5. [CLS] 토큰의 임베딩 (처음 10개 차원)")
print("=" * 60)
print(outputs.last_hidden_state[0][0][:10])