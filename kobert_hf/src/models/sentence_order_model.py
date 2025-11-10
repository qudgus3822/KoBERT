# 문장 순서 예측 모델 (KoBERT Fine-tuning)
# 2025-11-07, 김병현 작성

import torch
import torch.nn as nn
from transformers import BertModel
from kobert_tokenizer import KoBERTTokenizer


class SentenceOrderPredictor(nn.Module):
    """
    KoBERT 기반 문장 순서 예측 모델

    입력: 여러 문장들 (섞인 순서)
    출력: 각 문장의 올바른 순서 (0부터 시작하는 인덱스)
    """

    def __init__(self, max_sentences=5, hidden_size=768, dropout=0.1):
        super(SentenceOrderPredictor, self).__init__()

        # KoBERT 모델 로드 (pretrained)
        self.bert = BertModel.from_pretrained('skt/kobert-base-v1')

        # 문장 간 관계를 학습하기 위한 추가 레이어
        self.max_sentences = max_sentences

        # 각 문장의 임베딩을 결합하여 순서 예측
        # 입력: [batch_size, num_sentences, hidden_size]
        # 출력: [batch_size, num_sentences, num_sentences] (각 문장의 위치 확률)

        self.sentence_attention = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=8,
            dropout=dropout,
            batch_first=True
        )

        # 순서 예측을 위한 분류기
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, max_sentences)
        )

        # 위치 임베딩 (문장이 몇 번째에 있는지)
        self.position_embedding = nn.Embedding(max_sentences, hidden_size)

    def forward(self, input_ids_list, attention_mask_list):
        """
        가변 길이 문장 처리 지원
        2025-11-07, 김병현 수정 - 가변 길이 처리 추가

        Args:
            input_ids_list: List of [batch_size, seq_len] - 각 문장의 토큰 ID
            attention_mask_list: List of [batch_size, seq_len] - 각 문장의 마스크

        Returns:
            logits: [batch_size, num_sentences, max_sentences] - 각 문장의 위치 확률
        """
        batch_size = input_ids_list[0].size(0)
        num_sentences = len(input_ids_list)

        # 각 문장을 BERT로 인코딩
        sentence_embeddings = []
        for i in range(num_sentences):
            outputs = self.bert(
                input_ids=input_ids_list[i],
                attention_mask=attention_mask_list[i]
            )
            # [CLS] 토큰의 출력 사용 (문장 전체 표현)
            sentence_emb = outputs.pooler_output  # [batch_size, hidden_size]
            sentence_embeddings.append(sentence_emb)

        # 모든 문장 임베딩을 쌓기
        # [batch_size, num_sentences, hidden_size]
        sentence_embeddings = torch.stack(sentence_embeddings, dim=1)

        # Self-attention으로 문장 간 관계 학습
        attended_embeddings, _ = self.sentence_attention(
            sentence_embeddings,
            sentence_embeddings,
            sentence_embeddings
        )

        # 각 문장의 순서 예측
        # [batch_size, num_sentences, max_sentences]
        # 가변 길이를 위해 num_sentences 대신 max_sentences 사용
        logits = self.classifier(attended_embeddings)

        # num_sentences가 max_sentences보다 작으면 잘라냄
        if num_sentences < self.max_sentences:
            logits = logits[:, :, :num_sentences]

        return logits

    def predict_order(self, sentences, tokenizer, device='cpu'):
        """
        실제 추론에 사용할 메서드

        Args:
            sentences: List[str] - 섞인 문장들
            tokenizer: KoBERTTokenizer
            device: 'cpu' or 'cuda'

        Returns:
            predicted_order: List[int] - 예측된 순서 (0부터 시작)
        """
        self.eval()
        with torch.no_grad():
            # 각 문장을 토큰화
            input_ids_list = []
            attention_mask_list = []

            for sent in sentences:
                inputs = tokenizer(
                    sent,
                    return_tensors='pt',
                    padding='max_length',
                    max_length=128,
                    truncation=True
                )
                input_ids_list.append(inputs['input_ids'].to(device))
                attention_mask_list.append(inputs['attention_mask'].to(device))

            # 모델 실행
            logits = self.forward(input_ids_list, attention_mask_list)

            # 각 문장에 대해 가장 높은 확률의 위치 선택
            # [batch_size, num_sentences]
            predicted_positions = torch.argmax(logits, dim=-1).squeeze(0)

            return predicted_positions.cpu().tolist()


# ==================== 테스트 코드 ====================

if __name__ == "__main__":
    print("=" * 70)
    print("문장 순서 예측 모델 테스트")
    print("=" * 70)

    # 모델 초기화
    model = SentenceOrderPredictor(max_sentences=5)
    tokenizer = KoBERTTokenizer.from_pretrained('skt/kobert-base-v1')

    print(f"\n✅ 모델 초기화 완료")
    print(f"   - BERT 파라미터: {sum(p.numel() for p in model.bert.parameters()):,}")
    print(f"   - 추가 파라미터: {sum(p.numel() for p in model.classifier.parameters()):,}")
    print(f"   - 전체 파라미터: {sum(p.numel() for p in model.parameters()):,}")

    # 테스트 문장
    test_sentences = [
        "학교에 갔다",
        "일어났다",
        "밥을 먹었다",
        "세수를 했다",
        "옷을 입었다"
    ]

    print(f"\n테스트 문장 (섞인 순서):")
    for i, sent in enumerate(test_sentences):
        print(f"  {i}: {sent}")

    # 예측 (학습 전이라 랜덤)
    predicted_order = model.predict_order(test_sentences, tokenizer)
    print(f"\n예측된 순서 (학습 전): {predicted_order}")
    print("⚠️  학습 전이라 정확하지 않습니다. Fine-tuning이 필요합니다!")
