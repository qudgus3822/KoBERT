# 문장 순서 예측 모델 (KoBERT Fine-tuning)
# 2025-11-07, 김병현 작성

import torch
import torch.nn as nn
from transformers import BertModel
from kobert_tokenizer import KoBERTTokenizer


import torch.nn.functional as F


class Attention(nn.Module):
    # 이 어텐션은 인코더 출력(Source)과 디코더 상태(Query)를 받아
    # 입력 시퀀스의 요소들 중 하나를 가리키는(Point) 확률을 계산합니다.
    def __init__(self, hidden_size):
        super(Attention, self).__init__()
        # 어텐션 점수 계산을 위한 레이어 (일반적으로 바나우 어텐션(Bahdanau) 또는 루옹 어텐션(Luong) 스타일)
        self.W1 = nn.Linear(hidden_size, hidden_size, bias=False)
        self.W2 = nn.Linear(hidden_size, hidden_size, bias=False)
        self.V = nn.Linear(hidden_size, 1, bias=False)

    def forward(self, decoder_hidden, encoder_outputs):
        # decoder_hidden: [1, batch_size, hidden_size]
        # encoder_outputs: [batch_size, num_sentences, hidden_size]

        # 1. 디코더 상태를 문장 개수만큼 복제하여 비교 준비
        # [batch_size, 1, hidden_size] -> [batch_size, num_sentences, hidden_size]
        hidden_expanded = (
            decoder_hidden.squeeze(0).unsqueeze(1).repeat(1, encoder_outputs.size(1), 1)
        )

        # 2. Alignment Score (어텐션 점수) 계산
        # 점수 = V * tanh(W1(Encoder Output) + W2(Decoder Hidden))
        # [batch_size, num_sentences, 1]
        score = self.V(torch.tanh(self.W1(encoder_outputs) + self.W2(hidden_expanded)))

        # 3. softmax 전의 score를 반환 (마스킹을 위해)
        # [batch_size, num_sentences]
        # 2025-11-17, 김병현 수정 - 마스킹을 위해 softmax 전 score 반환
        return score.squeeze(2)


class SentenceOrderPredictor(nn.Module):
    """
    KoBERT 기반 문장 순서 예측 모델

    입력: 여러 문장들 (섞인 순서)
    출력: 각 문장의 올바른 순서 (0부터 시작하는 인덱스)
    """

    def __init__(self, max_sentences=5, hidden_size=768, dropout=0.1):
        super(SentenceOrderPredictor, self).__init__()

        # KoBERT 모델 로드 (pretrained)
        self.bert = BertModel.from_pretrained("skt/kobert-base-v1")

        # 문장 간 관계를 학습하기 위한 추가 레이어
        self.max_sentences = max_sentences

        # 각 문장의 임베딩을 결합하여 순서 예측
        # 입력: [batch_size, num_sentences, hidden_size]
        # 출력: [batch_size, num_sentences, num_sentences] (각 문장의 위치 확률)

        # self.sentence_attention = nn.MultiheadAttention(
        #     embed_dim=hidden_size, num_heads=8, dropout=dropout, batch_first=True
        # )
        # 1. 순서 인코더: BERT 출력을 문맥 정보로 변환
        self.sequence_encoder = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            batch_first=True,
            bidirectional=True,
        )
        # 2. 포인터 네트워크 디코더
        self.pointer_decoder = PointerDecoder(hidden_size=hidden_size)

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
                input_ids=input_ids_list[i], attention_mask=attention_mask_list[i]
            )
            # [CLS] 토큰의 출력 사용 (문장 전체 표현)
            sentence_emb = outputs.pooler_output  # [batch_size, hidden_size]
            sentence_embeddings.append(sentence_emb)

        # 모든 문장 임베딩을 쌓기
        # [batch_size, num_sentences, hidden_size]
        sentence_embeddings = torch.stack(sentence_embeddings, dim=1)

        # 1. 순서 인코더 실행
        # encoder_outputs: [batch_size, num_sentences, hidden_size * 2]
        # 2025-11-17, 김병현 수정 - sentence_attention 제거하고 바로 인코더 실행
        encoder_outputs, _ = self.sequence_encoder(sentence_embeddings)

        # 2. 포인터 디코더 실행
        # logits: [batch_size, num_sentences(생성 순서), num_sentences(선택 확률)]
        logits = self.pointer_decoder(encoder_outputs, num_sentences)

        return logits

    def predict_order(self, sentences, tokenizer, device="cpu"):
        """
        실제 추론에 사용할 메서드
        2025-11-17, 김병현 수정 - 출력 형식 수정 (선택 순서 → 위치 매핑)

        Args:
            sentences: List[str] - 섞인 문장들
            tokenizer: KoBERTTokenizer
            device: 'cpu' or 'cuda'

        Returns:
            predicted_order: List[int] - 각 문장이 몇 번째 위치에 있어야 하는지
            예: [1, 0, 2] → 0번 문장은 1번째, 1번 문장은 0번째, 2번 문장은 2번째
        """
        self.eval()
        with torch.no_grad():
            # 각 문장을 토큰화
            input_ids_list = []
            attention_mask_list = []

            for sent in sentences:
                inputs = tokenizer(
                    sent,
                    return_tensors="pt",
                    padding="max_length",
                    max_length=128,
                    truncation=True,
                )
                input_ids_list.append(inputs["input_ids"].to(device))
                attention_mask_list.append(inputs["attention_mask"].to(device))

            # 모델 실행
            # logits: [batch_size, num_sentences, num_sentences]
            logits = self.forward(input_ids_list, attention_mask_list)

            # 각 디코딩 스텝에서 선택된 문장 인덱스 추출
            # [batch_size, num_sentences]
            selected_indices = torch.argmax(logits, dim=-1).squeeze(0)

            # 선택 순서를 위치로 변환
            # selected_indices[i]는 i번째 위치에 올 문장의 인덱스
            # 반대로, 각 문장이 몇 번째 위치에 있는지 계산
            num_sentences = len(sentences)
            position_map = [0] * num_sentences
            for position, sentence_idx in enumerate(selected_indices.cpu().tolist()):
                position_map[sentence_idx] = position

            return position_map


class PointerDecoder(nn.Module):
    def __init__(self, hidden_size):
        super(PointerDecoder, self).__init__()
        self.hidden_size = hidden_size * 2  # 인코더가 Bi-LSTM이므로 2배 크기
        self.lstm = nn.LSTM(
            input_size=hidden_size
            * 2,  # 인코더 출력 크기 (BERT 임베딩은 사용하지 않음)
            hidden_size=hidden_size * 2,
            batch_first=True,
        )
        self.attention = Attention(
            self.hidden_size
        )  # 위에서 정의한 Attention 클래스 사용
        # 초기 입력 토큰 역할을 할 벡터 (학습 대상)
        self.initial_input = nn.Parameter(torch.randn(1, 1, self.hidden_size))

    def forward(self, encoder_outputs, num_sentences):
        # encoder_outputs: [batch_size, num_sentences, hidden_size * 2]
        batch_size = encoder_outputs.size(0)

        # 순서열을 저장할 리스트
        logits_list = []

        # LSTM 초기 Hidden State 설정 (보통 인코더의 최종 상태를 사용하지만, 여기서는 단순화)
        h0 = torch.zeros(1, batch_size, self.hidden_size).to(encoder_outputs.device)
        c0 = torch.zeros(1, batch_size, self.hidden_size).to(encoder_outputs.device)
        decoder_hidden = (h0, c0)

        # 디코더의 첫 입력은 학습 가능한 초기 벡터 [batch_size, 1, hidden_size*2]
        decoder_input = self.initial_input.repeat(batch_size, 1, 1)

        # 이미 선택된 문장을 마스킹하기 위한 텐서 (초기에는 모두 False)
        mask = torch.zeros(batch_size, num_sentences, dtype=torch.bool).to(
            encoder_outputs.device
        )

        # num_sentences 만큼 반복하며 순서 예측
        for i in range(num_sentences):
            # 1. LSTM 실행
            # decoder_output: [batch_size, 1, hidden_size*2]
            # decoder_hidden: (h_n, c_n)
            decoder_output, decoder_hidden = self.lstm(decoder_input, decoder_hidden)

            # 2. Attention (Pointing) 실행
            # scores: [batch_size, num_sentences] (softmax 전 점수)
            # 2025-11-17, 김병현 수정 - score 반환으로 변경
            scores = self.attention(decoder_hidden[0], encoder_outputs)

            # 3. 마스킹 적용 (이미 선택된 문장의 점수를 매우 작게 만들어 선택 못하게 함)
            # 2025-11-17, 김병현 수정 - -inf 대신 매우 큰 음수 사용 (loss 안정성)
            masked_scores = scores.masked_fill(mask, -1e9)

            # 4. 다음 토큰(문장 인덱스) 예측 및 마스크 업데이트
            # predicted_index: [batch_size]
            # 2025-11-17, 김병현 수정 - gradient 계산을 위해 detach() 사용
            predicted_index = torch.argmax(masked_scores, dim=1).detach()

            # 선택된 문장을 마스킹 (다음 반복에서 재선택 방지)
            # 2025-11-17, 김병현 수정 - inplace 연산 대신 새로운 텐서 생성
            mask = mask.clone()
            mask.scatter_(dim=1, index=predicted_index.unsqueeze(1), value=True)

            # 5. 다음 LSTM 입력 준비
            # 다음 입력은 (선택된 문장의 인코더 출력)을 사용해야 하지만, 구현 편의상
            # 여기서는 인코더 출력을 가져와 사용합니다.
            # [batch_size, 1, hidden_size*2]
            decoder_input = torch.gather(
                encoder_outputs,
                1,
                predicted_index.view(batch_size, 1, 1).repeat(1, 1, self.hidden_size),
            )

            # 예측된 로짓 저장 (CrossEntropyLoss가 softmax를 자동으로 적용하므로 raw scores 저장)
            # 2025-11-17, 김병현 수정 - log_softmax 제거, masked_scores를 그대로 저장
            logits_list.append(masked_scores)

        # [num_sentences, batch_size, num_sentences] -> [batch_size, num_sentences, num_sentences]
        # (생성 순서, 배치, 입력 문장 개수)
        return torch.stack(logits_list, dim=1)


# ==================== 테스트 코드 ====================

if __name__ == "__main__":
    print("=" * 70)
    print("문장 순서 예측 모델 테스트")
    print("=" * 70)

    # 모델 초기화
    model = SentenceOrderPredictor(max_sentences=5)
    tokenizer = KoBERTTokenizer.from_pretrained("skt/kobert-base-v1")

    print(f"\n✅ 모델 초기화 완료")
    print(f"   - BERT 파라미터: {sum(p.numel() for p in model.bert.parameters()):,}")
    # 2025-11-17, 김병현 수정 - classifier 대신 pointer_decoder로 변경
    print(
        f"   - Pointer Decoder 파라미터: {sum(p.numel() for p in model.pointer_decoder.parameters()):,}"
    )
    print(f"   - 전체 파라미터: {sum(p.numel() for p in model.parameters()):,}")

    # 테스트 문장
    test_sentences = [
        "학교에 갔다",
        "일어났다",
        "밥을 먹었다",
        "세수를 했다",
        "옷을 입었다",
    ]

    print(f"\n테스트 문장 (섞인 순서):")
    for i, sent in enumerate(test_sentences):
        print(f"  {i}: {sent}")

    # 예측 (학습 전이라 랜덤)
    predicted_order = model.predict_order(test_sentences, tokenizer)
    print(f"\n예측된 순서 (학습 전): {predicted_order}")
    print("⚠️  학습 전이라 정확하지 않습니다. Fine-tuning이 필요합니다!")
