# FastAPI 백엔드 서버 - 문장 순서 예측 모델 API
# 2025-11-13, 김병현 작성

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import List
import torch
import os
import sys

# 현재 디렉토리를 sys.path에 추가 (모듈 import를 위해)
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from kobert_tokenizer import KoBERTTokenizer
from src.models.sentence_order_model import SentenceOrderPredictor

# ==================== FastAPI 앱 초기화 ====================

app = FastAPI(
    title="KoBERT 문장 순서 예측 API",
    description="한국어 문장들의 올바른 순서를 예측하는 API",
    version="1.0.0"
)

# CORS 설정 (프론트엔드 연결을 위해)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 모든 출처 허용 (프로덕션에서는 특정 도메인만 허용)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==================== 전역 변수 ====================

# 모델과 토크나이저를 전역으로 로드 (서버 시작 시 한 번만 로드)
MODEL = None
TOKENIZER = None
DEVICE = None

# ==================== 요청/응답 모델 ====================

class PredictRequest(BaseModel):
    """문장 순서 예측 요청"""
    sentences: List[str]

    class Config:
        json_schema_extra = {
            "example": {
                "sentences": [
                    "학교에 갔다",
                    "일어났다",
                    "밥을 먹었다",
                    "세수를 했다"
                ]
            }
        }

class PredictResponse(BaseModel):
    """문장 순서 예측 응답"""
    original_sentences: List[str]  # 입력받은 섞인 문장들
    predicted_order: List[int]  # 예측된 순서 (인덱스)
    sorted_sentences: List[str]  # 정렬된 문장들
    confidence_scores: List[List[float]]  # 각 문장의 위치별 확률

# ==================== 서버 시작/종료 이벤트 ====================

@app.on_event("startup")
async def startup_event():
    """서버 시작 시 모델과 토크나이저 로드"""
    global MODEL, TOKENIZER, DEVICE

    print("=" * 70)
    print("🚀 FastAPI 서버 시작 중...")
    print("=" * 70)

    # Device 설정
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"✅ Device: {DEVICE}")

    # 토크나이저 로드
    print("✅ 토크나이저 로드 중...")
    TOKENIZER = KoBERTTokenizer.from_pretrained('skt/kobert-base-v1')
    print("   토크나이저 로드 완료!")

    # 모델 로드
    print("✅ 모델 로드 중...")
    MODEL = SentenceOrderPredictor(max_sentences=12, hidden_size=768, dropout=0.1).to(DEVICE)

    # 학습된 모델 가중치 로드
    model_path = "models/sentence_order_model_best.pt"
    if os.path.exists(model_path):
        MODEL.load_state_dict(torch.load(model_path, map_location=DEVICE))
        print(f"   ✅ 학습된 모델 로드 완료: {model_path}")
    else:
        print(f"   ⚠️  학습된 모델이 없습니다: {model_path}")
        print(f"   ⚠️  랜덤 가중치로 실행됩니다 (정확도 낮음)")

    MODEL.eval()  # 평가 모드로 설정

    total_params = sum(p.numel() for p in MODEL.parameters())
    print(f"   전체 파라미터: {total_params:,}")
    print("=" * 70)
    print("✅ 서버 준비 완료!")
    print("=" * 70)

# ==================== API 엔드포인트 ====================

@app.get("/")
async def root():
    """루트 경로 - 프론트엔드 HTML 제공"""
    html_path = os.path.join(os.path.dirname(__file__), "static", "index.html")
    if os.path.exists(html_path):
        return FileResponse(html_path)
    return {"message": "KoBERT 문장 순서 예측 API", "docs": "/docs"}

@app.get("/health")
async def health_check():
    """헬스 체크"""
    return {
        "status": "healthy",
        "model_loaded": MODEL is not None,
        "device": str(DEVICE)
    }

@app.post("/predict", response_model=PredictResponse)
async def predict_sentence_order(request: PredictRequest):
    """
    문장 순서 예측 API

    Args:
        request: 섞인 문장들의 리스트

    Returns:
        예측된 순서와 정렬된 문장들
    """
    if MODEL is None or TOKENIZER is None:
        raise HTTPException(status_code=503, detail="모델이 아직 로드되지 않았습니다")

    sentences = request.sentences

    # 입력 검증
    if not sentences:
        raise HTTPException(status_code=400, detail="문장이 비어있습니다")

    if len(sentences) > 7:
        raise HTTPException(status_code=400, detail="최대 7개의 문장까지 지원합니다")

    try:
        # 모델 추론
        # 2025-11-18, 김병현 수정 - 중복 선택 방지를 위한 greedy decoding 구현
        with torch.no_grad():
            # 각 문장을 BERT로 인코딩
            num_sentences = len(sentences)
            sentence_embeddings = []

            for sent in sentences:
                inputs = TOKENIZER(
                    sent,
                    return_tensors='pt',
                    padding='max_length',
                    max_length=64,
                    truncation=True
                )
                input_ids = inputs['input_ids'].to(DEVICE)
                attention_mask = inputs['attention_mask'].to(DEVICE)

                # BERT 인코딩
                outputs = MODEL.bert(input_ids=input_ids, attention_mask=attention_mask)
                sentence_emb = outputs.pooler_output  # [1, hidden_size]
                sentence_embeddings.append(sentence_emb)

            # 모든 문장 임베딩을 쌓기
            sentence_embeddings = torch.stack(sentence_embeddings, dim=1)  # [1, num_sentences, hidden_size]

            # 순서 인코더 실행
            encoder_outputs, _ = MODEL.sequence_encoder(sentence_embeddings)

            # Pointer Decoder - Greedy Decoding (중복 방지)
            batch_size = 1
            hidden_size = MODEL.pointer_decoder.hidden_size

            # LSTM 초기 상태
            h0 = torch.zeros(1, batch_size, hidden_size).to(DEVICE)
            c0 = torch.zeros(1, batch_size, hidden_size).to(DEVICE)
            decoder_hidden = (h0, c0)

            # 초기 입력
            decoder_input = MODEL.pointer_decoder.initial_input.repeat(batch_size, 1, 1)

            # 이미 선택된 문장 마스킹
            mask = torch.zeros(batch_size, num_sentences, dtype=torch.bool).to(DEVICE)

            selected_indices_list = []
            all_probabilities = []

            # 각 스텝마다 문장 선택
            for step in range(num_sentences):
                # LSTM 실행
                decoder_output, decoder_hidden = MODEL.pointer_decoder.lstm(decoder_input, decoder_hidden)

                # Attention 점수 계산
                scores = MODEL.pointer_decoder.attention(decoder_hidden[0], encoder_outputs)

                # 마스킹 적용 (이미 선택된 문장 제외)
                masked_scores = scores.masked_fill(mask, -1e9)

                # Softmax로 확률 계산
                probabilities = torch.softmax(masked_scores, dim=1)
                all_probabilities.append(probabilities.squeeze(0).cpu().tolist())

                # 가장 높은 확률의 문장 선택
                predicted_index = torch.argmax(probabilities, dim=1)
                selected_indices_list.append(predicted_index.item())

                # 선택된 문장 마스킹
                mask.scatter_(dim=1, index=predicted_index.unsqueeze(1), value=True)

                # 다음 입력 준비
                decoder_input = torch.gather(
                    encoder_outputs,
                    1,
                    predicted_index.view(batch_size, 1, 1).repeat(1, 1, hidden_size)
                )

            # 결과 정리
            sorted_sentences = [sentences[idx] for idx in selected_indices_list]
            predicted_order = [0] * num_sentences
            for position, sentence_idx in enumerate(selected_indices_list):
                predicted_order[sentence_idx] = position

            return PredictResponse(
                original_sentences=sentences,
                predicted_order=predicted_order,
                sorted_sentences=sorted_sentences,
                confidence_scores=all_probabilities
            )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"모델 추론 중 오류 발생: {str(e)}")

# ==================== 정적 파일 제공 ====================

# static 폴더가 있으면 정적 파일 제공
static_dir = os.path.join(os.path.dirname(__file__), "static")
if os.path.exists(static_dir):
    app.mount("/static", StaticFiles(directory=static_dir), name="static")

# ==================== 서버 실행 ====================

if __name__ == "__main__":
    import uvicorn

    print("=" * 70)
    print("🚀 서버 시작")
    print("=" * 70)
    print("📍 URL: http://localhost:8000")
    print("📖 API 문서: http://localhost:8000/docs")
    print("=" * 70)

    uvicorn.run(app, host="0.0.0.0", port=8000)
