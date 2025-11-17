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

    if len(sentences) > 12:
        raise HTTPException(status_code=400, detail="최대 12개의 문장까지 지원합니다")

    try:
        # 모델 추론
        with torch.no_grad():
            # 각 문장을 토큰화
            input_ids_list = []
            attention_mask_list = []

            for sent in sentences:
                inputs = TOKENIZER(
                    sent,
                    return_tensors='pt',
                    padding='max_length',
                    max_length=64,
                    truncation=True
                )
                input_ids_list.append(inputs['input_ids'].to(DEVICE))
                attention_mask_list.append(inputs['attention_mask'].to(DEVICE))

            # 모델 실행
            logits = MODEL(input_ids_list, attention_mask_list)

            # 확률로 변환 (소프트맥스)
            probabilities = torch.softmax(logits, dim=-1).squeeze(0)  # [num_sentences, num_sentences]

            # 각 문장에 대해 가장 높은 확률의 위치 선택
            predicted_positions = torch.argmax(logits, dim=-1).squeeze(0)  # [num_sentences]

            # 리스트로 변환
            predicted_order = predicted_positions.cpu().tolist()
            confidence_scores = probabilities.cpu().tolist()

            # 예측된 순서대로 문장 정렬
            # 2025-11-13, 김병현 수정 - 중복 위치 처리 개선 (모든 문장 표시)
            # predicted_order[i] = j 는 "i번째 문장이 j번째 위치에 와야 한다"는 의미

            # 위치별로 문장들을 그룹화 (같은 위치에 여러 문장이 올 수 있음)
            position_to_sentences = {}
            for i, pos in enumerate(predicted_order):
                if pos not in position_to_sentences:
                    position_to_sentences[pos] = []
                position_to_sentences[pos].append(sentences[i])

            # 위치 순서대로 정렬하여 최종 리스트 생성
            sorted_sentences = []
            for pos in sorted(position_to_sentences.keys()):
                sorted_sentences.extend(position_to_sentences[pos])

            return PredictResponse(
                original_sentences=sentences,
                predicted_order=predicted_order,
                sorted_sentences=sorted_sentences,
                confidence_scores=confidence_scores
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
