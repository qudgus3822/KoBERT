# 문장 순서 예측 모델 학습 스크립트
# 2025-11-07, 김병현 작성

import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from tqdm import tqdm
import random
from kobert_tokenizer import KoBERTTokenizer
from src.models.sentence_order_model import SentenceOrderPredictor


# ==================== 데이터셋 클래스 ====================


class SentenceOrderDataset(Dataset):
    """문장 순서 예측 데이터셋"""

    def __init__(self, json_path, tokenizer, max_length=128):
        """
        Args:
            json_path: sentence_order_dataset.json 경로
            tokenizer: KoBERTTokenizer
            max_length: 최대 토큰 길이
        """
        with open(json_path, "r", encoding="utf-8") as f:
            self.data = json.load(f)

        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        """
        Returns:
            input_ids_list: List of [seq_len] - 각 문장의 토큰 ID
            attention_mask_list: List of [seq_len] - 각 문장의 마스크
            labels: [num_sentences] - 각 문장의 올바른 순서
        """
        item = self.data[idx]
        sentences = item["shuffled_sentences"]
        labels = item["correct_order"]

        # 각 문장을 토큰화
        input_ids_list = []
        attention_mask_list = []

        for sent in sentences:
            inputs = self.tokenizer(
                sent,
                padding="max_length",
                max_length=self.max_length,
                truncation=True,
                return_tensors="pt",
            )
            input_ids_list.append(inputs["input_ids"].squeeze(0))
            attention_mask_list.append(inputs["attention_mask"].squeeze(0))

        return {
            "input_ids_list": input_ids_list,
            "attention_mask_list": attention_mask_list,
            "labels": torch.tensor(labels, dtype=torch.long),
        }


def collate_fn(batch):
    """
    배치 데이터를 적절한 형태로 변환 (가변 길이 문장 지원)
    2025-11-07, 김병현 수정 - 가변 길이 문장 처리 추가
    """
    # 배치 내에서 최대 문장 개수 찾기
    max_num_sentences = max(len(item["input_ids_list"]) for item in batch)

    # 각 문장별로 배치 구성
    input_ids_batch = []
    attention_mask_batch = []
    labels_batch = []

    for i in range(max_num_sentences):
        # i번째 문장이 있는 샘플들만 수집
        input_ids_list = []
        attention_mask_list = []

        for item in batch:
            if i < len(item["input_ids_list"]):
                # 문장이 있으면 실제 데이터 추가
                input_ids_list.append(item["input_ids_list"][i])
                attention_mask_list.append(item["attention_mask_list"][i])
            else:
                # 문장이 없으면 패딩 (모두 0인 텐서)
                seq_len = item["input_ids_list"][0].shape[0]
                input_ids_list.append(torch.zeros(seq_len, dtype=torch.long))
                attention_mask_list.append(torch.zeros(seq_len, dtype=torch.long))

        input_ids_batch.append(torch.stack(input_ids_list))
        attention_mask_batch.append(torch.stack(attention_mask_list))

    # 레이블도 패딩 (-100은 loss 계산 시 무시됨)
    for item in batch:
        labels = item["labels"].tolist()
        # max_num_sentences까지 패딩
        labels += [-100] * (max_num_sentences - len(labels))
        labels_batch.append(torch.tensor(labels, dtype=torch.long))

    labels = torch.stack(labels_batch)

    return {
        "input_ids_list": input_ids_batch,
        "attention_mask_list": attention_mask_batch,
        "labels": labels,
    }


# ==================== 학습 함수 ====================


def train_epoch(
    model, dataloader, optimizer, criterion, device, gradient_accumulation_steps=1
):
    """
    한 에포크 학습
    2025-11-07, 김병현 수정 - Gradient Accumulation 추가 (메모리 절약)
    """
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    progress_bar = tqdm(dataloader, desc="Training")
    optimizer.zero_grad()

    for batch_idx, batch in enumerate(progress_bar):
        input_ids_list = [ids.to(device) for ids in batch["input_ids_list"]]
        attention_mask_list = [mask.to(device) for mask in batch["attention_mask_list"]]
        labels = batch["labels"].to(device)

        # Forward pass
        logits = model(input_ids_list, attention_mask_list)

        # Loss 계산
        # logits: [batch_size, num_sentences, num_sentences]
        # labels: [batch_size, num_sentences]
        batch_size, num_sentences, _ = logits.shape
        loss = criterion(logits.view(-1, logits.size(-1)), labels.view(-1))

        # Gradient Accumulation을 위해 loss를 나눔
        loss = loss / gradient_accumulation_steps

        # Backward pass
        loss.backward()

        # Gradient Accumulation: N번마다 업데이트
        if (batch_idx + 1) % gradient_accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()

        # 정확도 계산 (레이블 -100은 제외)
        predictions = torch.argmax(logits, dim=-1)
        mask = labels != -100
        correct += ((predictions == labels) & mask).all(dim=1).sum().item()
        total += batch_size

        total_loss += loss.item() * gradient_accumulation_steps
        progress_bar.set_postfix(
            {
                "loss": loss.item() * gradient_accumulation_steps,
                "acc": correct / total if total > 0 else 0,
            }
        )

    avg_loss = total_loss / len(dataloader)
    accuracy = correct / total if total > 0 else 0

    return avg_loss, accuracy


def evaluate(model, dataloader, criterion, device):
    """
    모델 평가
    2025-11-07, 김병현 수정 - 정확도 계산 시 -100 레이블 제외
    """
    model.eval()
    total_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            input_ids_list = [ids.to(device) for ids in batch["input_ids_list"]]
            attention_mask_list = [
                mask.to(device) for mask in batch["attention_mask_list"]
            ]
            labels = batch["labels"].to(device)

            logits = model(input_ids_list, attention_mask_list)

            batch_size = logits.size(0)
            loss = criterion(logits.view(-1, logits.size(-1)), labels.view(-1))

            # 정확도 계산 (레이블 -100은 제외)
            predictions = torch.argmax(logits, dim=-1)
            mask = labels != -100
            correct += ((predictions == labels) & mask).all(dim=1).sum().item()
            total += batch_size
            total_loss += loss.item()

    avg_loss = total_loss / len(dataloader) if len(dataloader) > 0 else 0
    accuracy = correct / total if total > 0 else 0

    return avg_loss, accuracy


# ==================== 메인 학습 루프 ====================


def main():
    print("=" * 70)
    print("문장 순서 예측 모델 학습")
    print("=" * 70)

    # 하이퍼파라미터
    # 2025-11-07, 김병현 수정 - 메모리 절약을 위한 설정 조정
    BATCH_SIZE = 2  # 8 → 2 (메모리 부족 방지)
    LEARNING_RATE = 2e-5
    EPOCHS = 20  # 10 → 20 (성능 향상을 위해 증가)
    MAX_SENTENCES = 12  # 데이터셋에 12개 문장까지 있음
    MAX_LENGTH = 64  # 128 → 64 (문장이 짧으므로 줄임)
    TRAIN_SPLIT = 0.8
    GRADIENT_ACCUMULATION_STEPS = 4  # 실질적 배치 크기 = 2 × 4 = 8

    # Device 설정
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n✅ Device: {device}")

    # 토크나이저 로드
    print("✅ 토크나이저 로드 중...")
    tokenizer = KoBERTTokenizer.from_pretrained("skt/kobert-base-v1")

    # 데이터셋 로드
    print("✅ 데이터셋 로드 중...")
    dataset = SentenceOrderDataset(
        "data/sentence_order_dataset.json", tokenizer, max_length=MAX_LENGTH
    )
    print(f"   전체 데이터: {len(dataset)}개")

    # Train/Val 분할
    train_size = int(len(dataset) * TRAIN_SPLIT)
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )
    print(f"   학습 데이터: {train_size}개")
    print(f"   검증 데이터: {val_size}개")

    # DataLoader
    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn
    )
    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn
    )

    # 모델 초기화
    print("✅ 모델 초기화 중...")
    model = SentenceOrderPredictor(
        max_sentences=MAX_SENTENCES, hidden_size=768, dropout=0.1
    ).to(device)

    # 기존 모델이 있으면 로드 (이어서 학습)
    # 2025-11-07, 김병현 수정 - 이어서 학습 기능 추가
    import os

    pretrained_model_path = "models/sentence_order_model_best.pt"
    if os.path.exists(pretrained_model_path):
        print(f"   🔄 기존 모델 발견: {pretrained_model_path}")
        print(f"   📥 기존 모델 로드 중... (이어서 학습)")
        model.load_state_dict(torch.load(pretrained_model_path, map_location=device))
        print(f"   ✅ 기존 모델 로드 완료!")
    else:
        print(f"   🆕 새로운 모델 생성")

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"   전체 파라미터: {total_params:,}")
    print(f"   학습 가능 파라미터: {trainable_params:,}")

    # Optimizer & Loss
    # 2025-11-07, 김병현 수정 - Discriminative Learning Rate 적용
    # 하위 레이어는 작은 lr, 상위 레이어와 새 레이어는 큰 lr
    optimizer = AdamW(
        [
            # BERT 임베딩 & 하위 레이어 (0-5): 매우 작은 lr
            {"params": model.bert.embeddings.parameters(), "lr": 1e-6},
            {"params": model.bert.encoder.layer[:6].parameters(), "lr": 5e-6},
            # BERT 상위 레이어 (6-11): 중간 lr
            {"params": model.bert.encoder.layer[6:].parameters(), "lr": 1e-5},
            {"params": model.bert.pooler.parameters(), "lr": 1e-5},
            # 새로 추가된 레이어: 큰 lr
            {"params": model.sentence_attention.parameters(), "lr": LEARNING_RATE},
            {"params": model.classifier.parameters(), "lr": LEARNING_RATE},
        ],
        weight_decay=0.01,
    )

    criterion = nn.CrossEntropyLoss(ignore_index=-100)

    # 학습 정보 출력
    print("\n" + "=" * 70)
    print("⚙️  학습 설정")
    print("=" * 70)
    print(f"   📊 Learning Rate 전략: Discriminative")
    print(f"      - BERT 임베딩 & 하위 레이어 (0-5): 1e-6 ~ 5e-6")
    print(f"      - BERT 상위 레이어 (6-11): 1e-5")
    print(f"      - 새 레이어 (Attention, Classifier): {LEARNING_RATE}")
    print(f"   🎯 Weight Decay: 0.01 (L2 정규화)")
    print(f"   🔄 Epochs: {EPOCHS}")
    print(
        f"   📦 Batch Size: {BATCH_SIZE} (실질적: {BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS})"
    )
    print(f"   💾 Gradient Accumulation: {GRADIENT_ACCUMULATION_STEPS} steps")
    print(f"   📏 Max Length: {MAX_LENGTH} tokens")

    # 학습 시작
    print("\n" + "=" * 70)
    print("🚀 학습 시작")
    print("=" * 70)

    best_val_acc = 0
    for epoch in range(EPOCHS):
        print(f"\n📍 Epoch {epoch + 1}/{EPOCHS}")

        # 학습
        train_loss, train_acc = train_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            device,
            gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        )
        print(f"   Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")

        # 검증
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        print(f"   Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

        # 최고 모델 저장
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), "models/sentence_order_model_best.pt")
            print(f"   ✨ 최고 모델 저장! (Val Acc: {val_acc:.4f})")

    print("\n" + "=" * 70)
    print(f"✅ 학습 완료! 최고 검증 정확도: {best_val_acc:.4f}")
    print("=" * 70)

    # 최종 모델 저장
    torch.save(model.state_dict(), "models/sentence_order_model_final.pt")
    print("✅ 최종 모델 저장: models/sentence_order_model_final.pt")


if __name__ == "__main__":
    main()
