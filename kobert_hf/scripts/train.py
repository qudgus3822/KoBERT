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
from src.utils import is_running_in_colab


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

        2025-11-10, 김병현 수정 - Subset 인덱싱 오류 수정
        """
        # idx를 정수로 변환 (Subset에서 넘어올 수 있음)
        idx = int(idx) if not isinstance(idx, int) else idx

        # 인덱스 범위 체크
        if idx < 0 or idx >= len(self.data):
            raise IndexError(
                f"Index {idx} out of range for dataset of size {len(self.data)}"
            )

        item = self.data[idx]

        # item이 딕셔너리인지 확인
        if not isinstance(item, dict):
            raise TypeError(f"Expected dict at index {idx}, got {type(item)}")

        sentences = item["shuffled_sentences"]
        labels = item["correct_order"]

        # 레이블 변환: 위치 매핑 → 선택 순서
        # 2025-11-17, 김병현 수정 - 포인터 네트워크 형식에 맞게 레이블 변환
        # 데이터셋: [6, 7, 1, 2, 5, 0, 4, 3] (0번 문장이 6번째 위치)
        # 필요한 형식: [5, 2, 3, 7, 6, 4, 0, 1] (0번째 위치에 5번 문장)
        num_sentences = len(sentences)
        pointer_labels = [0] * num_sentences
        for sentence_idx, position in enumerate(labels):
            pointer_labels[position] = sentence_idx

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
            "labels": torch.tensor(pointer_labels, dtype=torch.long),
        }


# 모델 저장 함수
def save_model(model):
    if is_running_in_colab():
        torch.save(
            model.state_dict(), "/content/drive/MyDrive/models/sentence_order_model.pt"
        )
    else:
        torch.save(model.state_dict(), "models/sentence_order_model.pt")


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
    2025-11-17, 김병현 수정 - 부분 정확도 메트릭 추가
    """
    model.train()
    total_loss = 0
    correct = 0  # 완전 정확도 (모든 스텝 일치)
    correct_steps = 0  # 스텝별 정확도 (부분 정답)
    total = 0
    total_steps = 0

    progress_bar = tqdm(dataloader, desc="Training")
    optimizer.zero_grad()

    for batch_idx, batch in enumerate(progress_bar):
        input_ids_list = [ids.to(device) for ids in batch["input_ids_list"]]
        attention_mask_list = [mask.to(device) for mask in batch["attention_mask_list"]]
        labels = batch["labels"].to(device)

        # Forward pass
        logits = model(input_ids_list, attention_mask_list)

        # Loss 계산
        # 2025-11-17, 김병현 수정 - 포인터 네트워크 loss 계산 방식 변경
        # logits: [batch_size, num_steps, num_sentences] (각 스텝에서 선택할 문장의 확률)
        # labels: [batch_size, num_steps] (각 스텝에서 선택해야 할 문장 인덱스)
        batch_size, num_steps, num_choices = logits.shape

        # 디버깅: logits과 labels 확인
        # 2025-11-17, 김병현 수정 - loss 무한대 문제 디버깅
        if batch_idx == 0:
            print(f"\n[DEBUG] logits shape: {logits.shape}")
            print(f"[DEBUG] labels shape: {labels.shape}")
            print(
                f"[DEBUG] logits min/max: {logits.min().item():.2f} / {logits.max().item():.2f}"
            )
            print(f"[DEBUG] labels sample: {labels[0]}")
            print(f"[DEBUG] logits has inf: {torch.isinf(logits).any()}")
            print(f"[DEBUG] logits has nan: {torch.isnan(logits).any()}")

        # 각 스텝의 loss를 계산
        loss = criterion(
            logits.view(batch_size * num_steps, num_choices), labels.view(-1)
        )

        # 디버깅: loss 값 확인
        if batch_idx == 0:
            print(f"[DEBUG] loss value: {loss.item()}")

        # Gradient Accumulation을 위해 loss를 나눔
        loss = loss / gradient_accumulation_steps

        # Backward pass
        loss.backward()

        # Gradient Accumulation: N번마다 업데이트
        if (batch_idx + 1) % gradient_accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()

        # 정확도 계산 (레이블 -100은 제외)
        # 2025-11-17, 김병현 수정 - 완전 정확도 + 스텝별 정확도 계산
        predictions = torch.argmax(logits, dim=-1)  # [batch_size, num_steps]
        mask = labels != -100

        # 완전 정확도: 모든 스텝에서 정확히 맞춰야 정답
        correct += ((predictions == labels) & mask).all(dim=1).sum().item()
        total += batch_size

        # 스텝별 정확도: 각 스텝마다 맞춘 개수
        correct_steps += ((predictions == labels) & mask).sum().item()
        total_steps += mask.sum().item()

        total_loss += loss.item() * gradient_accumulation_steps
        progress_bar.set_postfix(
            {
                "loss": loss.item() * gradient_accumulation_steps,
                "acc": correct / total if total > 0 else 0,
                "step_acc": correct_steps / total_steps if total_steps > 0 else 0,
            }
        )

    avg_loss = total_loss / len(dataloader)
    accuracy = correct / total if total > 0 else 0
    step_accuracy = correct_steps / total_steps if total_steps > 0 else 0

    return avg_loss, accuracy, step_accuracy


def evaluate(model, dataloader, criterion, device):
    """
    모델 평가
    2025-11-17, 김병현 수정 - 부분 정확도 메트릭 추가
    """
    model.eval()
    total_loss = 0
    correct = 0
    correct_steps = 0
    total = 0
    total_steps = 0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            input_ids_list = [ids.to(device) for ids in batch["input_ids_list"]]
            attention_mask_list = [
                mask.to(device) for mask in batch["attention_mask_list"]
            ]
            labels = batch["labels"].to(device)

            logits = model(input_ids_list, attention_mask_list)

            # Loss 계산
            # 2025-11-17, 김병현 수정 - train과 동일한 방식으로 계산
            batch_size, num_steps, num_choices = logits.shape
            loss = criterion(
                logits.view(batch_size * num_steps, num_choices), labels.view(-1)
            )

            # 정확도 계산 (레이블 -100은 제외)
            # 2025-11-17, 김병현 수정 - 완전 정확도 + 스텝별 정확도 계산
            predictions = torch.argmax(logits, dim=-1)
            mask = labels != -100

            # 완전 정확도
            correct += ((predictions == labels) & mask).all(dim=1).sum().item()
            total += batch_size

            # 스텝별 정확도
            correct_steps += ((predictions == labels) & mask).sum().item()
            total_steps += mask.sum().item()

            total_loss += loss.item()

    avg_loss = total_loss / len(dataloader) if len(dataloader) > 0 else 0
    accuracy = correct / total if total > 0 else 0
    step_accuracy = correct_steps / total_steps if total_steps > 0 else 0

    return avg_loss, accuracy, step_accuracy


# ==================== 메인 학습 루프 ====================


def main():
    print("=" * 70)
    print("문장 순서 예측 모델 학습")
    print("=" * 70)

    # 하이퍼파라미터
    # 2025-11-07, 김병현 수정 - 메모리 절약을 위한 설정 조정
    BATCH_SIZE = 16  # 8 → 2 (메모리 부족 방지)
    LEARNING_RATE = 1e-4  # Pointer Network 레이어의 Learning Rate
    BERT_LR = 2e-5  # 2025-11-17, 김병현 수정 - BERT Fine-tuning Learning Rate
    EPOCHS = 20
    MAX_SENTENCES = 12  # 데이터셋에 12개 문장까지 있음
    MAX_LENGTH = 64  # 128 → 64 (문장이 짧으므로 줄임)
    TRAIN_SPLIT = 0.8
    GRADIENT_ACCUMULATION_STEPS = 4  # 실질적 배치 크기 = 2 × 4 = 8

    # Early Stopping 설정
    # 2025-11-13, 김병현 수정 - 과적합 방지를 위한 Early Stopping 추가
    EARLY_STOPPING_PATIENCE = 3  # 검증 정확도가 개선되지 않으면 N epoch 후 중단
    RESUME_TRAINING = True  # True: 기존 모델 이어서 학습, False: 새로 시작
    UNFREEZE_BERT = True  # 2025-11-17, 김병현 수정 - BERT Unfreeze 여부

    # Device 설정
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n✅ Device: {device}")

    # 토크나이저 로드
    print("✅ 토크나이저 로드 중...")
    tokenizer = KoBERTTokenizer.from_pretrained("skt/kobert-base-v1")

    # 데이터셋 로드
    # 2025-11-13, 김병현 수정 - 일부 데이터만 사용하는 옵션 추가
    print("✅ 데이터셋 로드 중...")
    dataset = SentenceOrderDataset(
        "data/sentence_order_dataset.json", tokenizer, max_length=MAX_LENGTH
    )
    print(f"   전체 데이터: {len(dataset)}개")

    # 데이터 일부만 사용 (테스트용)
    USE_SUBSET = False  # True: 일부만 사용, False: 전체 사용
    SUBSET_SIZE = 500  # 사용할 데이터 개수

    if USE_SUBSET and len(dataset) > SUBSET_SIZE:
        # 랜덤하게 일부 선택
        import random

        indices = list(range(len(dataset)))
        random.shuffle(indices)
        selected_indices = indices[:SUBSET_SIZE]
        dataset = torch.utils.data.Subset(dataset, selected_indices)
        print(f"   ⚠️  일부 데이터만 사용: {SUBSET_SIZE}개")

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

    # 기존 모델 로드 여부 확인
    # 2025-11-13, 김병현 수정 - RESUME_TRAINING 옵션 추가
    import os

    if is_running_in_colab():
        pretrained_model_path = "/content/drive/MyDrive/models/sentence_order_model_best.pt"
    else:
        pretrained_model_path = "models/sentence_order_model_best.pt"
    if RESUME_TRAINING and os.path.exists(pretrained_model_path):
        print(f"   🔄 기존 모델 발견: {pretrained_model_path}")
        print(f"   📥 기존 모델 로드 중... (이어서 학습)")
        model.load_state_dict(torch.load(pretrained_model_path, map_location=device))
        print(f"   ✅ 기존 모델 로드 완료!")
    else:
        if not RESUME_TRAINING:
            print(f"   🆕 새로운 모델 생성 (RESUME_TRAINING=False)")
        else:
            print(f"   🆕 새로운 모델 생성 (기존 모델 없음)")

    # BERT Freeze/Unfreeze 설정
    # 2025-11-17, 김병현 수정 - BERT Unfreeze 옵션 추가
    if not UNFREEZE_BERT:
        # BERT를 Freeze (기존 방식)
        for param in model.bert.parameters():
            param.requires_grad = False
        print(f"   🔒 BERT Frozen: BERT 파라미터는 학습하지 않음")
    else:
        # BERT를 Unfreeze (모든 파라미터 학습)
        for param in model.bert.parameters():
            param.requires_grad = True
        print(f"   🔓 BERT Unfrozen: BERT 파라미터도 함께 학습")

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"   전체 파라미터: {total_params:,}")
    print(f"   학습 가능 파라미터: {trainable_params:,}")

    # Optimizer & Loss
    # 2025-11-17, 김병현 수정 - Discriminative Learning Rate 적용 (BERT Unfreeze 시)
    # BERT: 작은 lr (2e-5), Pointer Network: 큰 lr (1e-4)
    if UNFREEZE_BERT:
        optimizer = AdamW(
            [
                # BERT 전체: 작은 lr
                {"params": model.bert.parameters(), "lr": BERT_LR},
                # Pointer Network 레이어: 큰 lr
                {"params": model.sequence_encoder.parameters(), "lr": LEARNING_RATE},
                {"params": model.pointer_decoder.parameters(), "lr": LEARNING_RATE},
            ],
            weight_decay=0.01,
        )
    else:
        # BERT가 Frozen일 때는 Pointer Network만 학습
        optimizer = AdamW(
            [
                {"params": model.sequence_encoder.parameters(), "lr": LEARNING_RATE},
                {"params": model.pointer_decoder.parameters(), "lr": LEARNING_RATE},
            ],
            weight_decay=0.01,
        )

    criterion = nn.CrossEntropyLoss(ignore_index=-100)

    # 학습 정보 출력
    print("\n" + "=" * 70)
    print("⚙️  학습 설정")
    print("=" * 70)
    print(f"   🔄 Epochs: {EPOCHS}")
    print(
        f"   📦 Batch Size: {BATCH_SIZE} (실질적: {BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS})"
    )
    print(f"   💾 Gradient Accumulation: {GRADIENT_ACCUMULATION_STEPS} steps")
    print(f"   📏 Max Length: {MAX_LENGTH} tokens")
    print(f"   🛑 Early Stopping: Patience {EARLY_STOPPING_PATIENCE} epochs")
    # 2025-11-17, 김병현 수정 - BERT Unfreeze 정보 추가
    if UNFREEZE_BERT:
        print(f"   🔓 BERT Unfreeze: True (LR: {BERT_LR}, Pointer LR: {LEARNING_RATE})")
    else:
        print(f"   🔒 BERT Frozen: True (Pointer LR: {LEARNING_RATE})")

    # 학습 시작
    print("\n" + "=" * 70)
    print("🚀 학습 시작")
    print("=" * 70)

    # 모델 저장 디렉토리 생성
    # 2025-11-17, 김병현 수정 - models 폴더가 없으면 생성
    os.makedirs("models", exist_ok=True)

    # Early Stopping 변수 초기화
    # 2025-11-13, 김병현 수정 - Early Stopping 구현
    best_val_acc = 0
    best_step_acc = 0
    patience_counter = 0  # 개선되지 않은 연속 epoch 수

    for epoch in range(EPOCHS):
        print(f"\n📍 Epoch {epoch + 1}/{EPOCHS}")

        # 학습
        train_loss, train_acc, train_step_acc = train_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            device,
            gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        )
        print(
            f"   Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Train Step Acc: {train_step_acc:.4f}"
        )

        # 검증
        val_loss, val_acc, val_step_acc = evaluate(model, val_loader, criterion, device)
        print(
            f"   Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, Val Step Acc: {val_step_acc:.4f}"
        )

        # 최고 모델 저장 및 Early Stopping 체크
        # 스텝 정확도, 전체 문장을 맞춘 정확도 둘 다 고려
        if val_step_acc > best_step_acc:
            best_step_acc = val_step_acc
            patience_counter = 0  # 개선되었으므로 카운터 리셋
            torch.save(model.state_dict(), "models/sentence_order_model_best.pt")
            print(f"   ✨ 최고 모델 저장! (Val Step Acc: {val_step_acc:.4f})")
        elif val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0  # 개선되었으므로 카운터 리셋
            if is_running_in_colab():
                torch.save(model.state_dict(), "models/sentence_order_model_best.pt")
            else:
                torch.save(model.state_dict(), "models/sentence_order_model_best.pt")
            print(f"   ✨ 최고 모델 저장! (Val Acc: {val_acc:.4f})")
        else:
            patience_counter += 1
            print(
                f"   ⚠️  검증 정확도 개선 없음 ({patience_counter}/{EARLY_STOPPING_PATIENCE})"
            )

            # Early Stopping 조건 충족
            if patience_counter >= EARLY_STOPPING_PATIENCE:
                print(
                    f"\n🛑 Early Stopping: {EARLY_STOPPING_PATIENCE} epoch 동안 개선 없음"
                )
                print(f"   최고 검증 정확도: {best_val_acc:.4f}")
                break

    print("\n" + "=" * 70)
    print(f"✅ 학습 완료! 최고 검증 정확도: {best_val_acc:.4f}")
    print("=" * 70)

    # 최종 모델 저장
    torch.save(model.state_dict(), "models/sentence_order_model_final.pt")
    print("✅ 최종 모델 저장: models/sentence_order_model_final.pt")


if __name__ == "__main__":
    main()
