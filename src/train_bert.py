"""Model 3 - fine-tune BERT for clean/offensive classification (technique area C).

bert-base-uncased was pretrained by Google on ~3.3B words (BooksCorpus + Wikipedia).
We add a 2-class head on top of its [CLS] output and fine-tune the WHOLE network
on our training messages. Tokenization is WordPiece (subwords): frequent words stay whole,
rare words are split into known pieces that start with "##".

Same recipe as model 2 (forward -> loss -> backward -> step), but:
  - optimizer AdamW with a small learning rate (2e-5), so we don't destroy pretrained knowledge
  - a few epochs only (BERT already "knows" English)
  - we keep the epoch with the best VALIDATION macro-F1, then evaluate once on TEST

Run (GPU recommended, ~5 min on an RTX 4060):  python -m src.train_bert
"""
import json
import time

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

from src.config import DATA_DIR, LABELS, MODELS_DIR, SEED
from src.evaluate import evaluate
from src.normalize import normalize

BASE_MODEL = "bert-base-uncased"
OUT_DIR = MODELS_DIR / "bert"
MAX_LEN, BATCH, LR, EPOCHS = 128, 32, 2e-5, 4


def load(split):
    df = pd.read_csv(DATA_DIR / f"{split}.csv")
    return df["text"].tolist(), [normalize(t) for t in df["text"]], df["label"].tolist()


@torch.no_grad()
def predict_proba(model, tokenizer, texts, device):
    model.eval()
    probs = []
    for i in range(0, len(texts), 64):
        batch = tokenizer(texts[i:i + 64], truncation=True, max_length=MAX_LEN, padding=True, return_tensors="pt").to(device)
        probs.append(torch.softmax(model(**batch).logits.float(), dim=1)[:, 1].cpu())
    return torch.cat(probs).numpy()


def main():
    torch.manual_seed(SEED)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", device, torch.cuda.get_device_name(0) if device == "cuda" else "")

    _, X_train, y_train = load("train")
    _, X_val, y_val = load("val")
    raw_test, X_test, y_test = load("test")

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    print("WordPiece example:", tokenizer.tokenize("you idiots are disgusting"))
    model = AutoModelForSequenceClassification.from_pretrained(
        BASE_MODEL, num_labels=2, id2label=dict(enumerate(LABELS)), label2id={l: i for i, l in enumerate(LABELS)}
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    steps = EPOCHS * ((len(X_train) + BATCH - 1) // BATCH)
    # learning rate: warm up for the first 10% of steps, then decrease linearly to 0
    scheduler = get_linear_schedule_with_warmup(optimizer, int(0.1 * steps), steps)
    class_weights = torch.tensor(len(y_train) / (2 * np.bincount(y_train)), dtype=torch.float32, device=device)
    loss_fn = torch.nn.CrossEntropyLoss(weight=class_weights)
    use_amp = device == "cuda"  # mixed precision (fp16) on GPU -> faster training, less memory

    best_f1, best_epoch, t0 = -1, 0, time.time()
    y_tr = torch.tensor(y_train)
    for epoch in range(1, EPOCHS + 1):
        model.train()
        perm = torch.randperm(len(X_train)).tolist()
        for step, i in enumerate(range(0, len(perm), BATCH)):
            idx = perm[i:i + BATCH]
            batch = tokenizer([X_train[j] for j in idx], truncation=True, max_length=MAX_LEN,
                              padding=True, return_tensors="pt").to(device)
            labels = y_tr[idx].to(device)

            optimizer.zero_grad()
            with torch.autocast(device_type=device, dtype=torch.float16, enabled=use_amp):
                loss = loss_fn(model(**batch).logits.float(), labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # avoid exploding gradients
            optimizer.step()
            scheduler.step()
            if step % 100 == 0:
                print(f"  epoch {epoch} step {step:4d}  loss={loss.item():.4f}")

        val_f1 = f1_score(y_val, (predict_proba(model, tokenizer, X_val, device) >= 0.5).astype(int), average="macro")
        print(f"epoch {epoch}: val macro-F1={val_f1:.4f}  ({time.time() - t0:.0f}s)")
        if val_f1 > best_f1:
            best_f1, best_epoch = val_f1, epoch
            model.save_pretrained(OUT_DIR)       # keep only the best epoch on disk
            tokenizer.save_pretrained(OUT_DIR)

    train_seconds = round(time.time() - t0)
    print(f"-> best epoch {best_epoch}, val macro-F1={best_f1:.4f}, training took {train_seconds}s")

    # Reload the best checkpoint and evaluate once on the test set
    model = AutoModelForSequenceClassification.from_pretrained(OUT_DIR).to(device)
    probs = predict_proba(model, tokenizer, X_test, device)
    evaluate("3_bert_fp32", raw_test, y_test, (probs >= 0.5).astype(int).tolist(), scores=probs.round(4),
             extra={"base_model": BASE_MODEL, "best_epoch": best_epoch, "val_macro_f1": round(best_f1, 4),
                    "lr": LR, "batch": BATCH, "train_seconds": train_seconds,
                    "train_device": torch.cuda.get_device_name(0) if device == "cuda" else "cpu"})


if __name__ == "__main__":
    main()
