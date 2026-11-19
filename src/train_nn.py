"""Model 2 - our own Word2Vec + a feedforward network trained with SGD (technique area A).

Step 1: train Word2Vec (skip-gram) on the TRAINING messages only - no pretrained vectors.
Step 2: each message = average of its word vectors (100 numbers).
Step 3: a small feedforward network learns clean/offensive from these vectors,
        trained with mini-batch SGD + backpropagation. We keep the epoch with the
        best VALIDATION macro-F1 (early stopping), then evaluate once on TEST.

Run:  python -m src.train_nn
"""
import json

import numpy as np
import pandas as pd
import torch
from gensim.models import Word2Vec
from sklearn.metrics import f1_score
from torch import nn

from src.config import DATA_DIR, MODELS_DIR, SEED
from src.evaluate import evaluate
from src.nn_model import FeedForward, message_vector
from src.normalize import normalize

DIM = 100


def load(split):
    df = pd.read_csv(DATA_DIR / f"{split}.csv")
    return df["text"].tolist(), [normalize(t) for t in df["text"]], np.array(df["label"])


def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    MODELS_DIR.mkdir(exist_ok=True)

    _, X_train, y_train = load("train")
    _, X_val, y_val = load("val")
    raw_test, X_test, y_test = load("test")

    # ---------- Step 1: Word2Vec on our own training messages ----------
    w2v = Word2Vec(
        sentences=[t.split() for t in X_train],
        vector_size=DIM,   # each word becomes 100 numbers
        window=5,          # context = 5 words left and right
        min_count=2,       # ignore words seen only once
        sg=1,              # skip-gram: predict context words from the center word
        epochs=20,
        workers=1, seed=SEED,  # 1 worker -> reproducible result
    )
    wv = w2v.wv
    wv.save(str(MODELS_DIR / "word2vec.kv"))
    print(f"Word2Vec vocabulary: {len(wv)} words")
    for probe in ["idiot", "stupid", "love"]:
        if probe in wv:
            print(f"  most similar to '{probe}':", ", ".join(w for w, _ in wv.most_similar(probe, topn=6)))

    # ---------- Step 2: message vectors ----------
    def to_tensor(texts):
        return torch.tensor(np.stack([message_vector(t, wv, DIM) for t in texts]))

    Xtr, Xva, Xte = to_tensor(X_train), to_tensor(X_val), to_tensor(X_test)
    ytr = torch.tensor(y_train)

    # ---------- Step 3: feedforward network, SGD + backprop ----------
    model = FeedForward(dim=DIM, hidden=64, dropout=0.3)
    # class weights: offensive is ~1/3 of the data, so its errors count ~2x
    counts = np.bincount(y_train)
    class_weights = torch.tensor(len(y_train) / (2 * counts), dtype=torch.float32)
    loss_fn = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.05, momentum=0.9)

    best_f1, best_state, best_epoch, patience = -1, None, 0, 0
    for epoch in range(1, 101):
        model.train()
        perm = torch.randperm(len(Xtr))
        for i in range(0, len(Xtr), 64):              # mini-batches of 64 messages
            idx = perm[i:i + 64]
            optimizer.zero_grad()                     # 1. clear old gradients
            loss = loss_fn(model(Xtr[idx]), ytr[idx]) # 2. forward pass + loss
            loss.backward()                           # 3. backpropagation: compute gradients
            optimizer.step()                          # 4. SGD: move weights against the gradient

        model.eval()
        with torch.no_grad():
            val_f1 = f1_score(y_val, model(Xva).argmax(1).numpy(), average="macro")
        if val_f1 > best_f1:
            best_f1, best_epoch, patience = val_f1, epoch, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            patience += 1
        if epoch % 5 == 0 or patience == 0:
            print(f"  epoch {epoch:3d}  loss={loss.item():.4f}  val macro-F1={val_f1:.4f}")
        if patience >= 15:                            # no improvement for 15 epochs -> stop
            break

    print(f"-> best epoch {best_epoch}, val macro-F1={best_f1:.4f}")
    model.load_state_dict(best_state)
    torch.save(model.state_dict(), MODELS_DIR / "ffn.pt")
    (MODELS_DIR / "ffn_config.json").write_text(json.dumps({"dim": DIM, "hidden": 64, "dropout": 0.3}))

    model.eval()
    with torch.no_grad():
        probs = torch.softmax(model(Xte), dim=1)[:, 1].numpy()
    evaluate("2_word2vec_ffn", raw_test, y_test.tolist(), (probs >= 0.5).astype(int).tolist(),
             scores=probs.round(4),
             extra={"w2v_vocab": len(wv), "dim": DIM, "best_epoch": best_epoch, "optimizer": "SGD lr=0.05 momentum=0.9"})


if __name__ == "__main__":
    main()
