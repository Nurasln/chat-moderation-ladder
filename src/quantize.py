"""Model 4 - dynamic INT8 quantization of the fine-tuned BERT (technique area D).

What changes:  every nn.Linear weight matrix (most of BERT's 110M parameters) goes
from float32 (4 bytes) to int8 (1 byte) + one scale factor. Embeddings and LayerNorm
stay float32. No retraining needed ("post-training dynamic quantization").

Then we evaluate the INT8 model on the SAME test set, on CPU (where it runs in the app).

Run:  python -m src.quantize
"""
import shutil

import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification

from src.config import DATA_DIR, MODELS_DIR
from src.evaluate import evaluate
from src.predictors import BertInt8Predictor, BertPredictor, quantize_bert


def main():
    src_dir, out_dir = MODELS_DIR / "bert", MODELS_DIR / "bert_int8"
    out_dir.mkdir(exist_ok=True)

    model = AutoModelForSequenceClassification.from_pretrained(src_dir).eval()
    qmodel = quantize_bert(model)
    torch.save(qmodel.state_dict(), out_dir / "model_int8.pt")
    for f in ["config.json", "tokenizer.json", "tokenizer_config.json"]:
        shutil.copy(src_dir / f, out_dir / f)

    fp32_mb = (src_dir / "model.safetensors").stat().st_size / 1e6
    int8_mb = (out_dir / "model_int8.pt").stat().st_size / 1e6
    print(f"weights: FP32 {fp32_mb:.1f} MB -> INT8 {int8_mb:.1f} MB ({fp32_mb / int8_mb:.1f}x smaller)")

    # Evaluate FP32 and INT8 exactly the way the app uses them (load from disk, CPU, one message at a time),
    # so the comparison is fair. This overwrites the GPU-batch test score written by train_bert.py.
    df = pd.read_csv(DATA_DIR / "test.csv")
    for name, predictor_cls, extra in [
        ("3_bert_fp32", BertPredictor, {"precision": "float32", "eval_device": "cpu"}),
        ("4_bert_int8", BertInt8Predictor, {"precision": "int8 (dynamic, nn.Linear weights)", "eval_device": "cpu"}),
    ]:
        predictor = predictor_cls()
        probs = [predictor.predict_proba(t) for t in df["text"]]
        evaluate(name, df["text"].tolist(), df["label"].tolist(), [int(p >= 0.5) for p in probs],
                 scores=[round(p, 4) for p in probs], extra=extra)


if __name__ == "__main__":
    main()
