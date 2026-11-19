# Real-Time Chat Moderation: from TF-IDF to a quantized BERT

SWE015 homework, Use Case 4. An English chat app labels each message **clean / offensive**, shows how confident it is, masks offensive messages and logs how long each prediction took on CPU.

We compare four models on the same held-out test set:

| # | Model | Technique area |
|---|-------|----------------|
| 1 | Regex normalization + edit-distance lexicon + TF-IDF / Logistic Regression | B · Classical NLP |
| 2 | Our own Word2Vec + feedforward network (SGD) | A · Neural networks |
| 3 | Fine-tuned `bert-base-uncased` | C · Transformers |
| 4 | BERT with dynamic INT8 quantization (used in the app) | D · Efficient models |

> Work in progress. Setup steps, the single run command and the results table come in later PRs.

## Data
TweetEval `offensive` (OffensEval 2019), loaded from [cardiffnlp/tweet_eval](https://huggingface.co/datasets/cardiffnlp/tweet_eval). We use the official train / validation / test splits.

## Development setup (current)
```bash
uv venv .venv --python 3.12
uv pip install --python .venv -r requirements.txt
.venv\Scripts\python -m src.data              # downloads data into data/
.venv\Scripts\python -m src.train_classical   # model 1: lexicon rule + TF-IDF/LogReg
```

Each training script writes `results/metrics/<model>.json` (scores on the test set) and
`results/predictions/<model>.csv` (every test message with its prediction).
