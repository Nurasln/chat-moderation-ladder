"""Step 1 - Download the dataset and save fixed train/val/test splits as CSV.

Dataset: TweetEval "offensive" (= OffensEval 2019, SemEval Task 6, English tweets).
  - Source: https://huggingface.co/datasets/cardiffnlp/tweet_eval
  - Papers: Zampieri et al. 2019 (OffensEval), Barbieri et al. 2020 (TweetEval)
  - Labels: 0 = not offensive, 1 = offensive

We use the official splits as-is. The TEST split is the one held-out set that
ALL four models are evaluated on (requirement R3). We never train on it.

Run:  python -m src.data
"""
import pandas as pd
from datasets import load_dataset

from src.config import DATA_DIR, LABELS


def main():
    DATA_DIR.mkdir(exist_ok=True)
    ds = load_dataset("cardiffnlp/tweet_eval", "offensive")

    # HF calls it "validation"; we save it as val.csv
    for hf_name, our_name in [("train", "train"), ("validation", "val"), ("test", "test")]:
        df = ds[hf_name].to_pandas()[["text", "label"]]
        df.to_csv(DATA_DIR / f"{our_name}.csv", index=False)

        # Print size and class balance -> goes into the "Data" slide
        counts = df["label"].value_counts().sort_index()
        share = ", ".join(f"{LABELS[i]}={n} ({n / len(df):.0%})" for i, n in counts.items())
        print(f"{our_name:5s}: {len(df):6d} messages | {share}")


if __name__ == "__main__":
    main()
