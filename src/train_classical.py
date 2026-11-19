"""Model 1 - classical baselines (technique area B).

  1a. Rule baseline : regex normalization + edit-distance lexicon matcher
  1b. TF-IDF + Logistic Regression

Hyper-parameters are chosen on the VALIDATION set. The TEST set is used once, at the end.

Run:  python -m src.train_classical
"""
import json

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.pipeline import make_pipeline

from src.config import DATA_DIR, MODELS_DIR, SEED
from src.evaluate import evaluate
from src.lexicon import LexiconMatcher, build_lexicon
from src.normalize import normalize


def load(split):
    df = pd.read_csv(DATA_DIR / f"{split}.csv")
    return df["text"].tolist(), [normalize(t) for t in df["text"]], df["label"].tolist()


def main():
    MODELS_DIR.mkdir(exist_ok=True)
    _, X_train, y_train = load("train")
    _, X_val, y_val = load("val")
    raw_test, X_test, y_test = load("test")

    # ---------- 1a. Lexicon rule baseline ----------
    best = None
    for min_count in [5, 8, 15]:
        for min_ratio in [0.7, 0.8, 0.9]:
            lexicon = build_lexicon(X_train, y_train, min_count, min_ratio)
            f1 = f1_score(y_val, LexiconMatcher(lexicon).predict(X_val), average="macro")
            print(f"  lexicon min_count={min_count:2d} min_ratio={min_ratio} size={len(lexicon):4d} val macro-F1={f1:.4f}")
            if best is None or f1 > best[0]:
                best = (f1, min_count, min_ratio, lexicon)

    _, min_count, min_ratio, lexicon = best
    print(f"-> chosen lexicon: min_count={min_count}, min_ratio={min_ratio}, {len(lexicon)} words")
    (MODELS_DIR / "lexicon.json").write_text(json.dumps(lexicon, indent=0))
    evaluate("1a_lexicon_rule", raw_test, y_test, LexiconMatcher(lexicon).predict(X_test),
             extra={"lexicon_size": len(lexicon), "min_count": min_count, "min_ratio": min_ratio})

    # ---------- 1b. TF-IDF + Logistic Regression ----------
    def make_model(C):
        return make_pipeline(
            # words + word pairs; ignore words seen in < 2 messages; log-scaled term frequency
            TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True),
            # class_weight="balanced": offensive is the minority class, give it more weight
            LogisticRegression(C=C, class_weight="balanced", max_iter=2000, random_state=SEED),
        )

    best = None
    for C in [0.25, 0.5, 1, 2, 4, 8]:
        model = make_model(C).fit(X_train, y_train)
        f1 = f1_score(y_val, model.predict(X_val), average="macro")
        print(f"  tfidf C={C:<5} val macro-F1={f1:.4f}")
        if best is None or f1 > best[0]:
            best = (f1, C, model)

    _, C, model = best
    print(f"-> chosen C={C}")
    joblib.dump(model, MODELS_DIR / "tfidf_logreg.joblib")
    evaluate("1b_tfidf_logreg", raw_test, y_test, model.predict(X_test),
             scores=model.predict_proba(X_test)[:, 1].round(4),
             extra={"C": C, "vocab_size": len(model[0].vocabulary_)})

    # Most "offensive" words according to the model -> nice for the slides
    vocab = model[0].get_feature_names_out()
    top = model[1].coef_[0].argsort()[-15:][::-1]
    print("top offensive features:", ", ".join(vocab[i] for i in top))


if __name__ == "__main__":
    main()
