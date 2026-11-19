"""One common interface for all models, used by BOTH the benchmark and the app.

    predictor.predict_proba(raw_text) -> probability that the message is offensive

Each predict_proba call does the full work for one message: normalize -> features -> model.
That is exactly what we time in the benchmark, so latency = what a user would wait for.
"""
import json
import warnings

import joblib
import torch
from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer

from src.config import MODELS_DIR
from src.lexicon import LexiconMatcher
from src.normalize import normalize

warnings.filterwarnings("ignore", message=".*torch.ao.quantization.*")  # deprecation notice, API still works in torch 2.11

MAX_LEN = 128


class LexiconPredictor:
    name = "1a · Lexicon + edit distance"
    files = ["lexicon.json"]

    def __init__(self):
        self.matcher = LexiconMatcher(json.loads((MODELS_DIR / "lexicon.json").read_text()))

    def predict_proba(self, text):
        return 1.0 if self.matcher.offensive_words(normalize(text)) else 0.0


class TfidfPredictor:
    name = "1b · TF-IDF + LogReg"
    files = ["tfidf_logreg.joblib"]

    def __init__(self):
        self.model = joblib.load(MODELS_DIR / "tfidf_logreg.joblib")

    def predict_proba(self, text):
        return float(self.model.predict_proba([normalize(text)])[0, 1])


class Word2VecPredictor:
    name = "2 · Word2Vec + FFN"
    files = ["word2vec.kv", "ffn.pt", "ffn_config.json"]

    def __init__(self):
        from gensim.models import KeyedVectors
        from src.nn_model import FeedForward

        self.wv = KeyedVectors.load(str(MODELS_DIR / "word2vec.kv"))
        cfg = json.loads((MODELS_DIR / "ffn_config.json").read_text())
        self.dim = cfg["dim"]
        self.model = FeedForward(**cfg)
        self.model.load_state_dict(torch.load(MODELS_DIR / "ffn.pt"))
        self.model.eval()

    @torch.no_grad()
    def predict_proba(self, text):
        from src.nn_model import message_vector

        x = torch.from_numpy(message_vector(normalize(text), self.wv, self.dim)).unsqueeze(0)
        return float(torch.softmax(self.model(x), dim=1)[0, 1])


def quantize_bert(model):
    """Dynamic INT8 quantization: weights of every nn.Linear layer are stored as 8-bit integers
    (instead of 32-bit floats); activations are quantized on the fly at inference time. CPU only."""
    from torch.ao.quantization import quantize_dynamic

    return quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)


class BertPredictor:
    name = "3 · BERT (FP32)"
    files = ["bert/config.json", "bert/model.safetensors", "bert/tokenizer.json", "bert/tokenizer_config.json"]

    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained(MODELS_DIR / "bert")
        self.model = AutoModelForSequenceClassification.from_pretrained(MODELS_DIR / "bert").eval()

    @torch.no_grad()
    def predict_proba(self, text):
        batch = self.tokenizer(normalize(text), truncation=True, max_length=MAX_LEN, return_tensors="pt")
        return float(torch.softmax(self.model(**batch).logits, dim=1)[0, 1])


class BertInt8Predictor(BertPredictor):
    name = "4 · BERT INT8 (quantized)"
    files = ["bert_int8/config.json", "bert_int8/model_int8.pt", "bert_int8/tokenizer.json", "bert_int8/tokenizer_config.json"]

    def __init__(self):
        d = MODELS_DIR / "bert_int8"
        self.tokenizer = AutoTokenizer.from_pretrained(d)
        # Build an empty BERT with the same shape, quantize it, then load the saved INT8 weights into it
        model = AutoModelForSequenceClassification.from_config(AutoConfig.from_pretrained(d))
        self.model = quantize_bert(model.eval())
        self.model.load_state_dict(torch.load(d / "model_int8.pt", weights_only=False))
        self.model.eval()


ALL_PREDICTORS = [LexiconPredictor, TfidfPredictor, Word2VecPredictor, BertPredictor, BertInt8Predictor]


def model_size_mb(predictor_cls):
    """Size on disk of every file the model needs (what you would have to download)."""
    return round(sum((MODELS_DIR / f).stat().st_size for f in predictor_cls.files) / 1e6, 3)
