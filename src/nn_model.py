"""Model 2 building blocks: message -> averaged Word2Vec vector -> feedforward network.

Kept in its own file because both training (src/train_nn.py) and the app use it.
"""
import numpy as np
import torch
from torch import nn


def message_vector(normalized_text, word_vectors, dim):
    """Average the Word2Vec vectors of the words we know. Unknown words are skipped."""
    vecs = [word_vectors[w] for w in normalized_text.split() if w in word_vectors]
    if not vecs:
        return np.zeros(dim, dtype=np.float32)   # no known word -> zero vector
    return np.mean(vecs, axis=0).astype(np.float32)


class FeedForward(nn.Module):
    """input (dim) -> hidden (ReLU + dropout) -> 2 logits (clean / offensive)."""

    def __init__(self, dim=100, hidden=64, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),   # randomly zero 30% of hidden units while training -> less overfitting
            nn.Linear(hidden, 2),
        )

    def forward(self, x):
        return self.net(x)
