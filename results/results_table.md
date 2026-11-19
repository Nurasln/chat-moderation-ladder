| model                        |   macro_f1 |   size_mb |   latency_mean_ms |   latency_median_ms |   latency_p95_ms |   latency_1thread_mean_ms |   runs | under_50ms_1thread   |
|:-----------------------------|-----------:|----------:|------------------:|--------------------:|-----------------:|--------------------------:|-------:|:---------------------|
| 1a · Lexicon + edit distance |     0.7186 |     0.001 |              0.54 |                0.47 |             1.17 |                      0.51 |    300 | yes                  |
| 1b · TF-IDF + LogReg         |     0.7258 |     0.922 |              0.44 |                0.43 |             0.52 |                      0.44 |    300 | yes                  |
| 2 · Word2Vec + FFN           |     0.6714 |     3.709 |              0.12 |                0.12 |             0.18 |                      0.12 |    300 | yes                  |
| 3 · BERT (FP32)              |     0.7821 |   438.672 |             20.38 |               19.38 |            29.92 |                     65.34 |    300 | no                   |
| 4 · BERT INT8 (quantized)    |     0.7971 |   182.208 |             19.72 |               19.11 |            26.31 |                     29.55 |    300 | yes                  |

Test set: 860 messages (TweetEval offensive test split). Latency: CPU, batch size 1, mean of 300 single-message runs after 10 warm-up runs, measured with 14 torch threads and again with 1 thread. Budget: 50 ms per message.
Hardware: 13th Gen Intel(R) Core(TM) i7-13650HX, 20 logical cores, 14 torch threads, Windows-11-10.0.26300-SP0, torch 2.11.0+cu128.
