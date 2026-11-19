# Quantization experiments (validation set, 1,324 messages, CPU)

Goal: find out how much dynamic INT8 quantization changes the fine-tuned BERT.
Measured on the **validation** set so the test set stays untouched.

| Variant | Val macro-F1 | Same label as FP32 | Mean abs. change of P(offensive) |
|---|---|---|---|
| FP32 (reference) | 0.7759 | 100% | 0.000 |
| INT8, all `nn.Linear`, per-tensor (**used**) | 0.7717 | 88.9% | 0.124 |
| INT8, encoder only (classifier + pooler kept FP32) | 0.7725 | 88.8% | 0.125 |
| INT8, all `nn.Linear`, per-channel | 0.7715 | 89.9% | 0.122 |

Findings:
- The three INT8 variants are within 0.001 macro-F1 of each other, so we use the simplest one.
- About 11% of labels flip after quantization. The error comes from the encoder,
  not the classification head. A likely cause is the known activation outliers in BERT
  that 8-bit dynamic quantization rounds badly (hypothesis, not verified).
- On the **test** set INT8 scores higher than FP32 (0.7971 vs 0.7821) but it predicts
  "clean" more often: offensive recall falls from 73% to 60%. On validation INT8 is slightly
  lower than FP32. So the test gain is not a real improvement, it comes from a shift toward
  the majority class.
