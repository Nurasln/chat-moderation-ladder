"""Results table (R4): task metric + model size + CPU latency for every model.

Latency = time for ONE message, end to end (normalize -> features -> model -> probability),
on CPU, batch size 1, like a chat server answering one message at a time.
  - 10 warm-up calls (not timed: first calls are slow due to caching / lazy init)
  - then N_RUNS timed calls on different real test messages, we report mean / median / p95
  - measured twice: with all CPU threads (torch default) and with 1 thread
    (a busy chat server usually gives each request a single core)

Run:  python -m src.benchmark          -> results/results_table.md + .csv + hardware.json
"""
import json
import os
import platform
import statistics
import time

import pandas as pd
import torch

from src.config import DATA_DIR, RESULTS_DIR
from src.predictors import ALL_PREDICTORS, model_size_mb

N_WARMUP, N_RUNS = 10, 300
LATENCY_BUDGET_MS = 50  # our requirement: one message must be classified in < 50 ms on CPU
METRIC_FILES = {
    "LexiconPredictor": "1a_lexicon_rule", "TfidfPredictor": "1b_tfidf_logreg",
    "Word2VecPredictor": "2_word2vec_ffn", "BertPredictor": "3_bert_fp32", "BertInt8Predictor": "4_bert_int8",
}


def cpu_name():
    if platform.system() == "Windows":
        import winreg
        key = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0")
        return winreg.QueryValueEx(key, "ProcessorNameString")[0].strip()
    try:
        return next(l.split(":")[1].strip() for l in open("/proc/cpuinfo") if l.startswith("model name"))
    except (OSError, StopIteration):
        return platform.processor()


def main():
    torch.set_grad_enabled(False)
    texts = pd.read_csv(DATA_DIR / "test.csv")["text"].tolist()
    hardware = {
        "cpu": cpu_name(), "logical_cores": os.cpu_count(), "torch_threads": torch.get_num_threads(),
        "os": platform.platform(), "python": platform.python_version(), "torch": torch.__version__,
        "device_used": "CPU only (GPU not used for latency)",
    }
    print("hardware:", hardware)

    def time_one_by_one(predictor):
        for t in texts[:N_WARMUP]:
            predictor.predict_proba(t)
        times_ms = []
        for t in texts[N_WARMUP:N_WARMUP + N_RUNS]:
            start = time.perf_counter()
            predictor.predict_proba(t)
            times_ms.append((time.perf_counter() - start) * 1000)
        return times_ms

    default_threads = torch.get_num_threads()
    rows = []
    for cls in ALL_PREDICTORS:
        predictor = cls()
        torch.set_num_threads(default_threads)
        times_ms = time_one_by_one(predictor)
        torch.set_num_threads(1)
        times_1t = time_one_by_one(predictor)
        torch.set_num_threads(default_threads)

        metrics = json.loads((RESULTS_DIR / "metrics" / f"{METRIC_FILES[cls.__name__]}.json").read_text())
        mean = statistics.mean(times_ms)
        rows.append({
            "model": cls.name,
            "macro_f1": metrics["macro_f1"],
            "size_mb": model_size_mb(cls),
            "latency_mean_ms": round(mean, 2),
            "latency_median_ms": round(statistics.median(times_ms), 2),
            "latency_p95_ms": round(sorted(times_ms)[int(0.95 * len(times_ms)) - 1], 2),
            "latency_1thread_mean_ms": round(statistics.mean(times_1t), 2),
            "runs": len(times_ms),
            f"under_{LATENCY_BUDGET_MS}ms_1thread": "yes" if statistics.mean(times_1t) < LATENCY_BUDGET_MS else "no",
        })
        print(rows[-1])
        del predictor

    df = pd.DataFrame(rows)
    df.to_csv(RESULTS_DIR / "results_table.csv", index=False)
    (RESULTS_DIR / "hardware.json").write_text(json.dumps(hardware, indent=2))
    md = df.to_markdown(index=False)
    note = (f"\n\nTest set: {len(texts)} messages (TweetEval offensive test split). Latency: CPU, batch size 1, "
            f"mean of {N_RUNS} single-message runs after {N_WARMUP} warm-up runs, measured with "
            f"{default_threads} torch threads and again with 1 thread. Budget: {LATENCY_BUDGET_MS} ms per message.\n"
            f"Hardware: {hardware['cpu']}, {hardware['logical_cores']} logical cores, "
            f"{hardware['torch_threads']} torch threads, {hardware['os']}, torch {hardware['torch']}.\n")
    (RESULTS_DIR / "results_table.md").write_text(md + note, encoding="utf-8")
    print("\n" + md + note)


if __name__ == "__main__":
    main()
