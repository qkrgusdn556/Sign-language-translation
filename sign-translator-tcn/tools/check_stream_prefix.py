import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
import project_paths
"""Synthetic joined-recording regression, not a live continuous-sign accuracy test."""
import json
from pathlib import Path
import numpy as np
from tcn_backend import runtime


def replay(first, second, minimum):
    stream = first + second
    candidate = emitted = None
    since = 0
    words = []
    for end in range(minimum, len(stream) + 1, 2):
        ranked = sorted(runtime.predict(stream[max(0, end - 20):end])["scores"], key=lambda item: item["score"], reverse=True)
        best = ranked[0]
        now = (end - 1) * 75
        if best["score"] < 0.85 or best["score"] - ranked[1]["score"] < 0.3:
            candidate = None
            continue
        if candidate != best["label"]:
            candidate, since = best["label"], now
        if now - since >= 300 and emitted != candidate:
            words.append(candidate)
            emitted = candidate
    return words


def shorter(sequence):
    indices = np.linspace(0, len(sequence) - 1, max(12, round(len(sequence) * 0.75))).round().astype(int)
    return [sequence[index] for index in indices]


def main():
    root = project_paths.ROOT
    data = json.loads((root / "data" / "sign-samples.json").read_text(encoding="utf-8"))
    first = [frame[:130] for frame in next(sample["sequence"] for sample in data["samples"] if sample["label"] == "안녕하세요")]
    second = [frame[:130] for frame in next(sample["sequence"] for sample in data["samples"] if sample["label"] == "만나서 반가워요")]
    before = {"original_length": replay(first, second, 19), "shortened": replay(shorter(first), shorter(second), 19)}
    runtime.train([sample for sample in data["samples"] if sample.get("sequence")])
    if runtime.job != "done":
        raise RuntimeError(runtime.error)
    after = {"original_length": replay(first, second, 12), "shortened": replay(shorter(first), shorter(second), 12)}
    report = {"before": before, "after": after,
              "limitations": "Joined isolated training recordings with assumed 75ms cadence; no actual transition, new-session or live accuracy validation."}
    (root / "stream-prefix-report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
