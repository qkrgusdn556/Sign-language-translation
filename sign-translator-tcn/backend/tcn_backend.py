import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
import project_paths
"""Local TCN inference and single-job training; isolated from the DTW app."""
import json
import os
import shutil
from pathlib import Path
import sys
import threading
import hashlib

ROOT = project_paths.ROOT
EVALUATION = ROOT.parent / "slrt-evaluation"
MODEL = ROOT / "data" / "korean-position-tcn.pt"
INITIAL_MODEL = EVALUATION / "assets" / "tcn-motion-probe" / "korean-motion-tcn.pt"


def position_recordings(samples):
    from sample_labels import REST_LABEL
    motion = [sample for sample in samples if sample.get("sequence") and sample.get('label') != REST_LABEL]
    from web_server import valid_samples
    if not valid_samples(motion):
        raise ValueError("Invalid motion recordings")
    usable = [sample for sample in motion if all(len(frame) in (130, 136) for frame in sample["sequence"])]
    excluded = [sample for sample in motion if any(len(frame) == 126 for frame in sample["sequence"])]
    return usable, excluded


class TcnRuntime:
    def __init__(self):
        self.lock = threading.RLock()
        self.model = None
        self.job = "idle"
        self.error = None
        self.model_source = None
        self.training_counts = {}

    def dependencies(self):
        import torch
        from hand_tcn import MotionTCN, read_motion, fit
        torch.set_num_threads(1)
        return torch, MotionTCN, read_motion, fit

    def load(self):
        if self.model is not None:
            return
        try:
            torch, MotionTCN, _, _ = self.dependencies()
            source = MODEL
            if not source.exists():
                raise RuntimeError("Position-aware TCN needs training; previous models are preserved")
            checkpoint = torch.load(source, map_location="cpu", weights_only=True)
            self.model_sha256 = hashlib.sha256(source.read_bytes()).hexdigest()
            if checkpoint.get("input_dim") != 130 or checkpoint.get("feature_schema") != "shoulder-hands-130-v1":
                raise RuntimeError("Incompatible TCN model; train a position-aware model")
            model = MotionTCN(len(checkpoint["labels"])).eval()
            model.load_state_dict(checkpoint["model_state"], strict=True)
            self.torch, self.model = torch, model
            self.labels, self.mean, self.scale = checkpoint["labels"], checkpoint["mean"], checkpoint["scale"]
            self.model_source = "local_training" if source == MODEL else "two_word_probe"
            self.prefix_augmented = checkpoint.get("prefix_augmented", False)
            self.missing_hand_augmented = checkpoint.get("missing_hand_augmented", False)
            self.excluded_labels = checkpoint.get("excluded_labels", [])
            self.excluded_recordings = checkpoint.get("excluded_recordings", 0)
            self.original_recordings = checkpoint.get("original_recordings", 0)
            self.sliding_window_augmented = checkpoint.get("sliding_window_augmented", False)
        except RuntimeError:
            raise
        except Exception:
            raise RuntimeError("TCN runtime unavailable; start with the evaluation Python") from None

    def status(self):
        with self.lock:
            try:
                self.load()
            except RuntimeError as error:
                return {"ready": False, "error": self.error or str(error), "job": self.job,
                        "training_counts": self.training_counts}
            return {"ready": True, "labels": self.labels, "job": self.job, "error": self.error,
                    "model_sha256": self.model_sha256,
                    "model_source": self.model_source, "experimental": True, "input_dim": 130,
                    "feature_schema": "shoulder-hands-130-v1", "missing_hand_augmented": self.missing_hand_augmented,
                    "excluded_labels": self.excluded_labels, "excluded_recordings": self.excluded_recordings,
                    "prefix_augmented": self.prefix_augmented,
                    "original_recordings": self.original_recordings,
                    "sliding_window_augmented": self.sliding_window_augmented,
                    "training_counts": self.training_counts,
                    "trained_on_continuous_signs": False}

    def predict(self, sequence):
        from web_server import valid_samples
        if not valid_samples([{"label": "input", "sequence": sequence}]) or any(len(frame) != 130 for frame in sequence):
            raise ValueError("Expected 6-32 finite 130-dimensional shoulder-relative frames")
        with self.lock:
            self.load()
            torch = self.torch
            with torch.inference_mode():
                frames = torch.tensor(sequence, dtype=torch.float32)
                x = ((frames - self.mean) / self.scale).T.unsqueeze(0)
                scores = self.model(x, torch.tensor([len(sequence)])).softmax(-1)[0].tolist()
            return {"scores": [{"label": label, "score": score} for label, score in zip(self.labels, scores)], "calibrated": False}

    def start_training(self, samples):
        motion, excluded = position_recordings(samples)
        counts = {}
        for sample in motion:
            counts[sample["label"]] = counts.get(sample["label"], 0) + 1
        if len(counts) < 2 or min(counts.values()) < 5:
            raise ValueError(f"각 단어에 130차원 동작 5개 이상, 최소 2단어가 필요합니다. 사용 가능: {counts}; 제외: {len(excluded)}개 (126차원은 재수집 필요)")
        with self.lock:
            if self.job == "training":
                raise ValueError("Training already running")
            self.job, self.error = "training", None
            self.training_counts = counts
        threading.Thread(target=self.train, args=(samples,), daemon=True).start()
        return {"started": True, "recordings": len(motion), "excluded_recordings": len(excluded)}

    def train(self, samples):
        try:
            samples, excluded = position_recordings(samples)
            torch, _, read_motion, fit = self.dependencies()
            MODEL.parent.mkdir(parents=True, exist_ok=True)
            snapshot = MODEL.parent / "position-tcn-training-source.json"
            snapshot.write_text(json.dumps({"samples": samples}, ensure_ascii=False), encoding="utf-8")
            sequences, targets, labels, digest = read_motion(snapshot)
            from hand_tcn import augment_motion
            augmented, augmented_targets = augment_motion(sequences, targets)
            model, mean, scale, _ = fit(augmented, augmented_targets, labels,
                                       list(range(len(augmented))), 321, 100)
            temporary = MODEL.with_suffix(".tmp")
            torch.save({"model_state": model.state_dict(), "labels": labels, "mean": torch.tensor(mean),
                        "scale": torch.tensor(scale), "source_sha256": digest, "epochs": 100,
                        "input_dim": 130, "feature_schema": "shoulder-hands-130-v1",
                        "missing_hand_augmented": True, "prefix_augmented": True, "original_recordings": len(sequences),
                        "sliding_window_augmented": True,
                        "excluded_recordings": len(excluded), "excluded_labels": sorted({sample["label"] for sample in excluded}),
                        "augmented_clips": len(augmented)}, temporary)
            if MODEL.exists():
                shutil.copy2(MODEL, MODEL.with_suffix(".pt.bak"))
            os.replace(temporary, MODEL)
            with self.lock:
                self.model = None
                self.load()
                self.job = "done"
        except Exception as error:
            with self.lock:
                self.job, self.error = "failed", str(error)


runtime = TcnRuntime()
