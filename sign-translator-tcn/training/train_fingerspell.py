import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
import project_paths
"""Small, local YOLO11-cls smoke test; never touches TCN samples or models."""
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT = project_paths.ROOT
(ROOT / "data" / "yolo-config").mkdir(parents=True, exist_ok=True)
os.environ["YOLO_CONFIG_DIR"] = str(ROOT / "data" / "yolo-config")
os.environ["YOLO_AUTOINSTALL"] = "false"
sys.path.insert(0, str(ROOT / "data" / "yolo-runtime"))
if os.environ.get('SIGN_YOLO26') == '1':
    sys.path.insert(0, str(ROOT / 'data' / 'yolo26-runtime'))

import numpy as np
from PIL import Image
import torch
import torchvision.transforms as T
from ultralytics import YOLO, settings
from ultralytics.models.yolo.classify import ClassificationTrainer


class FullFrameTrainer(ClassificationTrainer):
    def build_dataset(self, img_path, mode="train", batch=None):
        dataset = super().build_dataset(img_path, mode, batch)
        transforms = [T.Resize((224, 224))]
        if mode == "train":
            transforms.append(T.ColorJitter(brightness=.15, contrast=.15, saturation=.1))
        transforms.append(T.ToTensor())
        dataset.torch_transforms = T.Compose(transforms)
        return dataset


def letterbox(image):
    image = image.convert("RGB")
    scale = min(224 / image.width, 224 / image.height)
    size = (round(image.width * scale), round(image.height * scale))
    canvas = Image.new("RGB", (224, 224), (114, 114, 114))
    canvas.paste(image.resize(size, Image.Resampling.BILINEAR), ((224-size[0])//2, (224-size[1])//2))
    return canvas


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--run-name", default="yolo-g-smoke")
    args = parser.parse_args()
    settings.update({"sync": False, "wandb": False, "mlflow": False, "tensorboard": False})
    torch.set_num_threads(2)
    source = ROOT / "data" / "fingerspell-images"
    if not args.run_name or any(char not in "abcdefghijklmnopqrstuvwxyz0123456789-" for char in args.run_name):
        raise ValueError("Run name must contain only lowercase letters, digits and hyphens")
    output = ROOT / "data" / args.run_name
    if output.exists():
        raise RuntimeError("Run directory already exists; preserve it and select a new run directory before retraining")
    prepared = output / "prepared"
    hashes = {"train": set(), "val": set()}
    counts = {"train": {}, "val": {}}
    for split in hashes:
        for folder in sorted((source / split).iterdir()):
            if not folder.is_dir():
                continue
            files = sorted(folder.glob("*.jpg"))
            counts[split][folder.name] = len(files)
            destination = prepared / split / folder.name
            destination.mkdir(parents=True, exist_ok=True)
            for path in files:
                hashes[split].add(hashlib.sha256(path.read_bytes()).hexdigest())
                with Image.open(path) as image:
                    letterbox(image).save(destination / (path.stem + ".png"))
    if hashes["train"] & hashes["val"]:
        raise RuntimeError("Training and validation images overlap")
    print("DATASET", json.dumps(counts, ensure_ascii=True), flush=True)
    model = YOLO(str(ROOT / "data" / "yolo11n-cls.pt"))
    model.train(data=str(prepared), trainer=FullFrameTrainer, epochs=args.epochs, imgsz=224,
                batch=16, workers=0, device=0 if torch.cuda.is_available() else "cpu",
                project=str(output), name="training", exist_ok=False, seed=321,
                optimizer="AdamW", lr0=.001, patience=10, plots=False,
                amp=False, deterministic=True, fliplr=0, flipud=0, erasing=0)
    trained = YOLO(str(output / "training" / "weights" / "best.pt"))
    exported = Path(trained.export(format="onnx", imgsz=224, batch=1, opset=17,
                                  simplify=False, dynamic=False, device="cpu"))
    import shutil
    onnx_path = output / "fingerspell.onnx"
    shutil.copy2(exported, onnx_path)
    labels = [trained.names[index] for index in range(len(trained.names))]
    (output / "labels.json").write_text(json.dumps(labels, ensure_ascii=False), encoding="utf-8")
    import onnxruntime as ort
    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    report = {"classes": labels, "counts": counts, "validation": {}, "onnx_max_difference": 0,
              "warning": "Same-person, same-room small smoke test; not independent camera accuracy."}
    trained.model.eval()
    for index, label in enumerate(labels):
        rows = []
        for path in sorted((prepared / "val" / label).glob("*.png")):
            with Image.open(path) as image:
                x = np.asarray(image.convert("RGB"), dtype=np.float32).transpose(2, 0, 1)[None] / 255
            scores = session.run(None, {session.get_inputs()[0].name: x})[0][0]
            with torch.inference_mode():
                native = trained.model(torch.from_numpy(x))
                native = native[0] if isinstance(native, tuple) else native
                difference = float(np.max(np.abs(scores - native[0].numpy())))
            report["onnx_max_difference"] = max(report["onnx_max_difference"], difference)
            rows.append({"file": path.name, "predicted": labels[int(scores.argmax())],
                         "score": float(scores.max()), "correct": int(scores.argmax()) == index})
        report["validation"][label] = rows
    (output / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print("SMOKE_RESULT", json.dumps({label: {"correct": sum(row["correct"] for row in rows), "total": len(rows)}
                                     for label, rows in report["validation"].items()}, ensure_ascii=True), flush=True)
    print("ONNX", onnx_path, flush=True)


if __name__ == "__main__":
    main()
