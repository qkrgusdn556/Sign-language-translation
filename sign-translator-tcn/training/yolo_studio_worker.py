import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
import project_paths
"""Train only an immutable studio snapshot; reuse the runtime's full-frame transform."""
import json
from pathlib import Path
import shutil
import sys
import os

os.environ['SIGN_YOLO26'] = '1'

from train_fingerspell import FullFrameTrainer, ROOT, YOLO, letterbox, settings, torch
from PIL import Image
from yolo_studio import write_json


def main(folder):
    source = json.loads((folder / 'source.json').read_text(encoding='utf-8'))
    settings.update({'sync': False, 'wandb': False, 'mlflow': False, 'tensorboard': False})
    torch.set_num_threads(2)
    if source.get('task') == 'detect':
        return train_detection(folder, source)
    weights = ROOT / 'data/yolo26n-cls.pt'
    if not weights.is_file():
        raise ValueError('기본 모델 data/yolo26n-cls.pt가 없습니다.')
    for split in ('train', 'val'):
        for path in (folder / 'source' / split).glob('*/*.jpg'):
            dest = folder / 'prepared' / split / path.parent.name / (path.stem + '.png')
            dest.parent.mkdir(parents=True, exist_ok=True)
            with Image.open(path) as image:
                letterbox(image).save(dest)
    model = YOLO(str(weights))

    def progress(trainer):
        write_json(folder / 'progress.json', {'epoch': trainer.epoch + 1, 'epochs': source['epochs'], 'phase': 'training'})

    model.add_callback('on_fit_epoch_end', progress)
    model.train(data=str(folder / 'prepared'), trainer=FullFrameTrainer, epochs=source['epochs'], imgsz=224,
                batch=8, workers=0, device=0 if torch.cuda.is_available() else 'cpu',
                project=str(folder), name='training', exist_ok=False, seed=321,
                optimizer='AdamW', lr0=.001, patience=10, plots=False, amp=False,
                deterministic=True, fliplr=0, flipud=0, erasing=0)
    write_json(folder / 'progress.json', {'phase': 'exporting'})
    trained = YOLO(str(folder / 'training/weights/best.pt'))
    exported = Path(trained.export(format='onnx', imgsz=224, batch=1, opset=17, simplify=False, dynamic=False, device='cpu'))
    shutil.copy2(exported, folder / 'fingerspell.onnx')
    labels = [source['labels'][int(trained.names[i][1:])] for i in range(len(trained.names))]
    write_json(folder / 'labels.json', {'task': 'classify', 'names': labels, 'model': 'YOLO26n-cls'})
    # Evaluate the exported artifact with exactly the browser's RGB / 255 input.
    import numpy as np
    import onnxruntime as ort
    session = ort.InferenceSession(str(folder / 'fingerspell.onnx'), providers=['CPUExecutionProvider'])
    confusion = [[0] * len(labels) for _ in labels]
    for path in (folder / 'prepared/val').glob('*/*.png'):
        truth = labels.index(source['labels'][int(path.parent.name[1:])])
        with Image.open(path) as image:
            tensor = np.asarray(image.convert('RGB'), dtype=np.float32).transpose(2, 0, 1)[None] / 255
        scores = session.run(None, {session.get_inputs()[0].name: tensor})[0][0]
        confusion[truth][int(scores.argmax())] += 1
    total = sum(map(sum, confusion))
    write_json(folder / 'result.json', {'model': 'YOLO26n-cls', 'labels': labels, 'confusion': confusion, 'total': total,
               'correct': sum(confusion[i][i] for i in range(len(labels)))})


def train_detection(folder, source):
    import yaml
    weights = ROOT / 'data/yolo26n.pt'
    if not weights.is_file():
        raise ValueError('탐지 기본 모델 data/yolo26n.pt가 없습니다.')
    prepared = folder / 'prepared'
    for row in source['images']:
        image_dir = prepared / 'images' / row['split']
        label_dir = prepared / 'labels' / row['split']
        image_dir.mkdir(parents=True, exist_ok=True)
        label_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(folder / 'source' / row['split'] / 'images' / f'{row["id"]}.jpg', image_dir / f'{row["id"]}.jpg')
        lines = []
        for box in row['boxes']:
            lines.append(f"{source['labels'].index(box['label'])} {box['x'] + box['w']/2:.8f} {box['y'] + box['h']/2:.8f} {box['w']:.8f} {box['h']:.8f}")
        (label_dir / f'{row["id"]}.txt').write_text('\n'.join(lines), encoding='utf-8')
    config = {'path': str(prepared), 'train': 'images/train', 'val': 'images/val',
              'names': {i: f'c{i:04d}' for i in range(len(source['labels']))}}
    (folder / 'dataset.yaml').write_text(yaml.safe_dump(config), encoding='utf-8')
    model = YOLO(str(weights))
    model.add_callback('on_fit_epoch_end', lambda trainer: write_json(folder / 'progress.json',
                       {'epoch': trainer.epoch + 1, 'epochs': source['epochs'], 'phase': 'training'}))
    model.train(data=str(folder / 'dataset.yaml'), epochs=source['epochs'], imgsz=320, batch=8, workers=0,
                device=0 if torch.cuda.is_available() else 'cpu', project=str(folder), name='training',
                exist_ok=False, seed=321, patience=10, plots=False, amp=False, deterministic=True,
                fliplr=0, flipud=0, degrees=0, mosaic=0, mixup=0)
    write_json(folder / 'progress.json', {'phase': 'exporting'})
    trained = YOLO(str(folder / 'training/weights/best.pt'))
    exported = Path(trained.export(format='onnx', imgsz=320, batch=1, opset=17, simplify=False, dynamic=False, nms=False, device='cpu'))
    shutil.copy2(exported, folder / 'fingerspell.onnx')
    labels = [source['labels'][int(trained.names[i][1:])] for i in range(len(trained.names))]
    write_json(folder / 'labels.json', {'task': 'detect', 'imgsz': 320, 'names': labels, 'format': 'yolo26-end2end', 'model': 'YOLO26n'})
    # Validate the exported artifact, not only the training checkpoint.
    metrics = YOLO(str(folder / 'fingerspell.onnx'), task='detect').val(data=str(folder / 'dataset.yaml'), imgsz=320,
                     batch=1, workers=0, device='cpu', plots=False, project=str(folder), name='onnx-validation')
    write_json(folder / 'result.json', {'model': 'YOLO26n', 'task': 'detect', 'labels': labels, 'map50': float(metrics.box.map50),
               'map50_95': float(metrics.box.map), 'total': sum(r['split'] == 'val' for r in source['images'])})


if __name__ == '__main__':
    main(Path(sys.argv[1]).resolve())
