import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
import project_paths
"""Local-only image collection and isolated YOLO classification jobs."""
import argparse
import base64
import hashlib
import io
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import threading
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from PIL import Image, ImageOps

ROOT = project_paths.ROOT
ID = re.compile(r'[a-f0-9]{32}')


def write_json(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False), encoding='utf-8')
    os.replace(temporary, path)


class Studio:
    def __init__(self, root):
        self.root = root
        root.mkdir(parents=True, exist_ok=True)
        self.lock = threading.RLock()
        self.process = None
        self.job = {'state': 'idle'}
        self.rows = json.loads((root / 'images.json').read_text(encoding='utf-8')) if (root / 'images.json').exists() else []

    def status(self):
        with self.lock:
            job = dict(self.job)
            if job.get('run'):
                progress = self.root / 'runs' / job['run'] / 'progress.json'
                if progress.exists():
                    job['progress'] = json.loads(progress.read_text(encoding='utf-8'))
            runs = []
            for path in sorted((self.root / 'runs').glob('*/result.json'), key=lambda p: p.stat().st_mtime, reverse=True):
                runs.append({'id': path.parent.name, **json.loads(path.read_text(encoding='utf-8'))})
            return {'images': self.rows[:], 'job': job, 'runs': runs, 'model': 'YOLO26n', 'baseModel': (ROOT / 'data/yolo26n-cls.pt').exists(),
                    'detectionModel': (ROOT / 'data/yolo26n.pt').exists()}

    def add(self, value):
        if not isinstance(value.get('label'), str) or not isinstance(value.get('image'), str):
            raise ValueError('이름과 사진이 필요합니다.')
        label = value['label'].strip()
        split = value.get('split')
        if not label or len(label) > 40 or any(ord(c) < 32 for c in label) or split not in ('train', 'val'):
            raise ValueError('이름(1~40자)과 학습용/검증용을 선택해 주세요.')
        raw = base64.b64decode(value['image'], validate=True)
        if len(raw) > 8 * 1024 * 1024:
            raise ValueError('사진은 8MB 이하여야 합니다.')
        with Image.open(io.BytesIO(raw)) as source:
            if source.width * source.height > 16000000:
                raise ValueError('사진은 1600만 화소 이하여야 합니다.')
            image = ImageOps.exif_transpose(source).convert('RGB')
        digest = hashlib.sha256(str(image.size).encode() + image.tobytes()).hexdigest()
        with self.lock:
            if len(self.rows) >= 10000:
                raise ValueError('사진 저장 한도(10000장)에 도달했습니다.')
            if any(row['hash'] == digest for row in self.rows):
                raise ValueError('같은 사진이 이미 저장되어 있습니다. 검증 사진은 따로 촬영해 주세요.')
            ident = uuid.uuid4().hex
            (self.root / 'images').mkdir(exist_ok=True)
            image.save(self.root / 'images' / f'{ident}.jpg', quality=95)
            row = {'id': ident, 'label': label, 'split': split, 'hash': digest}
            self.rows.append(row)
            write_json(self.root / 'images.json', self.rows)
            return row

    def remove(self, ident):
        if not isinstance(ident, str) or not ID.fullmatch(ident):
            raise ValueError('Invalid image ID')
        with self.lock:
            self.rows = [row for row in self.rows if row['id'] != ident]
            write_json(self.root / 'images.json', self.rows)
            (self.root / 'images' / f'{ident}.jpg').unlink(missing_ok=True)

    def annotate(self, value):
        ident, boxes = value.get('id'), value.get('boxes')
        if not isinstance(ident, str) or not ID.fullmatch(ident) or not isinstance(boxes, list) or len(boxes) > 20:
            raise ValueError('Invalid annotation')
        clean = []
        for box in boxes:
            if not isinstance(box, dict):
                raise ValueError('Invalid box')
            label = box.get('label')
            coords = [box.get(k) for k in ('x', 'y', 'w', 'h')]
            if not isinstance(label, str) or not 1 <= len(label.strip()) <= 40 or any(ord(c) < 32 for c in label):
                raise ValueError('박스 이름을 입력해 주세요.')
            if any(type(n) not in (int, float) or not math.isfinite(n) for n in coords):
                raise ValueError('Invalid coordinates')
            x, y, w, h = coords
            if x < 0 or y < 0 or w < .005 or h < .005 or x + w > 1.000001 or y + h > 1.000001:
                raise ValueError('박스는 사진 안에 있어야 합니다.')
            clean.append(dict(zip(('x', 'y', 'w', 'h'), coords), label=label.strip()))
        if not clean and value.get('background') is not True:
            raise ValueError('박스를 그리거나 대상 없음으로 지정해 주세요.')
        with self.lock:
            row = next((r for r in self.rows if r['id'] == ident), None)
            if row is None:
                raise ValueError('사진이 삭제되었습니다.')
            updated = {**row, 'boxes': clean, 'annotated': True}
            rows = [updated if r['id'] == ident else r for r in self.rows]
            write_json(self.root / 'images.json', rows)
            self.rows = rows
            return updated

    def remove_many(self, ids):
        if not isinstance(ids, list) or not ids or len(ids) > 10000 or any(not isinstance(i, str) or not ID.fullmatch(i) for i in ids):
            raise ValueError('삭제할 사진을 선택해 주세요. (최대 10000장)')
        selected = set(ids)
        with self.lock:
            rows = [row for row in self.rows if row['id'] in selected]
            if len(rows) != len(selected):
                raise ValueError('사진 목록이 변경되었습니다. 새로고침 후 다시 선택해 주세요.')
            # Retain a recoverable local copy; completed training runs are untouched.
            batch = self.root / 'deleted' / uuid.uuid4().hex
            batch.mkdir(parents=True)
            moved = []
            try:
                write_json(batch / 'images.json', rows)
                for row in rows:
                    source = self.root / 'images' / f'{row["id"]}.jpg'
                    if source.is_symlink() or source.resolve().parent != (self.root / 'images').resolve():
                        raise ValueError('Invalid image path')
                    destination = batch / source.name
                    source.rename(destination)
                    moved.append((source, destination))
                remaining = [row for row in self.rows if row['id'] not in selected]
                write_json(self.root / 'images.json', remaining)
            except Exception:
                for source, destination in reversed(moved):
                    destination.rename(source)
                raise
            self.rows = remaining
            return {'removed': len(rows)}

    def train(self, value):
        epochs = value.get('epochs')
        task = value.get('task', 'classify')
        if task not in ('classify', 'detect'):
            raise ValueError('Invalid task')
        if type(epochs) is not int or not 1 <= epochs <= 200:
            raise ValueError('학습 횟수는 1~200 사이여야 합니다.')
        with self.lock:
            if self.process is not None:
                raise ValueError('이미 학습 중입니다.')
            weights = 'yolo26n.pt' if task == 'detect' else 'yolo26n-cls.pt'
            if not (ROOT / 'data' / weights).is_file():
                raise ValueError(f'기본 모델 {weights} 설치가 필요합니다. 박스 작업은 저장할 수 있습니다.')
            rows = [r for r in self.rows if r.get('annotated')] if task == 'detect' else self.rows
            labels = sorted({b['label'] for r in rows for b in r['boxes']}) if task == 'detect' else sorted({row['label'] for row in rows})
            if len(labels) < (1 if task == 'detect' else 2):
                raise ValueError('저장된 손 박스가 필요합니다.' if task == 'detect' else '최소 두 종류가 필요합니다. 대기/기타 사진도 별도 이름으로 등록하세요.')
            for label in labels:
                counts = {split: sum(r['split'] == split and (any(b['label'] == label for b in r['boxes']) if task == 'detect' else r['label'] == label) for r in rows) for split in ('train', 'val')}
                if counts['train'] < 5 or counts['val'] < 2:
                    raise ValueError(f'{label}: 학습용 5장, 별도 촬영한 검증용 2장 이상 필요합니다.')
            run = uuid.uuid4().hex
            folder = self.root / 'runs' / run
            folder.mkdir(parents=True)
            write_json(folder / 'source.json', {'images': rows, 'labels': labels, 'epochs': epochs, 'task': task, 'model': weights})
            for row in rows:
                dest = folder / 'source' / row['split'] / ('images' if task == 'detect' else f'c{labels.index(row["label"]):04d}')
                dest.mkdir(parents=True, exist_ok=True)
                shutil.copy2(self.root / 'images' / f'{row["id"]}.jpg', dest / f'{row["id"]}.jpg')
            log = (folder / 'training.log').open('wb')
            try:
                self.process = subprocess.Popen([sys.executable, str(ROOT / 'training/yolo_studio_worker.py'), str(folder)], cwd=ROOT,
                    stdout=log, stderr=subprocess.STDOUT, creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0)
            finally:
                log.close()
            self.job = {'state': 'running', 'run': run, 'epochs': epochs, 'task': task}
            threading.Thread(target=self._wait, args=(self.process, folder), daemon=True).start()
            return dict(self.job)

    def _wait(self, process, folder):
        code = process.wait()
        with self.lock:
            cancelled = self.job['state'] == 'cancelling'
            self.job['state'] = 'cancelled' if cancelled else 'complete' if code == 0 and (folder / 'result.json').exists() else 'error'
            if self.job['state'] == 'error':
                self.job['error'] = (folder / 'training.log').read_text(encoding='utf-8', errors='replace')[-2400:]
            self.process = None

    def cancel(self):
        with self.lock:
            if self.process is not None:
                self.job['state'] = 'cancelling'
                self.process.terminate()


class Handler(BaseHTTPRequestHandler):
    def allowed(self):
        host = self.headers.get('Host')
        return host in (f'localhost:{self.server.server_port}', f'127.0.0.1:{self.server.server_port}') and self.headers.get('Origin') in (None, f'http://{host}')

    def send(self, code, body, mime='application/json; charset=utf-8'):
        if not isinstance(body, bytes):
            body = json.dumps(body, ensure_ascii=False).encode()
        self.send_response(code)
        self.send_header('Content-Type', mime)
        self.send_header('Content-Length', str(len(body)))
        self.send_header('Cache-Control', 'no-store')
        self.send_header('X-Content-Type-Options', 'nosniff')
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if not self.allowed():
            return self.send(403, {'error': 'Local access only'})
        route = self.path.split('?', 1)[0]
        if route == '/api/status':
            return self.send(200, self.server.studio.status())
        files = {'/': ('yolo-train.html', 'text/html; charset=utf-8'), '/yolo-train.html': ('yolo-train.html', 'text/html; charset=utf-8'),
                 '/yolo-train.js': ('yolo-train.js', 'text/javascript; charset=utf-8'), '/yolo-annotate.js': ('yolo-annotate.js', 'text/javascript; charset=utf-8'), '/yolo-train.css': ('yolo-train.css', 'text/css')}
        files['/yolo-annotate.css'] = ('yolo-annotate.css', 'text/css')
        files['/shoulder-guide.js'] = ('shoulder-guide.js', 'text/javascript; charset=utf-8')
        if route in files:
            name, mime = files[route]
            return self.send(200, (ROOT / 'frontend' / name).read_bytes(), mime)
        match = re.fullmatch(r'/api/images/([a-f0-9]{32})', route)
        if match:
            path = self.server.studio.root / 'images' / f'{match[1]}.jpg'
            if path.is_file():
                return self.send(200, path.read_bytes(), 'image/jpeg')
        match = re.fullmatch(r'/api/runs/([a-f0-9]{32})/(fingerspell.onnx|labels.json|training.log)', route)
        if match:
            folder = self.server.studio.root / 'runs' / match[1]
            path = folder / match[2]
            if path.is_file() and (match[2] == 'training.log' or (folder / 'result.json').exists()):
                return self.send(200, path.read_bytes(), 'application/octet-stream')
        self.send(404, {'error': 'Not found'})

    def do_POST(self):
        if not self.allowed():
            return self.send(403, {'error': 'Local access only'})
        try:
            length = int(self.headers.get('Content-Length', '0'))
            if not 0 < length <= 12 * 1024 * 1024:
                return self.send(413, {'error': '요청 크기가 너무 큽니다.'})
            value = json.loads(self.rfile.read(length))
            if not isinstance(value, dict):
                raise ValueError('Invalid request')
            studio = self.server.studio
            if self.path == '/api/images':
                return self.send(201, studio.add(value))
            if self.path == '/api/annotate':
                return self.send(200, studio.annotate(value))
            if self.path == '/api/remove-many':
                return self.send(200, studio.remove_many(value.get('ids')))
            if self.path == '/api/remove':
                studio.remove(value['id'])
            elif self.path == '/api/train':
                return self.send(202, studio.train(value))
            elif self.path == '/api/cancel':
                studio.cancel()
            else:
                return self.send(404, {'error': 'Not found'})
            self.send(200, {'ok': True})
        except (ValueError, KeyError, TypeError, OSError, Image.DecompressionBombError) as error:
            self.send(400, {'error': str(error)})


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=5196)
    parser.add_argument('--data-root', type=Path, default=ROOT / 'data/yolo-studio')
    args = parser.parse_args()
    from web_server import LocalServer
    server = LocalServer(('127.0.0.1', args.port), Handler)
    server.studio = Studio(args.data_root.resolve())
    print(f'http://127.0.0.1:{args.port}', flush=True)
    server.serve_forever()
