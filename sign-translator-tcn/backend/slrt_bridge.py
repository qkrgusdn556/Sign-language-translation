import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
import project_paths
"""Share new recordings with the independent MediaPipe/RGB experiment, locally only."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import threading
from datetime import datetime, timezone
import uuid
import sys

ROOT = project_paths.ROOT
LEGACY_PROJECT = ROOT.parents[3] / '2026-10-18/slrt-mediapipe-ksl/outputs/slrt-mediapipe-ksl' if len(ROOT.parents) > 3 else ROOT / 'slrt-mediapipe-ksl'
PROJECT = Path(os.environ.get('SLRT_PROJECT_PATH') or (LEGACY_PROJECT if LEGACY_PROJECT.is_dir() else ROOT / 'slrt-mediapipe-ksl'))
LEGACY_PYTHON = Path.home() / 'AppData/Local/Programs/Python/Python312/python.exe'
PYTHON = Path(os.environ.get('SLRT_PYTHON') or (str(LEGACY_PYTHON) if PROJECT == LEGACY_PROJECT and LEGACY_PYTHON.is_file() else sys.executable))
LOCK = threading.RLock()


def atomic_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, allow_nan=False), encoding='utf-8')
    os.replace(temporary, path)


class SlrtBridge:
    def __init__(self, project=PROJECT, data_root=None, training_enabled=True):
        self.project = Path(project).resolve()
        self.data = Path(data_root).resolve() if data_root else self.project / 'data'
        self.training_enabled = training_enabled and self.data == self.project / 'data'
        self.process = None
        self.process_lock = threading.RLock()
        self.output = None

    def contract(self):
        source = self.project / 'contract.py'
        if not source.is_file():
            raise ValueError('SLRT MediaPipe experiment is not installed')
        spec = importlib.util.spec_from_file_location('shared_slrt_contract', source)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    def records(self):
        return [json.loads(path.read_text(encoding='utf-8')) for path in sorted(self.data.glob('clips/*/metadata.json'))]

    def share(self, meta, source):
        from capture_store import validate_metadata, ID
        validate_metadata(meta)
        contract = self.contract()
        # The native classifier only learns whole isolated clips and background clips.
        if meta['kind'] == 'continuous':
            raise ValueError('Continuous clips need segment extraction before SLRT training')
        camera = meta['camera']
        frames = []
        for frame in meta['frames']:
            if frames and frame['media_time_ms'] <= frames[-1]['media_time_ms']:
                continue
            frames.append({'index': len(frames), 'media_time_ms': frame['media_time_ms'], 'capture_elapsed_ms': frame['elapsed_ms']})
        value = {'schema_version': 1, 'session_id': meta['session_id'], 'split': meta['split'],
                 'kind': meta['kind'], 'sequence': meta['sequence'], 'segments': meta.get('segments', []),
                 'consent': True, 'audio': False, 'started_at_utc': meta['started_at_utc'],
                 'duration_ms': meta['duration_ms'], 'camera': camera, 'mime_type': meta['mime_type'],
                 'preview_frames': frames, 'source_project': 'sign-translator-tcn',
                 'source_clip_id': meta['clip_id'], 'source_context': meta['context'],
                 'source_client_model': meta.get('client_model', {}),
                 'source_synchronization': meta['synchronization']}
        contract.validate(value)
        source = Path(source).resolve()
        digest = hashlib.sha256(source.read_bytes()).hexdigest()
        if digest != meta['video_sha256']:
            raise ValueError('Source recording fingerprint mismatch')
        identity = uuid.uuid5(uuid.NAMESPACE_URL, 'sign-translator-tcn/' + meta['clip_id']).hex
        with LOCK:
            existing = self.records()
            contract.validate_session_splits(existing + [value])
            for row in existing:
                if not ID.fullmatch(str(row.get('clip_id', ''))):
                    raise ValueError('Invalid existing SLRT clip identity')
                if row['clip_id'] != identity and row.get('video_sha256') != digest:
                    continue
                if row.get('video_sha256') != digest or any(row.get(key) != value[key] for key in ('session_id', 'split', 'kind', 'sequence')):
                    raise ValueError('Existing SLRT clip conflicts with this recording or split')
                destination = self.data / 'clips' / row['clip_id'] / 'rgb.webm'
                if not destination.is_file() or hashlib.sha256(destination.read_bytes()).hexdigest() != digest:
                    raise ValueError('Existing SLRT video fingerprint mismatch')
                return {'status': 'already_shared', 'clip_id': row['clip_id'], 'split': row['split']}
            value.update({'clip_id': identity, 'video_file': 'rgb.webm', 'video_sha256': digest,
                          'video_bytes': source.stat().st_size, 'saved_at_utc': datetime.now(timezone.utc).isoformat(),
                          'video_validation': 'header_only_decode_pending',
                          'landmarks': {'status': 'not_extracted', 'pose': None, 'hands': None, 'face_mouth': None},
                          'synchronization': {'status': 'preview_only_decode_mapping_pending', 'encoded_frame_mapping': None},
                          'annotation_status': 'timed' if value['segments'] else 'sequence_only'})
            temporary = self.data / 'pending' / identity
            destination = self.data / 'clips' / identity
            if temporary.exists() or destination.exists():
                raise ValueError('Incomplete existing SLRT import requires inspection; refusing overwrite')
            temporary.mkdir(parents=True)
            try:
                shutil.copyfile(source, temporary / 'rgb.webm')
                atomic_json(temporary / 'metadata.json', value)
                destination.parent.mkdir(parents=True, exist_ok=True)
                os.replace(temporary, destination)
            except Exception:
                if temporary.exists() and temporary.resolve().parent == (self.data / 'pending').resolve():
                    shutil.rmtree(temporary)
                raise
        return {'status': 'shared', 'clip_id': identity, 'split': value['split']}

    def status(self):
        with LOCK:
            records = self.records()
            self.contract().validate_session_splits(records)
            train = [row for row in records if row['split'] == 'train' and row['kind'] != 'continuous']
            counts = {}
            for row in train:
                label = row['sequence'][0] if row['kind'] == 'isolated' else '배경 동작'
                counts[label] = counts.get(label, 0) + 1
            pointer = self.project / 'models/current.json'
            report = {}
            if pointer.is_file():
                identity = json.loads(pointer.read_text(encoding='utf-8')).get('model_id')
                if not isinstance(identity, str) or Path(identity).name != identity or identity in {'.', '..'}:
                    raise ValueError('Invalid SLRT model identity')
                report_path = self.project / 'models' / identity / 'report.json'
                if report_path.is_file():
                    report = json.loads(report_path.read_text(encoding='utf-8'))
            previous = {row['clip_id']: row for row in report.get('training_manifest', [])}
            keys = ('clip_id', 'video_sha256', 'session_id', 'split', 'kind', 'sequence')
            pending = sum(previous.get(row['clip_id']) != {key: row[key] for key in keys} for row in train)
            job_path = self.project / 'models/training-job.json'
            job = json.loads(job_path.read_text(encoding='utf-8')) if job_path.is_file() else {'status': 'idle'}
            if self.process is not None:
                code = self.process.poll()
                if code is None and job.get('status') not in {'starting', 'running'}:
                    job = {'status': 'starting', 'message': 'SLRT native training process is preparing'}
                elif code not in (None, 0) and job.get('status') not in {'failed', 'timed_out', 'interrupted'}:
                    job = {'status': 'failed', 'message': 'SLRT training process failed; check data/slrt-shared-training.log'}
            return {'available': True, 'counts': counts, 'shared_clips': sum(row.get('source_project') == 'sign-translator-tcn' for row in records),
                    'saved_training_clips': len(train), 'pending_training_clips': pending, 'model_id': report.get('model_id'),
                    'training_allowed': self.training_enabled, 'job': job,
                    'can_train': self.training_enabled and len(counts) >= 2 and counts.get('배경 동작', 0) >= 2 and min(counts.values(), default=0) >= 2}

    def start_training(self):
        with self.process_lock:
            if not self.training_enabled:
                raise ValueError('SLRT training is disabled for this isolated storage')
            if self.process is not None and self.process.poll() is None:
                raise ValueError('SLRT training is already running')
            state = self.status()
            if state['job'].get('status') in {'starting', 'running'}:
                raise ValueError('SLRT project reports an existing training job; check that project first')
            if not state['can_train']:
                raise ValueError('SLRT needs at least two training clips per word and two background clips')
            if not PYTHON.is_file():
                raise ValueError('SLRT Python runtime not found')
            # Use the native project's environment, locks, time limit and model versioning.
            code = 'from pathlib import Path; from training_job import TrainingJob; j=TrainingJob(Path.cwd()); j.start(); j.worker.join(); s=j.snapshot(); print(s,flush=True); raise SystemExit(0 if s["status"]=="succeeded" else 1)'
            log = ROOT / 'data/slrt-shared-training.log'
            self.output = log.open('w', encoding='utf-8')
            try:
                options = {'creationflags': subprocess.CREATE_NO_WINDOW} if os.name == 'nt' else {}
                self.process = subprocess.Popen([str(PYTHON), '-X', 'utf8', '-u', '-c', code], cwd=self.project,
                                                stdout=self.output, stderr=subprocess.STDOUT, **options)
            finally:
                self.output.close()
                self.output = None
            return {'started': True, 'message': 'SLRT video model training requested; not continuous SLRT integration'}


bridge = SlrtBridge()
