import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
import project_paths
"""Local, append-only raw recordings. Legacy training samples are never modified."""
import base64
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re
import shutil
import threading
import uuid

ROOT = project_paths.ROOT / 'data' / 'recordings'
LOCK = threading.RLock()
ID = re.compile(r'^[a-f0-9]{32}$')
SESSION = re.compile(r'^[A-Za-z0-9_-]{1,80}$')
KINDS = {'isolated', 'continuous', 'rest', 'non_sign', 'transition'}
MAX_VIDEO = 32 * 1024 * 1024


def finite(value, low, high):
    return type(value) in (int, float) and math.isfinite(value) and low <= value <= high


def word(value):
    return isinstance(value, str) and 0 < len(value.strip()) <= 100 and all(ord(c) >= 32 for c in value)


def validate_metadata(meta):
    if not isinstance(meta, dict) or meta.get('schema_version') != 1:
        raise ValueError('Unsupported recording metadata')
    json.dumps(meta, allow_nan=False)
    if not ID.fullmatch(str(meta.get('request_id', ''))) or not SESSION.fullmatch(str(meta.get('session_id', ''))):
        raise ValueError('Invalid recording/session ID')
    if meta.get('consent') is not True or meta.get('audio') is not False:
        raise ValueError('Explicit recording consent and audio:false required')
    if meta.get('split') not in {'train', 'validation', 'test'} or meta.get('kind') not in KINDS:
        raise ValueError('Invalid recording split/kind')
    if not word(meta.get('pipeline')) or not isinstance(meta.get('context'), dict):
        raise ValueError('Recognition pipeline/context required')
    duration = meta.get('duration_ms')
    if not finite(duration, 300, 32000):
        raise ValueError('Recording must be 0.3-32 seconds')
    sequence = meta.get('sequence')
    if not isinstance(sequence, list) or len(sequence) > 40 or not all(word(item) for item in sequence):
        raise ValueError('Invalid expected sign sequence')
    if meta['kind'] == 'isolated' and len(sequence) != 1 or meta['kind'] == 'continuous' and not sequence:
        raise ValueError('Expected sign labels are required')
    if meta['kind'] in {'rest', 'non_sign', 'transition'} and sequence:
        raise ValueError('Background recordings must have no expected output')
    camera = meta.get('camera', {})
    if not isinstance(camera, dict) or camera.get('mirrored') is not False:
        raise ValueError('Raw, unmirrored camera recording required')
    if meta.get('mime_type') not in {'video/webm', 'video/webm;codecs=vp8', 'video/webm;codecs=vp9'}:
        raise ValueError('Unsupported video format')
    frames = meta.get('frames')
    if not isinstance(frames, list) or len(frames) > 1000:
        raise ValueError('Invalid observation frames')
    previous = -1
    for frame in frames:
        if not isinstance(frame, dict) or not finite(frame.get('elapsed_ms'), 0, duration + 200) or frame['elapsed_ms'] < previous:
            raise ValueError('Invalid observation time')
        previous = frame['elapsed_ms']
        if not finite(frame.get('media_time_ms'), 0, 1e12):
            raise ValueError('Invalid camera media time')
        hands = frame.get('hands')
        if not isinstance(hands, list) or len(hands) > 2:
            raise ValueError('Invalid hands')
        for points in [*hands, *([frame['pose']] if frame.get('pose') is not None else [])]:
            if not isinstance(points, list) or len(points) not in (21, 33):
                raise ValueError('Invalid observed landmarks')
            for point in points:
                if not isinstance(point, dict) or not all(finite(point.get(axis), -100, 100) for axis in ('x', 'y', 'z')):
                    raise ValueError('Invalid landmark coordinate')
        if frame.get('pose_status') not in {'detected', 'missing', 'not_run'}:
            raise ValueError('Invalid pose observation status')
    predictions = meta.get('predictions')
    if not isinstance(predictions, list) or len(predictions) > 500:
        raise ValueError('Invalid prediction events')
    previous = -1
    for event in predictions:
        if not isinstance(event, dict) or not word(event.get('label')) or not finite(event.get('at_ms'), 0, duration + 200) or event['at_ms'] < previous:
            raise ValueError('Invalid confirmed prediction')
        previous = event['at_ms']
    segments = meta.get('segments', [])
    if not isinstance(segments, list) or len(segments) > 40 or segments and len(segments) != len(sequence):
        raise ValueError('Timed labels must match the expected sequence')
    previous = 0
    for segment, label in zip(segments, sequence):
        if not isinstance(segment, dict) or segment.get('label') != label or not finite(segment.get('start_ms'), previous, duration) or not finite(segment.get('end_ms'), 0, duration) or segment['end_ms'] <= segment['start_ms']:
            raise ValueError('Invalid timed sign label')
        previous = segment['end_ms']


def list_metadata(root=None):
    root = Path(root or ROOT)
    records = []
    with LOCK:
        for path in sorted(root.glob('clips/*/metadata.json')):
            records.append(json.loads(path.read_text(encoding='utf-8')))
    return sorted(records, key=lambda meta: (meta.get('saved_at_utc', ''), meta['clip_id']))


def save_recording(value, root=None, context_end=None):
    root = Path(root or ROOT)
    if not isinstance(value, dict):
        raise ValueError('Expected recording object')
    meta = value.get('metadata')
    validate_metadata(meta)
    encoded = value.get('video_base64')
    if not isinstance(encoded, str) or len(encoded) > (MAX_VIDEO * 4 // 3 + 8):
        raise ValueError('Recording exceeds 32MB')
    try:
        video = base64.b64decode(encoded, validate=True)
    except (ValueError, TypeError) as error:
        raise ValueError('Invalid video encoding') from error
    if not 32 <= len(video) <= MAX_VIDEO or not video.startswith(b'\x1a\x45\xdf\xa3'):
        raise ValueError('Invalid WebM header/size')
    digest = hashlib.sha256(video).hexdigest()
    client_digest = hashlib.sha256(json.dumps(meta, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
    with LOCK:
        records = list_metadata(root)
        for saved in records:
            if saved['session_id'] == meta['session_id'] and saved['split'] != meta['split']:
                raise ValueError('This session already belongs to a different split; create a new session')
            if saved['request_id'] == meta['request_id']:
                if saved['video_sha256'] != digest or saved['client_metadata_sha256'] != client_digest:
                    raise ValueError('Recording ID already used with different content')
                return saved
            if saved['video_sha256'] == digest:
                raise ValueError('Identical video already exists; do not duplicate train/evaluation recordings')
        clip_id = uuid.uuid4().hex
        root.mkdir(parents=True, exist_ok=True)
        temporary = root / ('pending-' + clip_id)
        destination = root / 'clips' / clip_id
        temporary.mkdir()
        try:
            saved = {**meta, 'clip_id': clip_id, 'video_sha256': digest, 'client_metadata_sha256': client_digest,
                     'context_end': context_end,
                     'video_bytes': len(video), 'saved_at_utc': datetime.now(timezone.utc).isoformat(),
                     'video_validation': 'header_only_decode_pending', 'synchronization': 'preview_observations_not_encoded_frame_mapping',
                     'face_mouth': 'not_extracted', 'training_eligible': False}
            (temporary / 'rgb.webm').write_bytes(video)
            (temporary / 'metadata.json').write_text(json.dumps(saved, ensure_ascii=False), encoding='utf-8')
            destination.parent.mkdir(exist_ok=True)
            temporary.rename(destination)
        finally:
            if temporary.exists():
                if temporary.resolve().parent != root.resolve():
                    raise ValueError('Unsafe temporary cleanup path')
                shutil.rmtree(temporary)
        return saved


def summary(meta):
    return {key: meta[key] for key in ('clip_id', 'session_id', 'split', 'kind', 'sequence', 'duration_ms', 'saved_at_utc', 'video_sha256')}


def alignment(expected, actual):
    cells = [[None] * (len(actual) + 1) for _ in range(len(expected) + 1)]
    cells[0][0] = (0, 0, 0, [])
    for i in range(len(expected) + 1):
        for j in range(len(actual) + 1):
            if not i and not j:
                continue
            choices = []
            if i and j:
                s, d, ins, pairs = cells[i-1][j-1]
                match = expected[i-1] == actual[j-1]
                choices.append((s + (not match), d, ins, pairs + ([(i-1, j-1)] if match else [])))
            if i:
                s, d, ins, pairs = cells[i-1][j]
                choices.append((s, d+1, ins, pairs))
            if j:
                s, d, ins, pairs = cells[i][j-1]
                choices.append((s, d, ins+1, pairs))
            cells[i][j] = min(choices, key=lambda item: sum(item[:3]))
    return cells[-1][-1]


def evaluation_report(records):
    groups = {}
    excluded = []
    for meta in records:
        if meta['split'] not in {'validation', 'test'}:
            continue
        if meta.get('context_end') is not None and meta.get('context') != meta['context_end']:
            excluded.append({'clip_id': meta['clip_id'], 'reason': 'model_or_code_changed_during_capture'})
            continue
        key = json.dumps([meta['split'], meta.get('pipeline', 'unknown'), meta.get('context', {}), meta.get('client_model', {})], sort_keys=True)
        group = groups.setdefault(key, {'split': meta['split'], 'pipeline': meta.get('pipeline', 'unknown'),
            'context': meta.get('context', {}),
            'client_model': meta.get('client_model', {}),
            'clips': 0, 'reference_words': 0, 'substitutions': 0, 'deletions': 0, 'insertions': 0,
            'background_ms': 0, 'background_outputs': 0, 'timed_matches': 0, 'latencies_ms': [], 'early_outputs': 0})
        group['clips'] += 1
        events = meta['predictions']
        if meta['kind'] in {'rest', 'non_sign', 'transition'}:
            group['background_ms'] += meta['duration_ms']
            group['background_outputs'] += len(events)
            continue
        s, d, ins, pairs = alignment(meta['sequence'], [event['label'] for event in events])
        group['reference_words'] += len(meta['sequence'])
        group['substitutions'] += s
        group['deletions'] += d
        group['insertions'] += ins
        segments = meta.get('segments', [])
        for i, j in pairs:
            if len(segments) != len(meta['sequence']):
                break
            delay = events[j]['at_ms'] - segments[i]['end_ms']
            group['timed_matches'] += 1
            if delay < 0:
                group['early_outputs'] += 1
            else:
                group['latencies_ms'].append(delay)
    for group in groups.values():
        n = group['reference_words']
        group['wer'] = sum(group[k] for k in ('substitutions', 'deletions', 'insertions')) / n if n else None
        group['miss_rate'] = group['deletions'] / n if n else None
        group['false_outputs_per_minute'] = group['background_outputs'] * 60000 / group['background_ms'] if group['background_ms'] else None
        delays = sorted(group.pop('latencies_ms'))
        group['latency_p50_ms'] = delays[math.ceil(len(delays)*.5)-1] if delays else None
        group['latency_p95_ms'] = delays[math.ceil(len(delays)*.95)-1] if delays else None
    return {'scope': 'observed live outputs on validation/test recordings, not proof of independent generalization',
            'timing': 'browser observation clock; video decoded timestamps not verified', 'excluded': excluded, 'groups': list(groups.values())}
