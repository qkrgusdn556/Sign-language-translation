import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
import project_paths
import argparse
import json
import math
import os
import shutil
import threading
import hashlib
import socket
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = project_paths.ROOT
DATA = ROOT / "data" / "sign-samples.json"
LOCK = threading.Lock()


class LocalServer(ThreadingHTTPServer):
    allow_reuse_address = False

    def server_bind(self):
        if os.name == 'nt':
            self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
        super().server_bind()


def recording_context():
    names = ['app.js', 'recognition.js', 'input_activity.js', 'hand_tracking.js', 'hand_precision.js', 'hand_precision_worker.js', 'tcn_stream.js', 'fingerspell_consensus.js', 'fingerspell.js', 'raw_capture.js', 'shared_motion.js', 'slrt_bridge.py', 'rest_pose.js', 'sample_labels.py']
    from tcn_backend import runtime
    model = runtime.status()
    return {'code_sha256': hashlib.sha256(b''.join((ROOT / ('frontend' if name.endswith('.js') else 'backend') / name).read_bytes() for name in names)).hexdigest(),
            'tcn_model_sha256': model.get('model_sha256'),
            'note': 'No automatic model training or original-sample migration'}


def valid_samples(samples):
    def vector(value):
        return isinstance(value, list) and len(value) in (126, 130, 136) and all(
            type(n) in (int, float) and math.isfinite(n) and abs(n) <= 100 for n in value
        )

    if not isinstance(samples, list) or len(samples) > 5000:
        return False
    for sample in samples:
        if not isinstance(sample, dict) or not isinstance(sample.get("label"), str) or not 1 <= len(sample["label"].strip()) <= 40:
            return False
        sequence = sample.get("sequence")
        if not vector(sample.get("vector")) and not (
            isinstance(sequence, list) and 6 <= len(sequence) <= 32 and all(vector(frame) for frame in sequence)
        ):
            return False
    return True


class Handler(SimpleHTTPRequestHandler):
    def translate_path(self, path):
        from capture_store import ROOT as CAPTURES
        from slrt_bridge import bridge
        resolved = Path(super().translate_path(path)).resolve()
        if path.split('?', 1)[0] == '/':
            return str(ROOT / 'frontend/index.html')
        if not any(resolved.is_relative_to(ROOT / folder) for folder in ('frontend', 'vendor')):
            return str(ROOT / '.private-data-unavailable')
        if resolved.is_relative_to(DATA.parent) or resolved.is_relative_to(CAPTURES.resolve()) or resolved.is_relative_to(bridge.data.resolve()):
            return str(ROOT / ".private-data-unavailable")
        return str(resolved)

    def allowed(self):
        host = self.headers.get("Host", "")
        expected = f"127.0.0.1:{self.server.server_port}"
        localhost = f"localhost:{self.server.server_port}"
        origin = self.headers.get("Origin")
        return host in (expected, localhost) and (origin is None or origin == f"http://{host}")

    def reply(self, status, value):
        body = json.dumps(value, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if not self.allowed():
            return self.reply(403, {"error": "Local access only"})
        route = self.path.split('?', 1)[0]
        # Redirect old asset URLs so existing bookmarks and module bases remain valid.
        if route.count('/') == 1 and (ROOT / 'frontend' / route[1:]).is_file():
            self.send_response(307)
            self.send_header('Location', '/frontend' + self.path)
            self.end_headers()
            return
        if route == '/api/gemini/status':
            from gemini_polish import status
            return self.reply(200, status())
        if route == '/api/slrt/status':
            from slrt_bridge import bridge
            try:
                return self.reply(200, bridge.status())
            except (OSError, ValueError, KeyError) as error:
                return self.reply(503, {'available': False, 'error': str(error)})
        if route == '/api/recordings/context':
            return self.reply(200, recording_context())
        if route in ('/api/recordings', '/api/recordings/evaluation'):
            from capture_store import list_metadata, summary, evaluation_report
            try:
                records = list_metadata()
                return self.reply(200, evaluation_report(records) if route.endswith('/evaluation') else {'clips': [summary(row) for row in records]})
            except (OSError, ValueError) as error:
                return self.reply(500, {'error': str(error)})
        if route.startswith('/api/recordings/') and route.endswith('/video'):
            from capture_store import ROOT as CAPTURES, ID
            clip_id = route[len('/api/recordings/'):-len('/video')]
            if not ID.fullmatch(clip_id):
                return self.reply(400, {'error': 'Invalid recording ID'})
            path = CAPTURES / 'clips' / clip_id / 'rgb.webm'
            if not path.is_file():
                return self.reply(404, {'error': 'Recording not found'})
            body = path.read_bytes()
            self.send_response(200)
            self.send_header('Content-Type', 'video/webm')
            self.send_header('Content-Length', str(len(body)))
            self.send_header('Cache-Control', 'no-store')
            self.end_headers()
            self.wfile.write(body)
            return
        if self.path.split("?", 1)[0] == "/api/tcn/status":
            from tcn_backend import runtime
            return self.reply(200, runtime.status())
        if self.path.split("?", 1)[0] == "/api/samples":
            try:
                with LOCK:
                    value = json.loads(DATA.read_text(encoding="utf-8")) if DATA.exists() else None
                if value is not None and not valid_samples(value.get("samples")):
                    raise ValueError("Invalid saved samples")
                return self.reply(200, {"samples": value["samples"] if value else None})
            except (OSError, ValueError, AttributeError) as error:
                return self.reply(500, {"error": str(error)})
        if self.path.split("?", 1)[0].startswith("/data"):
            return self.reply(403, {"error": "Use sample backup export"})
        return super().do_GET()

    def do_POST(self):
        if not self.allowed():
            return self.reply(403, {"error": "Local access only"})
        if self.path in ('/api/gemini/configure', '/api/gemini/polish', '/api/gemini/diagnose'):
            from gemini_polish import configure, polish, diagnose
            try:
                length = int(self.headers.get('Content-Length', '0'))
                if not 0 < length <= 24000:
                    raise ValueError('요청 크기가 올바르지 않습니다.')
                value = json.loads(self.rfile.read(length))
                if not isinstance(value, dict):
                    raise ValueError('잘못된 요청입니다.')
                action = configure if self.path.endswith('/configure') else diagnose if self.path.endswith('/diagnose') else polish
                return self.reply(200, action(value))
            except (ValueError, TypeError):
                return self.reply(400, {'error': '입력 내용과 전송 동의를 확인해 주세요.'})
            except RuntimeError as error:
                return self.reply(503, {'error': str(error)})
        if self.path in {'/api/slrt/train', '/api/slrt/sync'}:
            from slrt_bridge import bridge
            try:
                length = int(self.headers.get('Content-Length', '0'))
                if not 0 < length <= 1024:
                    raise ValueError('Invalid SLRT request size')
                value = json.loads(self.rfile.read(length))
                command = 'train' if self.path.endswith('/train') else 'sync'
                if value != {command: True}:
                    raise ValueError('Invalid SLRT request')
                if command == 'train':
                    return self.reply(202, bridge.start_training())
                from capture_store import list_metadata, ROOT as CAPTURES
                results = []
                for meta in list_metadata():
                    if meta.get('share_slrt') is True:
                        try:
                            result = bridge.share(meta, CAPTURES / 'clips' / meta['clip_id'] / 'rgb.webm')
                        except (OSError, ValueError, KeyError) as error:
                            result = {'status': 'error', 'error': str(error)}
                        results.append({'source_clip_id': meta['clip_id'], **result})
                return self.reply(200, {'results': results})
            except (OSError, ValueError, KeyError) as error:
                return self.reply(409, {'error': str(error)})
        if self.path in {'/api/recordings', '/api/shared-motion'}:
            from capture_store import save_recording, summary
            try:
                length = int(self.headers.get('Content-Length', '0'))
                if not 0 < length <= 64 * 1024 * 1024:
                    return self.reply(413, {'error': 'Invalid recording size'})
                value = json.loads(self.rfile.read(length))
                motion = self.path == '/api/shared-motion'
                if not isinstance(value, dict):
                    raise ValueError('Expected recording object')
                if motion:
                    meta = value.get('metadata', {})
                    sample = meta.get('motion_sample')
                    from sample_labels import REST_LABEL
                    background = isinstance(sample, dict) and sample.get('label') == REST_LABEL and meta.get('kind') == 'rest' and meta.get('sequence') == []
                    isolated = isinstance(sample, dict) and sample.get('label') != REST_LABEL and meta.get('kind') == 'isolated' and meta.get('sequence') == [sample.get('label')]
                    if meta.get('share_slrt') is not True or meta.get('split') != 'train' or not (background or isolated) or not sample.get('sequence') or not valid_samples([sample]):
                        raise ValueError('Shared motion requires one labelled training sequence and video')
                    if any(len(frame) != 130 for frame in sample['sequence']):
                        raise ValueError('Shared motion needs observed shoulder-relative 130-value frames')
                saved = save_recording(value, context_end=recording_context())
                if motion:
                    with LOCK:
                        previous = json.loads(DATA.read_text(encoding='utf-8')) if DATA.exists() else {'samples': []}
                        if not valid_samples(previous.get('samples')):
                            raise ValueError('Existing samples are invalid; saved video retained')
                        applied = previous.get('applied_shared_motion', [])
                        if saved['clip_id'] not in applied:
                            from slrt_bridge import atomic_json
                            samples = [*previous['samples'], sample]
                            if not valid_samples(samples):
                                raise ValueError('Sample limit exceeded; saved video retained')
                            atomic_json(DATA, {'version': 3, 'samples': samples, 'applied_shared_motion': [*applied, saved['clip_id']]})
                result = {'clip': summary(saved), 'sample_saved': motion}
                if saved.get('share_slrt') is True:
                    from slrt_bridge import bridge
                    from capture_store import ROOT as CAPTURES
                    try:
                        result['slrt'] = bridge.share(saved, CAPTURES / 'clips' / saved['clip_id'] / 'rgb.webm')
                    except (OSError, ValueError, KeyError) as error:
                        result['slrt'] = {'status': 'error', 'error': str(error)}
                return self.reply(201, result)
            except (ValueError, TypeError) as error:
                return self.reply(400, {'error': str(error)})
            except OSError as error:
                return self.reply(500, {'error': str(error)})
        if self.path in ("/api/tcn/predict", "/api/tcn/train"):
            try:
                from tcn_backend import runtime
                length = int(self.headers.get("Content-Length", "0"))
                if not 0 < length <= 128 * 1024:
                    return self.reply(413, {"error": "Invalid request size"})
                value = json.loads(self.rfile.read(length))
                if self.path == "/api/tcn/train":
                    if value != {"train": True}:
                        return self.reply(400, {"error": "Invalid training request"})
                    with LOCK:
                        samples = json.loads(DATA.read_text(encoding="utf-8")).get("samples") if DATA.exists() else []
                    if not valid_samples(samples):
                        return self.reply(400, {"error": "Invalid saved samples"})
                    return self.reply(202, runtime.start_training(samples))
                sequence = value.get("sequence") if isinstance(value, dict) else None
                if not valid_samples([{"label": "input", "sequence": sequence}]) or any(len(frame) != 130 for frame in sequence):
                    return self.reply(400, {"error": "Expected 6-32 finite 130-dimensional frames"})
                return self.reply(200, runtime.predict(sequence))
            except (ValueError, TypeError) as error:
                return self.reply(400, {"error": str(error)})
            except (OSError, RuntimeError) as error:
                return self.reply(503, {"error": str(error)})
        if self.path != "/api/samples":
            return self.reply(404, {"error": "Not found"})
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if not 0 < length <= 64 * 1024 * 1024:
                return self.reply(413, {"error": "Invalid file size"})
            value = json.loads(self.rfile.read(length))
            if not isinstance(value, dict) or not valid_samples(value.get("samples")):
                return self.reply(400, {"error": "Invalid samples; existing data is unchanged"})
            with LOCK:
                DATA.parent.mkdir(parents=True, exist_ok=True)
                temporary = DATA.with_suffix(".tmp")
                with temporary.open("w", encoding="utf-8") as output:
                    previous = json.loads(DATA.read_text(encoding='utf-8')) if DATA.exists() else {}
                    json.dump({"version": 3, "samples": value["samples"], 'applied_shared_motion': previous.get('applied_shared_motion', [])}, output, ensure_ascii=False)
                    output.flush()
                    os.fsync(output.fileno())
                if DATA.exists():
                    shutil.copy2(DATA, DATA.with_suffix(".json.bak"))
                os.replace(temporary, DATA)
            return self.reply(200, {"saved": len(value["samples"])})
        except (OSError, ValueError) as error:
            return self.reply(500, {"error": str(error)})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=5191)
    parser.add_argument('--recordings-dir', type=Path, help='Separate recording storage, primarily for isolated tests')
    parser.add_argument('--samples-file', type=Path, help='Separate sample store for isolated tests')
    parser.add_argument('--slrt-data-root', type=Path, help='Separate SLRT import store; disables real training')
    args = parser.parse_args()
    if args.recordings_dir:
        import capture_store
        capture_store.ROOT = args.recordings_dir.resolve()
    if args.samples_file:
        DATA = args.samples_file.resolve()
    if args.slrt_data_root:
        import slrt_bridge
        slrt_bridge.bridge = slrt_bridge.SlrtBridge(data_root=args.slrt_data_root, training_enabled=False)
    server = LocalServer(("127.0.0.1", args.port), partial(Handler, directory=str(ROOT)))
    print(f"http://127.0.0.1:{args.port}", flush=True)
    server.serve_forever()
