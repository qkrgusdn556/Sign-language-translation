"""Verify relocated servers from a different working directory."""
from pathlib import Path
import json
import os
import socket
import subprocess
import sys
import tempfile
import time
import unittest
from urllib.request import urlopen
from urllib.error import HTTPError, URLError

ROOT = Path(__file__).resolve().parents[1]


class LayoutTest(unittest.TestCase):
    def check_server(self, script, routes, studio=False):
        with tempfile.TemporaryDirectory(dir=ROOT / '.build') as folder:
            with socket.socket() as sock:
                sock.bind(('127.0.0.1', 0))
                port = sock.getsockname()[1]
            args = [sys.executable, str(ROOT / 'backend' / script), '--port', str(port)]
            if studio:
                args += ['--data-root', str(Path(folder) / 'studio')]
            with (ROOT / '.build' / f'layout-server-{port}.log').open('w+') as log:
                process = subprocess.Popen(args, cwd=folder, stdout=log, stderr=log,
                    creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0)
                try:
                    base = f'http://127.0.0.1:{port}'
                    for _ in range(100):
                        try:
                            with urlopen(base + '/', timeout=1) as response:
                                self.assertEqual(response.status, 200)
                            break
                        except (URLError, TimeoutError):
                            if process.poll() is not None:
                                log.seek(0)
                                self.fail(log.read())
                            time.sleep(.1)
                    else:
                        self.fail('Server startup timed out')
                    for route in routes:
                        with urlopen(base + route, timeout=20) as response:
                            self.assertEqual(response.status, 200, route)
                            self.assertTrue(response.read(), route)
                    if not studio:
                        for route in ('/backend/gemini_polish.py', '/data/sign-samples.json', '/.build/before-folder-layout/app.js'):
                            with self.assertRaises(HTTPError) as error:
                                urlopen(base + route)
                            self.assertIn(error.exception.code, (403, 404))
                            error.exception.close()
                finally:
                    if os.name == 'nt':
                        # Windows venv launchers may own a second Python process.
                        subprocess.run(['taskkill', '/PID', str(process.pid), '/T', '/F'],
                                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                                       creationflags=subprocess.CREATE_NO_WINDOW)
                    else:
                        process.terminate()
                    process.wait(timeout=10)

    def test_web_routes(self):
        routes = ['/app.js?v=old-bookmark', '/vendor/hangul.js', '/api/gemini/status', '/api/recordings/context']
        routes += ['/frontend/' + p.name for p in (ROOT / 'frontend').iterdir() if p.suffix in ('.js', '.css')]
        self.check_server('web_server.py', routes)

    def test_studio_routes(self):
        self.check_server('yolo_studio.py', ['/yolo-train.js', '/yolo-annotate.js', '/yolo-train.css', '/yolo-annotate.css', '/api/status'], True)
