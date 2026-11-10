import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
import project_paths
import json
from pathlib import Path
import sys
import tempfile
import unittest
import zipfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from portable_backup import backup, restore


class PortableBackupTest(unittest.TestCase):
    def test_roundtrip_and_exclusions(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / 'source'
            root.mkdir()
            for name in ('project_paths.py', 'frontend/app.js', 'backend/web_server.py', 'scripts/start-web.ps1', 'data/sign-samples.json', 'data/yolo-studio/model.onnx',
                         '.env', 'data/yolo-runtime/installed.py', '.build/test.log'):
                path = root / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(name, encoding='utf-8')
            archive = Path(temp) / 'transfer.zip'
            self.assertEqual(backup(root, archive), 6)
            target = Path(temp) / 'restored'
            self.assertEqual(restore(archive, target), 6)
            self.assertTrue((target / 'frontend/app.js').is_file())
            self.assertTrue((target / 'scripts/start-web.ps1').is_file())
            self.assertEqual((target / 'data/sign-samples.json').read_text(), 'data/sign-samples.json')
            self.assertFalse((target / '.env').exists())
            self.assertFalse((target / 'data/yolo-runtime').exists())
            with self.assertRaises(ValueError):
                restore(archive, target)
            with self.assertRaises(FileExistsError):
                backup(root, archive)

    def test_unsafe_and_corrupt_archive_rejected_before_writes(self):
        with tempfile.TemporaryDirectory() as temp:
            for index, name in enumerate(('../escape', 'C:/escape', 'app.js')):
                archive = Path(temp) / f'{index}.zip'
                with zipfile.ZipFile(archive, 'w') as output:
                    output.writestr(name, 'content')
                    output.writestr('transfer-manifest.json', json.dumps({name: 'bad-checksum'}))
                target = Path(temp) / f'output{index}'
                with self.assertRaises(ValueError):
                    restore(archive, target)
                self.assertFalse(target.exists())


if __name__ == '__main__':
    unittest.main()
