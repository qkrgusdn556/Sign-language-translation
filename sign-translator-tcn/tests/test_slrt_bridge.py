import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
import project_paths
import base64
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

from capture_store import save_recording
from slrt_bridge import PROJECT, SlrtBridge


class SharedLearningTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=Path(__file__).parent)
        self.root = Path(self.temp.name)
        self.project = self.root / 'slrt'
        self.project.mkdir()
        shutil.copyfile(PROJECT / 'contract.py', self.project / 'contract.py')
        self.bridge = SlrtBridge(self.project, training_enabled=False)

    def tearDown(self):
        self.temp.cleanup()

    def recording(self, identity='a', split='train', kind='isolated', label='안녕하세요', video=None):
        metadata = {'schema_version': 1, 'request_id': identity * 32, 'session_id': 'session-one',
                    'split': split, 'kind': kind, 'consent': True, 'audio': False, 'share_slrt': True,
                    'started_at_utc': '2026-10-19T01:00:00Z', 'duration_ms': 2000, 'pipeline': 'dtw|jamo-joints',
                    'context': {'code': 'one'}, 'sequence': [label] if kind in {'isolated', 'continuous'} else [],
                    'camera': {'width': 960, 'height': 540, 'mirrored': False}, 'mime_type': 'video/webm',
                    'frames': [{'elapsed_ms': i * 100, 'media_time_ms': 2000 + i * 100, 'hands': [], 'pose': None, 'pose_status': 'not_run'} for i in range(15)], 'predictions': []}
        body = {'metadata': metadata, 'video_base64': base64.b64encode(video or b'\x1a\x45\xdf\xa3' + identity.encode() * 64).decode()}
        meta = save_recording(body, self.root / 'source')
        return meta, self.root / 'source/clips' / meta['clip_id'] / 'rgb.webm'

    def test_same_video_and_label_are_native_training_inputs_without_re_recording(self):
        meta, source = self.recording()
        shared = self.bridge.share(meta, source)
        target = self.bridge.data / 'clips' / shared['clip_id']
        saved = json.loads((target / 'metadata.json').read_text(encoding='utf-8'))
        self.bridge.contract().validate(saved)
        self.assertEqual((target / 'rgb.webm').read_bytes(), source.read_bytes())
        self.assertEqual(saved['sequence'], ['안녕하세요'])
        self.assertEqual(saved['source_clip_id'], meta['clip_id'])
        self.assertEqual(saved['landmarks']['status'], 'not_extracted')
        self.assertIsNone(saved['synchronization']['encoded_frame_mapping'])
        self.assertEqual(len(saved['preview_frames']), 15)
        self.assertEqual(self.bridge.status()['pending_training_clips'], 1)

    def test_retries_never_duplicate_or_overwrite_a_native_clip(self):
        meta, source = self.recording()
        first = self.bridge.share(meta, source)
        before = (self.bridge.data / 'clips' / first['clip_id'] / 'metadata.json').read_bytes()
        again = self.bridge.share(meta, source)
        self.assertEqual(first['clip_id'], again['clip_id'])
        self.assertEqual(again['status'], 'already_shared')
        self.assertEqual(len(self.bridge.records()), 1)
        self.assertEqual((self.bridge.data / 'clips' / first['clip_id'] / 'metadata.json').read_bytes(), before)

    def test_validation_video_stays_out_of_training_and_session_split_cannot_change(self):
        meta, source = self.recording(split='validation')
        self.bridge.share(meta, source)
        self.assertEqual(self.bridge.status()['saved_training_clips'], 0)
        changed = {**meta, 'split': 'train'}
        with self.assertRaises(ValueError):
            self.bridge.share(changed, source)

    def test_continuous_clips_are_not_mislabeled_as_single_words(self):
        meta, source = self.recording(kind='continuous')
        with self.assertRaisesRegex(ValueError, 'segment'):
            self.bridge.share(meta, source)
        self.assertEqual(self.bridge.records(), [])

    def test_corrupt_or_changed_video_is_not_imported(self):
        meta, source = self.recording()
        source.write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError, 'fingerprint'):
            self.bridge.share(meta, source)
        self.assertEqual(self.bridge.records(), [])

    def test_background_clips_keep_empty_sequence_and_training_needs_each_class(self):
        for identity, kind in [('a', 'isolated'), ('b', 'isolated'), ('c', 'rest'), ('d', 'non_sign')]:
            meta, source = self.recording(identity=identity, kind=kind)
            self.bridge.share(meta, source)
        state = self.bridge.status()
        self.assertEqual(state['counts'], {'안녕하세요': 2, '배경 동작': 2})
        self.assertFalse(state['training_allowed'])
        with patch('slrt_bridge.subprocess.Popen') as process:
            with self.assertRaisesRegex(ValueError, 'disabled'):
                self.bridge.start_training()
            process.assert_not_called()

    def test_missing_native_project_does_not_modify_source_video(self):
        meta, source = self.recording()
        before = hashlib.sha256(source.read_bytes()).hexdigest()
        with self.assertRaises(ValueError):
            SlrtBridge(self.root / 'missing').share(meta, source)
        self.assertEqual(hashlib.sha256(source.read_bytes()).hexdigest(), before)

    def test_training_button_delegates_to_native_environment_without_running_training_in_test(self):
        self.bridge.training_enabled = True
        self.root.joinpath('data').mkdir()
        state = {'can_train': True, 'job': {'status': 'idle'}}
        with patch.object(self.bridge, 'status', return_value=state), patch('slrt_bridge.ROOT', self.root), patch('slrt_bridge.subprocess.Popen') as process:
            self.assertTrue(self.bridge.start_training()['started'])
            arguments, options = process.call_args
            self.assertIn('training_job import TrainingJob', arguments[0][-1])
            self.assertIn('j.worker.join()', arguments[0][-1])
            self.assertEqual(options['cwd'], self.project)
            self.assertIn('python.exe', arguments[0][0].lower())

    def test_saved_shared_clips_are_pending_until_the_native_training_manifest_includes_them(self):
        meta, source = self.recording()
        shared = self.bridge.share(meta, source)
        saved = self.bridge.records()[0]
        model = self.project / 'models/test-model'
        model.mkdir(parents=True)
        keys = ('clip_id', 'video_sha256', 'session_id', 'split', 'kind', 'sequence')
        (self.project / 'models/current.json').write_text(json.dumps({'model_id': 'test-model'}), encoding='utf-8')
        (model / 'report.json').write_text(json.dumps({'model_id': 'test-model', 'training_manifest': [{key: saved[key] for key in keys}]}), encoding='utf-8')
        self.assertEqual(self.bridge.status()['pending_training_clips'], 0)
        self.assertEqual(self.bridge.status()['shared_clips'], 1)


if __name__ == '__main__':
    unittest.main()
