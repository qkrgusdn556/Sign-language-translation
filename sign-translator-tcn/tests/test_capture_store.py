import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
import project_paths
import base64
import copy
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from capture_store import alignment, evaluation_report, list_metadata, save_recording, validate_metadata


def metadata():
    return {'schema_version': 1, 'request_id': 'a' * 32, 'session_id': 'session-one', 'split': 'test',
            'kind': 'isolated', 'sequence': ['hello'], 'duration_ms': 1200, 'consent': True, 'audio': False,
            'mime_type': 'video/webm', 'camera': {'mirrored': False}, 'pipeline': 'tcn|jamo-consensus',
            'context': {'code_sha256': 'source-one'}, 'frames': [], 'predictions': [], 'segments': []}


def body(meta=None, payload=b'x'):
    return {'metadata': meta or metadata(), 'video_base64': base64.b64encode(b'\x1a\x45\xdf\xa3' + payload * 64).decode()}


class CaptureTests(unittest.TestCase):
    def test_append_only_storage_retry_and_private_source_preservation(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as folder:
            root = Path(folder)
            saved = save_recording(body(), root)
            self.assertTrue((root / 'clips' / saved['clip_id'] / 'rgb.webm').is_file())
            self.assertEqual(saved['video_validation'], 'header_only_decode_pending')
            self.assertFalse(saved['training_eligible'])
            self.assertEqual(save_recording(body(), root)['clip_id'], saved['clip_id'])
            self.assertEqual(len(list_metadata(root)), 1)
            changed = body(payload=b'y')
            with self.assertRaises(ValueError):
                save_recording(changed, root)
            self.assertEqual(list_metadata(root)[0]['video_sha256'], saved['video_sha256'])

    def test_session_split_and_duplicate_video_cannot_leak(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as folder:
            save_recording(body(), folder)
            meta = metadata()
            meta.update(request_id='b' * 32, split='train')
            with self.assertRaisesRegex(ValueError, 'different split'):
                save_recording(body(meta, b'y'), folder)
            meta.update(session_id='other-session')
            with self.assertRaisesRegex(ValueError, 'Identical video'):
                save_recording(body(meta), folder)
            self.assertEqual(len(list_metadata(folder)), 1)

    def test_metadata_rejects_consent_paths_invalid_times_and_background_labels(self):
        bad = [{'consent': False}, {'audio': True}, {'request_id': '../escape'}, {'session_id': '../../x'},
               {'duration_ms': float('nan')}, {'kind': 'rest'}, {'camera': {'mirrored': True}},
               {'predictions': [{'label': 'hi', 'at_ms': 99999}]}, {'split': 'anything'}]
        for change in bad:
            with self.subTest(change=change), self.assertRaises(ValueError):
                validate_metadata({**metadata(), **change})

    def test_real_observation_presence_and_order_are_validated(self):
        meta = metadata()
        frame = {'elapsed_ms': 10, 'media_time_ms': 5000, 'hands': [], 'pose': None, 'pose_status': 'missing'}
        meta['frames'] = [frame]
        validate_metadata(meta)
        meta['frames'] = [frame, {**frame, 'elapsed_ms': 5}]
        with self.assertRaises(ValueError):
            validate_metadata(meta)

    def test_metrics_include_substitution_omission_insertion_and_background_outputs(self):
        record = metadata()
        record.update(sequence=['hello', 'name'], kind='continuous', segments=[
            {'label': 'hello', 'start_ms': 0, 'end_ms': 100}, {'label': 'name', 'start_ms': 100, 'end_ms': 600}],
            predictions=[{'label': 'hello', 'at_ms': 300}, {'label': 'wrong', 'at_ms': 700}, {'label': 'extra', 'at_ms': 900}])
        rest = {**metadata(), 'kind': 'rest', 'sequence': [], 'duration_ms': 1000,
                'predictions': [{'label': 'wrong', 'at_ms': 600}]}
        train = {**record, 'split': 'train'}
        group = evaluation_report([record, rest, train])['groups'][0]
        self.assertEqual(group['clips'], 2)
        self.assertEqual(group['wer'], 1)
        self.assertEqual(group['false_outputs_per_minute'], 60)
        self.assertEqual(group['latency_p50_ms'], 200)
        self.assertEqual(alignment(['a', 'b'], ['a'])[:3], (0, 1, 0))

    def test_models_are_separate_and_untimed_latency_is_not_invented(self):
        first = metadata()
        first['predictions'] = [{'label': 'hello', 'at_ms': 300}]
        second = {**first, 'context': {'code_sha256': 'source-two'}}
        groups = evaluation_report([first, second])['groups']
        self.assertEqual(len(groups), 2)
        self.assertIsNone(groups[0]['latency_p50_ms'])
        self.assertEqual(groups[0]['wer'], 0)
        self.assertEqual(evaluation_report([])['groups'], [])
        second = {**first, 'client_model': {'yolo': 'different-model'}}
        self.assertEqual(len(evaluation_report([first, second])['groups']), 2)

    def test_changed_model_capture_is_saved_but_excluded_from_comparison(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as folder:
            saved = save_recording(body(), folder, context_end={'code_sha256': 'changed'})
            report = evaluation_report([saved])
            self.assertEqual(report['groups'], [])
            self.assertEqual(len(report['excluded']), 1)

    def test_timed_labels_require_matching_order_and_valid_bounds(self):
        meta = metadata()
        for segment in [{'label': 'wrong', 'start_ms': 0, 'end_ms': 100},
                        {'label': 'hello', 'start_ms': 100, 'end_ms': 10},
                        {'label': 'hello', 'start_ms': 0, 'end_ms': 1300}]:
            with self.assertRaises(ValueError):
                validate_metadata({**meta, 'segments': [segment]})


if __name__ == '__main__':
    unittest.main()
