import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
import project_paths
import sys
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import tcn_backend


class BackendTests(unittest.TestCase):
    def test_shared_augmentation_preserves_targets_and_source_recordings(self):
        import numpy as np
        from hand_tcn import augment_motion
        source = np.arange(24 * 130, dtype=np.float32).reshape(24, 130)
        before = source.copy()
        clips, targets = augment_motion([source], [3])
        np.testing.assert_array_equal(source, before)
        self.assertEqual(len(clips), len(targets))
        self.assertTrue(np.all(targets == 3))
        self.assertTrue(any(len(clip) == 12 for clip in clips))
        self.assertTrue(any(len(clip) == 24 for clip in clips))

    def test_position_tcn_is_causal_and_padding_does_not_change_pooled_output(self):
        import torch
        from hand_tcn import MotionTCN
        torch.set_num_threads(1)
        model = MotionTCN(2).eval()
        x = torch.randn(1, 130, 19)
        changed = x.clone()
        changed[:, :, 12:] += 10
        with torch.inference_mode():
            torch.testing.assert_close(model.encode(x)[:, :, :12], model.encode(changed)[:, :, :12])
            padded = torch.cat((x, torch.randn(1, 130, 4)), dim=-1)
            torch.testing.assert_close(model(x, torch.tensor([19])), model(padded, torch.tensor([19])))

    def test_missing_position_model_does_not_load_an_old_model(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as folder, patch.object(tcn_backend, "MODEL", Path(folder) / "missing.pt"):
            self.assertFalse(tcn_backend.TcnRuntime().status()["ready"])

    def test_training_rejects_insufficient_motion(self):
        with self.assertRaises(ValueError):
            tcn_backend.TcnRuntime().start_training([])

    def test_training_writes_only_isolated_model(self):
        samples = [{"label": label, "sequence": [[float(i + offset)] * 130 for _ in range(20)]}
                   for label, offset in (("a", 0), ("b", 20)) for i in range(5)]
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as folder, patch.object(tcn_backend, "MODEL", Path(folder) / "model.pt"):
            runtime = tcn_backend.TcnRuntime()
            runtime.start_training(samples)
            deadline = time.monotonic() + 60
            while runtime.job == "training" and time.monotonic() < deadline:
                time.sleep(0.05)
            self.assertEqual(runtime.job, "done", runtime.error)
            self.assertTrue(tcn_backend.MODEL.exists())
            self.assertEqual(runtime.status()["labels"], ["a", "b"])
            self.assertEqual(runtime.status()["input_dim"], 130)
            self.assertTrue(runtime.status()["missing_hand_augmented"])
            self.assertTrue(runtime.status()["sliding_window_augmented"])
            self.assertEqual(runtime.status()["original_recordings"], 10)
            self.assertEqual(runtime.status()["training_counts"], {"a": 5, "b": 5})
            result = runtime.predict([[0.0] * 130 for _ in range(20)])
            self.assertAlmostEqual(sum(item["score"] for item in result["scores"]), 1, places=5)
            with self.assertRaises(ValueError):
                runtime.predict([[0.0] * 126 for _ in range(20)])
            with self.assertRaises(ValueError):
                runtime.predict([[float("nan")] * 130 for _ in range(20)])

    def test_tracking_augmentation_keeps_wrist_positions_and_compacts_the_surviving_hand(self):
        import numpy as np
        from hand_tcn import tracking_variants
        full=np.concatenate((np.ones(63),np.full(63,2),np.array([0.1,0.2,0.3,0.4])))
        single=full.copy()
        single[63:126]=0
        single[128:130]=0
        original=np.stack((full,single))
        before=original.copy()
        base,swapped,left,right=tracking_variants(original)
        np.testing.assert_array_equal(original,before)
        np.testing.assert_array_equal(swapped[0,126:130],[0.3,0.4,0.1,0.2])
        np.testing.assert_array_equal(right[0,:63],np.full(63,2))
        np.testing.assert_array_equal(right[0,126:128],[0.3,0.4])
        np.testing.assert_array_equal(right[1,:63],np.ones(63))
        self.assertTrue(np.all(left[:,63:126]==0))

    def test_hand_only_recordings_require_recollection_instead_of_fake_wrist_coordinates(self):
        samples=[{"label":label,"sequence":[[1.0]*126 for _ in range(12)]} for label in ("a","b") for _ in range(5)]
        with self.assertRaisesRegex(ValueError,"126"):
            tcn_backend.TcnRuntime().start_training(samples)


    def test_reserved_rest_samples_are_not_trained_as_output_words(self):
        normal=[{"label":"hello","sequence":[[1.0]*130 for _ in range(12)]}]
        rest=[{"label":"__rest__","sequence":[[1.0]*130 for _ in range(12)]} for _ in range(5)]
        usable,excluded=tcn_backend.position_recordings(normal+rest)
        self.assertEqual(usable,normal)
        self.assertEqual(excluded,[])


if __name__ == "__main__":
    unittest.main()
