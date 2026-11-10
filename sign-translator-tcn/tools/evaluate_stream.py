import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
import project_paths
"""Replay existing training clips for regression diagnostics, not held-out accuracy."""
import json
from pathlib import Path

import numpy as np
import torch

from tcn_backend import TcnRuntime


def main():
    root = project_paths.ROOT
    rows = json.loads((root / 'data/position-tcn-training-source.json').read_text(encoding='utf-8-sig'))['samples']
    runtime = TcnRuntime()
    runtime.load()
    cases = []
    for index, row in enumerate(rows):
        cases.append({'kind': 'training-clip', 'id': index, 'expected': [row['label']], 'sequence': row['sequence']})
    representatives = {}
    for row in rows:
        representatives.setdefault(row['label'], row['sequence'])
    for a, first in representatives.items():
        for b, second in representatives.items():
            if a != b:
                # Synthetic bridge is a stress probe, not a recorded transition.
                bridge = [(np.asarray(first[-1]) * (1 - t) + np.asarray(second[0]) * t).tolist() for t in (.25, .5, .75)]
                cases.append({'kind': 'synthetic-pair', 'expected': [a, b], 'sequence': first + bridge + second})
    windows, destinations = [], []
    for case in cases:
        sequence = case.pop('sequence')
        case['observations'] = []
        for end in range(12, len(sequence) + 1, 2):
            windows.append(np.asarray(sequence[max(0, end - 20):end], dtype=np.float32))
            destinations.append((case, (end - 1) * 75))
    with torch.inference_mode():
        for start in range(0, len(windows), 128):
            batch = windows[start:start + 128]
            lengths = torch.tensor([len(window) for window in batch])
            x = torch.zeros(len(batch), 130, int(lengths.max()))
            for i, window in enumerate(batch):
                x[i, :, :len(window)] = ((torch.from_numpy(window) - runtime.mean) / runtime.scale).T
            probabilities = runtime.model(x, lengths).softmax(-1).tolist()
            for offset, scores in enumerate(probabilities):
                case, now = destinations[start + offset]
                case['observations'].append({'now': now, 'scores': [{'label': label, 'score': score} for label, score in zip(runtime.labels, scores)]})
    output = root / 'data/stream-replay-scores.json'
    output.write_text(json.dumps({'note': 'Training-set replay and synthetic pairs, NOT independent continuous-sign accuracy.', 'cases': cases}, ensure_ascii=False), encoding='utf-8')
    print(json.dumps({'cases': len(cases), 'windows': len(windows), 'output': str(output)}))


if __name__ == '__main__':
    main()
