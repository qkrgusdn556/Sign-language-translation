import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
import project_paths
"""Recording-level holdout benchmark. Never installs or overwrites a model."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from hand_tcn import augment_motion, fit, read_motion


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=100)
    args = parser.parse_args()
    if not 1 <= args.epochs <= 300:
        parser.error('epochs must be 1-300')
    torch.set_num_threads(1)
    root = project_paths.ROOT
    sequences, targets, labels, digest = read_motion(root / 'data/position-tcn-training-source.json')
    rng = np.random.default_rng(321)
    train, test = [], []
    for target in range(len(labels)):
        indices = rng.permutation(np.flatnonzero(targets == target)).tolist()
        count = max(1, int(np.ceil(len(indices) * .2)))
        test.extend(indices[:count])
        train.extend(indices[count:])
    # Split originals BEFORE augmentation and normalization to avoid variant leakage.
    augmented, augmented_targets = augment_motion([sequences[i] for i in train], targets[train])
    model, mean, scale, _ = fit(augmented, augmented_targets, labels, list(range(len(augmented))), 321, args.epochs)
    confusion = {label: {other: 0 for other in labels} for label in labels}
    accepted = correct = wrong_accepted = 0
    observations = []
    with torch.inference_mode():
        for index in test:
            sequence = sequences[index]
            x = torch.from_numpy(((sequence - mean) / scale).T).unsqueeze(0)
            scores = model(x, torch.tensor([len(sequence)])).softmax(-1)[0].tolist()
            order = np.argsort(scores)[::-1]
            prediction, expected = labels[order[0]], labels[targets[index]]
            confusion[expected][prediction] += 1
            correct += prediction == expected
            passes = scores[order[0]] >= .85 and scores[order[0]] - scores[order[1]] >= .3
            accepted += passes
            wrong_accepted += passes and prediction != expected
            row = {'kind': 'heldout-clip', 'id': index, 'expected': [expected], 'observations': []}
            for end in range(12, len(sequence) + 1, 2):
                window = sequence[max(0, end - 20):end]
                window_x = torch.from_numpy(((window - mean) / scale).T).unsqueeze(0)
                probabilities = model(window_x, torch.tensor([len(window)])).softmax(-1)[0].tolist()
                row['observations'].append({'now': (end - 1) * 75, 'scores': [{'label': label, 'score': score} for label, score in zip(labels, probabilities)]})
            observations.append(row)
    report = {'note': 'Seeded recording-level holdout, NOT session-disjoint or continuous-sign accuracy. No real non-sign negatives available.',
              'source_sha256': digest, 'seed': 321, 'epochs': args.epochs, 'train_recordings': len(train),
              'test_recordings': len(test), 'correct': int(correct), 'accepted': int(accepted),
              'wrong_accepted': int(wrong_accepted), 'confusion': confusion}
    (root / 'data/tcn-holdout-report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    (root / 'data/tcn-holdout-scores.json').write_text(json.dumps({'note': report['note'], 'cases': observations}, ensure_ascii=False), encoding='utf-8')
    print(json.dumps({key: value for key, value in report.items() if key != 'confusion'}, ensure_ascii=False))


if __name__ == '__main__':
    main()
