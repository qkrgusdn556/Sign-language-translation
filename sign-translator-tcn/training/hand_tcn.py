import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
import project_paths
"""Causal TCN using hand and shoulder-relative wrist features."""
import hashlib
import json
from collections import Counter

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

INPUT_DIM = 130
FEATURE_SCHEMA = "shoulder-hands-130-v1"


class CausalBlock(nn.Module):
    def __init__(self, channels, dilation):
        super().__init__()
        self.padding = 2 * dilation
        self.conv = nn.Conv1d(channels, channels, 3, dilation=dilation)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        return F.relu(x + self.dropout(F.relu(self.conv(F.pad(x, (self.padding, 0))))))


class MotionTCN(nn.Module):
    def __init__(self, classes):
        super().__init__()
        self.project = nn.Conv1d(INPUT_DIM, 32, 1)
        self.blocks = nn.Sequential(*(CausalBlock(32, d) for d in (1, 2, 4)))
        self.head = nn.Linear(32, classes)

    def encode(self, x):
        return self.blocks(F.relu(self.project(x)))

    def forward(self, x, lengths):
        features = self.encode(x)
        mask = torch.arange(x.shape[-1], device=x.device)[None, :] < lengths[:, None]
        return self.head((features * mask[:, None, :]).sum(-1) / lengths[:, None])


def read_motion(path):
    raw = path.read_bytes()
    sequences, words, seen = [], [], set()
    for row in json.loads(raw.decode("utf-8-sig"))["samples"]:
        sequence = np.asarray(row["sequence"], dtype=np.float32)
        if sequence.ndim != 2 or sequence.shape[1] not in (130, 136):
            raise ValueError("Shoulder-relative motion recordings need 130 values; recollect hand-only 126-value clips")
        if not 6 <= len(sequence) <= 32 or not np.isfinite(sequence).all():
            raise ValueError("Invalid motion recording")
        sequence = sequence[:, :INPUT_DIM].copy()
        key = sequence.tobytes()
        if key in seen:
            raise ValueError("Duplicate hand sequence; collect distinct recordings")
        seen.add(key)
        sequences.append(sequence)
        words.append(row["label"])
    labels = sorted(set(words))
    if len(labels) < 2 or min(Counter(words).values()) < 5:
        raise ValueError("Need two words with five motion recordings each")
    return sequences, np.array([labels.index(word) for word in words]), labels, hashlib.sha256(raw).hexdigest()


def tracking_variants(sequence):
    swapped = np.concatenate((sequence[:, 63:126], sequence[:, :63], sequence[:, 128:130], sequence[:, 126:128]), axis=1)
    variants = [sequence.copy(), swapped]
    # Compact the surviving hand into slot zero, matching HandLandmarker output.
    for preferred in (0, 1):
        masked = np.zeros_like(sequence)
        for i, frame in enumerate(sequence):
            slot = preferred if np.any(frame[preferred * 63:(preferred + 1) * 63]) else 1 - preferred
            masked[i, :63] = frame[slot * 63:(slot + 1) * 63]
            masked[i, 126:128] = frame[126 + slot * 2:128 + slot * 2]
        variants.append(masked)
    return variants


def augment_motion(sequences, targets):
    augmented, augmented_targets = [], []
    for sequence, target in zip(sequences, targets):
        lengths = sorted({min(len(sequence), max(12, int(np.ceil(len(sequence) * fraction))))
                          for fraction in (0.6, 0.75, 0.9, 1.0)})
        for variant in tracking_variants(sequence):
            for length in lengths:
                augmented.append(variant[:length])
                augmented_targets.append(target)
            for end in range(12, len(variant) + 1, 4):
                augmented.append(variant[max(0, end - 20):end])
                augmented_targets.append(target)
    return augmented, np.array(augmented_targets)


def fit(sequences, targets, labels, indices, seed, epochs):
    torch.manual_seed(seed)
    frames = np.concatenate([sequences[i] for i in indices])
    mean, scale = frames.mean(0), np.maximum(frames.std(0), 1e-4)
    lengths = torch.tensor([len(sequences[i]) for i in indices], dtype=torch.long)
    data = np.zeros((len(indices), INPUT_DIM, int(lengths.max())), dtype=np.float32)
    for row, index in enumerate(indices):
        data[row, :, :len(sequences[index])] = ((sequences[index] - mean) / scale).T
    x, y = torch.from_numpy(data), torch.tensor(targets[indices], dtype=torch.long)
    model = MotionTCN(len(labels))
    optimizer = torch.optim.Adam(model.parameters(), lr=0.003, weight_decay=1e-4)
    weights = len(indices) / (len(labels) * np.bincount(targets[indices], minlength=len(labels)))
    loss_fn = nn.CrossEntropyLoss(weight=torch.tensor(weights, dtype=torch.float32))
    model.train()
    for _ in range(epochs):
        optimizer.zero_grad(set_to_none=True)
        loss = loss_fn(model(x, lengths), y)
        if not torch.isfinite(loss):
            raise RuntimeError("Non-finite training loss")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
    return model.eval(), mean, scale, None
