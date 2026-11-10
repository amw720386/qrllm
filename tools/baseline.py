"""Evaluate smoothed, quantized n-grams on the same held-out split."""

from collections import Counter
from itertools import pairwise
import json
import math

import numpy as np

from train import load_split, ROOT
from wire import ALPHABET

VOCAB = len(ALPHABET)


def sequences(names):
    for name in names:
        yield [0, 0, *(ALPHABET.index(char) for char in name), 0]


def bigrams(seqs):
    counts = np.full((VOCAB, VOCAB), 0.25)
    for seq in seqs:
        for prev, char in pairwise(seq[1:]):
            counts[prev, char] += 1
    probs = counts / counts.sum(axis=1, keepdims=True)
    return np.rint(-np.log(probs) * 16).clip(0, 255).astype(np.uint8)


def probabilities(table):
    probs = np.exp(-table.astype(float) / 16)
    return probs / probs.sum(axis=-1, keepdims=True)


def trigram_entries(seqs, bigram):
    counts = Counter((a, b, c) for seq in seqs for a, b, c in zip(seq, seq[1:], seq[2:]))
    totals = Counter()
    for (a, b, _), count in counts.items():
        totals[a, b] += count
    entries = []
    for (a, b, c), count in counts.items():
        p = bigram[b, c]
        smoothed = (count + 15 * p) / (totals[a, b] + 15)
        boost = math.log(smoothed / p)
        if boost > 0:
            key = (a * VOCAB + b) * VOCAB + c
            entries.append((count * boost, key, round(16 * boost)))
    return sorted(entries, reverse=True)


def evaluate(seqs, probs):
    losses = [-math.log(probs[a, b, c]) for seq in seqs for a, b, c in zip(seq, seq[1:], seq[2:])]
    return sum(losses) / len(losses)


def main():
    train, _, test = map(lambda names: list(sequences(names)), load_split())
    budget = (ROOT / "demo/model.qrl").stat().st_size
    packed = bigrams(train)
    bigram = probabilities(packed)
    table = np.tile(bigram, (VOCAB, 1, 1))
    bigram_loss = evaluate(test, table)
    # Each entry stores its flattened trigram key (uint16) and log boost (uint8).
    entries = trigram_entries(train, bigram)[:max(0, (budget - packed.nbytes) // 3)]
    for _, key, boost in entries:
        table.flat[key] *= math.exp(boost / 16)
    table /= table.sum(axis=-1, keepdims=True)
    print(json.dumps({
        "bigram": {"bytes": packed.nbytes, "test_nll": bigram_loss},
        "sparse_trigram": {"bytes": packed.nbytes + 3 * len(entries), "test_nll": evaluate(test, table)},
    }, indent=2))


if __name__ == "__main__":
    main()
