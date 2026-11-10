"""Train small character LMs and export their independently decodable state."""

import argparse
import csv
import hashlib
import json
from pathlib import Path
import random
import time

import torch
from torch import nn
from torch.nn import functional as F

from wire import ALPHABET, serialize, shapes

ROOT = Path(__file__).resolve().parents[1]
CONTEXT = 16
SEED = 5249
DATA = ROOT / "data/yob2020.txt"
CANDIDATES = {
    "transformer-12-i4": (1, 12, 24, 4),
    "transformer-16-i4": (1, 16, 24, 4),
    "transformer-24-i3": (1, 24, 24, 3),
    "transformer-24-i2": (1, 24, 24, 2),
    "rnn-24-i4": (2, 24, 0, 4),
    "rnn-32-i3": (2, 32, 0, 3),
}


def load_split():
    names = sorted({
        name.lower() for name, _, _ in csv.reader(DATA.read_text(encoding="ascii").splitlines())
        if 2 <= len(name) < CONTEXT and name.isascii() and name.isalpha()
    })
    random.Random(SEED).shuffle(names)
    fifth = len(names) // 5
    return names[2 * fifth:], names[:fifth], names[fifth:2 * fifth]


def quantize(weight, bits):
    levels = (1 << (bits - 1)) - 1
    scale = (weight.detach().square().mean().sqrt() * (2.5 / levels)).clamp_min(1e-8)
    codes = (weight.detach() / scale).round().clamp(-levels, levels)
    return scale, codes, codes * scale


class Model(nn.Module):
    def __init__(self, arch, width, ff, bits):
        super().__init__()
        self.arch, self.width, self.ff, self.bits = arch, width, ff, bits
        self.weights = nn.ParameterList([
            nn.Parameter(torch.randn(rows, cols) * (cols ** -0.5))
            for rows, cols in shapes(arch, width, ff, CONTEXT)
        ])
        if arch == 2:
            # Register no extra trainable state. functional_call supplies both matrices.
            self.rnn = nn.RNN(width, width, bias=False, batch_first=True, device="meta")

    def forward(self, tokens, qat=False, weights=None):
        if weights is None:
            weights = self.weights
            if qat:
                weights = [w + (quantize(w, self.bits)[2] - w).detach() for w in weights]
        if self.arch == 2:
            embed, ih, hh, output = weights
            x, _ = torch.func.functional_call(
                self.rnn, {"weight_ih_l0": ih, "weight_hh_l0": hh}, (embed[tokens],)
            )
            return F.linear(x, output)
        embed, pos, q, k, v, proj, up, down = weights
        x = embed[tokens] + pos[:tokens.shape[1]]
        norm = F.rms_norm(x, (self.width,), eps=1e-5)
        attn = F.scaled_dot_product_attention(
            F.linear(norm, q)[:, None],
            F.linear(norm, k)[:, None],
            F.linear(norm, v)[:, None],
            is_causal=True,
        )[:, 0]
        x = x + F.linear(attn, proj)
        norm = F.rms_norm(x, (self.width,), eps=1e-5)
        x = x + F.linear(F.relu(F.linear(norm, up)), down)
        return F.linear(F.rms_norm(x, (self.width,), eps=1e-5), embed)

    def export(self):
        triples = [quantize(weight, self.bits) for weight in self.weights]
        scales, codes, matrices = zip(*triples)
        levels = (1 << (self.bits - 1)) - 1
        packed_codes = torch.cat([code.flatten() for code in codes]) + levels
        payload, raw_bytes = serialize(
            self.arch, self.width, self.ff, CONTEXT, self.bits,
            [scale.item() for scale in scales], packed_codes.int().tolist(),
        )
        return payload, raw_bytes, matrices


def batches(names):
    inputs, targets = [], []
    for name in names:
        ids = [ALPHABET.index(char) for char in name]
        inputs.append(([0] + ids + [0] * CONTEXT)[:CONTEXT])
        targets.append((ids + [0] + [-100] * CONTEXT)[:CONTEXT])
    return torch.tensor(inputs), torch.tensor(targets)


@torch.no_grad()
def evaluate(model, examples, weights=None):
    total = 0
    inputs, targets = examples
    for x, y in zip(inputs.split(256), targets.split(256)):
        logits = model(x, weights=weights)
        total += F.cross_entropy(logits.flatten(0, 1), y.flatten(), reduction="sum").item()
    return total / targets.ne(-100).sum().item()


@torch.no_grad()
def fixtures(model, matrices):
    cases = []
    for prefix in ("", "al", "th", "zo", "alexandria", "zyzzyvablablabl"):
        ids = [0] + [ALPHABET.index(char) for char in prefix]
        logits = model(torch.tensor([ids]), weights=matrices)[0]
        cases.append({"ids": ids, "logits": logits.tolist()})
    return {
        "matrices": [matrix.flatten().tolist() for matrix in matrices],
        "cases": cases,
    }


def run(name, steps, splits, directory):
    started = time.perf_counter()
    torch.manual_seed(SEED)
    model = Model(*CANDIDATES[name])
    optimizer = torch.optim.AdamW(model.weights, lr=0.008)
    train, validation, test = splits
    inputs, targets = train
    best = None
    for step in range(steps):
        qat = step >= steps // 3
        progress = step / steps
        optimizer.param_groups[0]["lr"] = 0.008 * (1 - 0.85 * progress)
        indices = torch.randint(len(inputs), (128,))
        logits = model(inputs[indices], qat=qat)
        loss = F.cross_entropy(logits.flatten(0, 1), targets[indices].flatten())
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if qat and ((step + 1) % 250 == 0 or step + 1 == steps):
            payload, raw_bytes, matrices = model.export()
            val_loss = evaluate(model, validation, matrices)
            if best is None or val_loss < best[0]:
                best = (val_loss, payload, raw_bytes, matrices, step + 1, evaluate(model, test))
    val_loss, payload, raw_bytes, matrices, checkpoint, float_test = best
    (directory / f"{name}.qrl").write_bytes(payload)
    (directory / f"{name}.json").write_text(json.dumps(fixtures(model, matrices)), encoding="utf-8")
    result = {
        "model": name, "parameters": sum(w.numel() for w in model.weights),
        "bytes": len(payload), "packed_bytes": raw_bytes, "step": checkpoint,
        "validation_nll": val_loss, "test_nll": evaluate(model, test, matrices),
        "float_test_nll": float_test,
        "seconds": round(time.perf_counter() - started, 2),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }
    print(json.dumps(result), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("models", nargs="*", choices=None)
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--out", type=Path, default=ROOT / "work")
    args = parser.parse_args()
    if args.steps < 1 or set(args.models) - CANDIDATES.keys():
        parser.error(f"Provide positive steps and model names from {', '.join(CANDIDATES)}")
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    args.out.mkdir(parents=True, exist_ok=True)
    names = load_split()
    print("Train / validation / test names:", *(len(part) for part in names), flush=True)
    splits = [batches(part) for part in names]
    results = [run(name, args.steps, splits, args.out) for name in (args.models or CANDIDATES)]
    (args.out / "training.json").write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
