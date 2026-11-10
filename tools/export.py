"""Benchmark QR transports, choose a model and export a verified PNG."""

import argparse
import hashlib
import json
from pathlib import Path
from urllib.parse import quote, urlsplit
import zlib

import brotli
import segno
from segno import consts
from PIL import Image
import zxingcpp

from wire import ORIGIN, decode, fragment

ROOT = Path(__file__).resolve().parents[1]
BASE45 = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ $%*+-./:"


def escaped_base45(payload):
    result = []
    for i in range(0, len(payload), 2):
        block = payload[i:i + 2]
        value = int.from_bytes(block, "big")
        for _ in range(len(block) + 1):
            value, digit = divmod(value, 45)
            result.append(BASE45[digit])
    return "#m=" + quote("".join(result), safe="$*+-./:")


def qr_options(payload, origin):
    for name, text, mode in [
        ("base64url", fragment(payload, "B"), None),
        ("base45-escaped", escaped_base45(payload), "alphanumeric"),
        ("radix43", fragment(payload, "A"), "alphanumeric"),
        ("decimal", fragment(payload, "N"), "numeric"),
    ]:
        url = origin + text
        if mode:
            boundary = len(origin) + (3 if name == "base45-escaped" else 4)
            segment_mode = consts.MODE_NUMERIC if mode == "numeric" else consts.MODE_ALPHANUMERIC
            content = [(url[:boundary], consts.MODE_BYTE), (url[boundary:], segment_mode)]
            digits = len(url) - boundary
            per_group, group_bits = (3, 10) if mode == "numeric" else (2, 11)
            groups, tail = divmod(digits, per_group)
            # Versions 27–40 use these count field widths.
            bits = 20 + 8 * boundary + 4 + (14 if mode == "numeric" else 13)
            bits += groups * group_bits + ((0, 4, 7) if mode == "numeric" else (0, 6))[tail]
        else:
            content = url
            bits = 20 + 8 * len(url)
        symbols = {}
        for level in "LMQH":
            try:
                symbols[level] = segno.make_qr(content, error=level, boost_error=False)
            except segno.DataOverflowError:
                pass
        yield name, url, bits, symbols


def read_qr(image):
    result = zxingcpp.read_barcode(image, formats=zxingcpp.BarcodeFormat.QRCode)
    return result.text if result else None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work", type=Path, default=ROOT / "work")
    parser.add_argument("--out", type=Path, default=ROOT / "demo")
    parser.add_argument("--origin", default=ORIGIN)
    parser.add_argument("--model", help="Export this trained model instead of selecting by validation loss")
    parser.add_argument("--max-version", type=int, default=35)
    parser.add_argument("--max-url-length", type=int, default=2048)
    parser.add_argument("--ecc", choices="LMQH", default="Q")
    args = parser.parse_args()
    origin = args.origin.rstrip("/") + "/"
    parsed = urlsplit(origin)
    if not origin.isascii() or parsed.scheme != "https" or not parsed.netloc or (
        parsed.username or parsed.password or parsed.path != "/" or parsed.query or parsed.fragment
    ):
        parser.error("Use an HTTPS origin without a path, query, credentials or fragment.")
    if not 1 <= args.max_version <= 40:
        parser.error("QR version must be between 1 and 40.")
    training = json.loads((args.work / "training.json").read_text(encoding="utf-8"))
    reports, eligible = [], []
    for result in training:
        name = result["model"]
        payload = (args.work / f"{name}.qrl").read_bytes()
        if hashlib.sha256(payload).hexdigest() != result["sha256"]:
            raise ValueError(f"Unexpected payload for {name}")
        packed = payload[10:-4]
        if payload[9] == 1:
            packed = zlib.decompress(packed)
        report = {
            **result, "brotli_bytes": 14 + len(brotli.compress(packed)), "encodings": {},
        }
        for mode, url, bits, symbols in qr_options(payload, origin):
            report["encodings"][mode] = {
                "characters": len(url), "bits_v27_plus": bits,
                "versions": {level: symbol.version for level, symbol in symbols.items()},
            }
            symbol = symbols.get(args.ecc)
            if (
                mode != "base45-escaped" and symbol and symbol.version <= args.max_version
                and len(url) <= args.max_url_length and (not args.model or name == args.model)
            ):
                eligible.append((result["validation_nll"], symbol.version, len(url), name, mode, url, symbol))
        reports.append(report)
    if not eligible:
        parser.error("No trained candidate fits the requested QR and URL bounds.")
    _, version, _, name, mode, url, symbol = min(eligible, key=lambda item: item[:3])
    payload = (args.work / f"{name}.qrl").read_bytes()
    args.out.mkdir(parents=True, exist_ok=True)
    png = args.out / "qrllm.png"
    symbol.save(png, scale=12, border=6, dark="black", light="white")
    with Image.open(png) as image:
        recovered = read_qr(image)
        pixels = image.width
    if recovered != url or decode("#" + urlsplit(recovered).fragment) != payload:
        raise RuntimeError("QR failed exact URL and payload round trip")
    (args.out / "model-url.txt").write_text(url + "\n", encoding="ascii")
    (args.out / "open.md").write_text(
        f"# Open QRLLM\n\n[Open the complete model link]({url})\n\n"
        "The destination requires the static site to be deployed. The fragment carries the model.\n",
        encoding="ascii",
    )
    (args.out / "model.qrl").write_bytes(payload)
    (args.out / "reference.json").write_bytes((args.work / f"{name}.json").read_bytes())
    selected = {
        "model": name, "encoding": mode, "version": version, "ecc": args.ecc,
        "pixels": pixels, "quiet_zone_modules": 6, "url_characters": len(url),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }
    (args.out / "measurements.json").write_text(
        json.dumps({"selected": selected, "models": reports}, indent=2) + "\n", encoding="utf-8",
    )
    print(json.dumps(selected, indent=2))


if __name__ == "__main__":
    main()
