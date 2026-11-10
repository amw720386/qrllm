"""Verify exact QR recovery under controlled raster transformations."""

import argparse
import hashlib
from io import BytesIO
import json
from pathlib import Path
from urllib.parse import urlsplit

from PIL import Image, ImageFilter

from export import ROOT, qr_options, read_qr
from wire import decode


def jpeg(image, quality):
    buffer = BytesIO()
    image.save(buffer, format="JPEG", quality=quality)
    buffer.seek(0)
    return Image.open(buffer).copy()


def resize(image, width):
    return image.resize((width, width), Image.Resampling.LANCZOS)


def stress(image, width, border):
    at_1080 = resize(image, 1080)
    at_720 = resize(image, 720)
    crop_4 = (border - 4) * 12
    crop_2 = (border - 2) * 12
    yield "original PNG", image
    yield "1080px JPEG q75", jpeg(at_1080, 75)
    yield "720px JPEG q55", jpeg(at_720, 55)
    yield "540px display", resize(image, 540)
    yield "360px display", resize(image, 360)
    yield "280px display", resize(image, 280)
    yield "720px JPEG q60 and 0.8px blur", jpeg(at_720.filter(ImageFilter.GaussianBlur(0.8)), 60)
    yield "720px and 4-module margin", resize(image.crop((crop_4, crop_4, width-crop_4, width-crop_4)), 720)
    yield "720px and 2-module margin", resize(image.crop((crop_2, crop_2, width-crop_2, width-crop_2)), 720)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--demo", type=Path, default=ROOT / "demo")
    parser.add_argument("--screenshot", type=Path)
    parser.add_argument("--matrix", action="store_true", help="Check every viable transport and ECC level")
    args = parser.parse_args()
    expected = (args.demo / "model-url.txt").read_text(encoding="ascii").strip()
    payload = (args.demo / "model.qrl").read_bytes()
    report = json.loads((args.demo / "measurements.json").read_text(encoding="utf-8"))
    assert hashlib.sha256(payload).hexdigest() == report["selected"]["sha256"]
    if args.screenshot:
        with Image.open(args.screenshot) as screenshot:
            assert read_qr(screenshot) == expected
        print("Browser screenshot decoded exactly.")
        return
    checks = {}
    with Image.open(args.demo / "qrllm.png") as png:
        image = png.convert("L")
    for label, variant in stress(image, image.width, report["selected"]["quiet_zone_modules"]):
        scanned = read_qr(variant)
        checks[label] = scanned == expected and decode("#" + urlsplit(scanned).fragment) == payload
    print(json.dumps(checks, indent=2))
    # The two smallest widths and the nonstandard quiet zone are exploratory.
    assert all(ok for label, ok in checks.items() if label not in {
        "360px display", "280px display", "720px and 2-module margin",
    })
    if args.matrix:
        for transport, url, _, symbols in qr_options(payload, expected.split("#", 1)[0]):
            for ecc, qr in symbols.items():
                buffer = BytesIO()
                qr.save(buffer, kind="png", scale=6, border=6)
                with Image.open(buffer) as variant:
                    assert read_qr(variant) == url, f"Round trip failed for {transport} v{qr.version}-{ecc}"
                print(f"{transport} v{qr.version}-{ecc}: exact")


if __name__ == "__main__":
    main()
