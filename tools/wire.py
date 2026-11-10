"""Wire layout shared with dist/runtime.mjs. No learned constants."""

import base64
import math
import struct
import zlib

ALPHABET = ".abcdefghijklmnopqrstuvwxyz"
RADIX43 = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ$*+-./:"
ORIGIN = "https://qrllm.ahamedwajibu.com/"


def shapes(arch, width, ff, context):
    if arch == 1:
        return [(27, width), (context, width)] + [(width, width)] * 4 + [
            (ff, width), (width, ff)
        ]
    if arch == 2:
        return [(27, width), (width, width), (width, width), (27, width)]
    raise ValueError("Unknown architecture")


def pack(codes, bits):
    output = bytearray((len(codes) * bits + 7) // 8)
    for i, code in enumerate(codes):
        byte, shift = divmod(i * bits, 8)
        value = int(code) << shift
        output[byte] |= value & 255
        if shift + bits > 8:
            output[byte + 1] |= value >> 8
    return bytes(output)


def serialize(arch, width, ff, context, bits, scales, codes):
    packed = pack(codes, bits)
    raw = struct.pack(f"<{len(scales)}f", *scales) + packed
    compressed = zlib.compress(raw, 9)
    codec, body = (1, compressed) if len(compressed) < len(raw) else (0, raw)
    payload = b"QRL1" + bytes((arch, width, ff, context, bits, codec)) + body
    return payload + struct.pack("<I", zlib.crc32(payload)), len(raw) + 14


def radix43_encode(data):
    groups = []
    for offset in range(0, len(data), 4):
        block = data[offset:offset + 4]
        value = int.from_bytes(block, "big")
        digits = []
        for _ in range(math.ceil(len(block) * 8 / math.log2(43))):
            value, digit = divmod(value, 43)
            digits.append(RADIX43[digit])
        groups.append("".join(reversed(digits)))
    return "".join(groups)


def radix43_decode(encoded):
    sizes = {2: 1, 3: 2, 5: 3, 6: 4}
    output = bytearray()
    for offset in range(0, len(encoded), 6):
        block = encoded[offset:offset + 6]
        value = 0
        for char in block:
            value = value * 43 + RADIX43.index(char)
        output.extend(value.to_bytes(sizes[len(block)], "big"))
    return bytes(output)


def fragment(payload, mode):
    if mode == "B":
        encoded = base64.urlsafe_b64encode(payload).decode("ascii").rstrip("=")
    elif mode == "A":
        encoded = radix43_encode(payload)
    elif mode == "N":
        # A leading nonzero byte preserves any leading zero bytes.
        encoded = str(int.from_bytes(b"\x01" + payload, "big"))
    else:
        raise ValueError("Unknown transport encoding")
    return f"#m={mode}{encoded}"


def decode(fragment_text):
    if not fragment_text.startswith("#m="):
        raise ValueError("Missing model fragment")
    mode, encoded = fragment_text[3], fragment_text[4:]
    if mode == "A":
        payload = radix43_decode(encoded)
    elif mode == "B":
        payload = base64.urlsafe_b64decode(encoded + "=" * (-len(encoded) % 4))
    elif mode == "N":
        value = int(encoded)
        raw = value.to_bytes((value.bit_length() + 7) // 8, "big")
        if not raw.startswith(b"\x01"):
            raise ValueError("Invalid decimal transport")
        payload = raw[1:]
    else:
        raise ValueError("Unknown transport encoding")
    if fragment(payload, mode) != fragment_text:
        raise ValueError("Noncanonical model fragment")
    return payload
