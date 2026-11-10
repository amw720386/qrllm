# QRL1 wire format

The fragment has the form `#m=<transport><body>`. The browser uses the fixed
alphabet `.abcdefghijklmnopqrstuvwxyz`. Token 0 is both the start sentinel
and the end-of-sequence token. Every learned value, including all four scales
of the shipped RNN, lives in the binary envelope.

| Offset | Bytes | Field |
| ---: | ---: | --- |
| 0 | 4 | ASCII `QRL1` |
| 4 | 1 | Architecture, 1 for Transformer or 2 for RNN |
| 5 | 1 | Model width |
| 6 | 1 | Transformer feed-forward width, 0 for RNN |
| 7 | 1 | Maximum token context |
| 8 | 1 | Quantization bits, 2–4 |
| 9 | 1 | Codec, 0 for raw or 1 for zlib-wrapped DEFLATE |
| 10 | Variable | Tensor body |
| End − 4 | 4 | Little-endian CRC-32 of header and stored body |

The decompressed body starts with one little-endian float32 scale per tensor.
Row-major weight codes follow in tensor order, packed continuously from the
least significant bit. The last byte is zero-padded. Code `q` represents
`(q - (2^(bits-1)-1)) * scale` and the largest bit pattern is reserved.
CRC-32 detects corruption. It does not authenticate a model.

- RNN tensor order is token embeddings, input-to-state, state-to-state,
  and state-to-logit. State starts at zero and uses a tanh activation.
- Transformer order is token embeddings, learned positions, query, key,
  value, attention output, feed-forward up, and feed-forward down. One causal
  attention head, RMS normalization and the tied vocabulary projection are
  fixed by the format.
- Transport `A` writes each block of up to four bytes as a zero-padded
  big-endian integer in radix 43. Block lengths 1, 2, 3 and 4 use 2, 3, 5 and 6
  digits. The alphabet is `0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ$*+-./:`.
  An explicit QR alphanumeric segment carries the digits.
- Transport `B` uses unpadded Base64url in QR byte mode.
- Transport `N` prepends byte 1, interprets the result as a big-endian
  integer and writes it in decimal in a QR numeric segment.

The reader validates lengths, bounds the decompressed output, checks CRC,
and rejects invalid scales, codes and padding. The app loads only the fragment.
