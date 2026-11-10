# Feasibility

Version 40-L has 2,953 bytes of single-segment binary capacity. At M, Q and H
the corresponding ceilings are 2,331, 1,663 and 1,273 bytes. The 33-character
HTTPS prefix (including the fragment marker) and QR segment headers take part of that budget.
At 40-Q, the working estimates are 1.2 KB for Base64url in byte mode and 1.57 KB
for URL-safe radix 43 in a separate alphanumeric segment. A numeric segment
can approach 1.62 KB, but yields nearly twice as many URL characters as radix
43. Symbol density and URL length both matter for a phone-camera demo.
[DENSO WAVE capacity tables](https://www.qrcode.com/en/about/version.html)

A width-16 decoder with 16 learned positions, a width-24 MLP, one attention
head and tied token embeddings needs 2,480 parameters. At four bits per
weight, the packed state starts at 1,240 bytes. Eight float32 scales and the
14-byte envelope put it at 1,286 bytes before entropy coding. A recurrent
decoder trades attention and positional embeddings for a shared recurrent
matrix. Compare them on held-out names with a bigram and a budgeted trigram.
Prefer a model that leaves room for Q correction and a less dense symbol.

## Prior art

- **PicoGPT** ports MicroGPT to dependency-free JavaScript. Its exporter
  gzips minified HTML, Base64-encodes it and includes a
  `DecompressionStream` bootstrap in a `data:text/html` URI. The version
  40-L symbol carries training and inference code. It trains after opening.
  QRLLM uses the same browser decompression primitive for packed pretrained
  state and serves a separate runtime over HTTPS.
  [PicoGPT and exporter](https://github.com/Kuberwastaken/picogpt/blob/main/qrgen.py)
- **The Backdooms** compresses a self-contained browser game into a QR code.
  Its code-as-payload approach inspired PicoGPT's exporter.
  [The Backdooms](https://github.com/Kuberwastaken/backdooms)
- **Snake in a QR code** uses Crinkler to shrink a Windows executable to
  around 1.4 KB. A binary-capable reader extracts the program.
  [Snake in a QR code](https://mattkc.com/etc/snakeqr/)
- **stream-finetune** demonstrates a learned classifier carried in a QR URL,
  with local inference from the fragment. Its larger NF4 trainer is separate
  work. QRLLM uses the same transport idea for an autoregressive neural model.
  [stream-finetune](https://github.com/byjameson/stream-finetune)
- **MicroGPT** shows that a character model can learn names with a tiny
  one-layer Transformer, a from-scratch autodiff engine and autoregressive
  sampling. QRLLM uses PyTorch training and budgets the serialized weights
  and QR transport together.
  [MicroGPT](https://karpathy.github.io/2026/02/12/microgpt/)
