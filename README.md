# QRLLM

QRLLM is a pretrained character-level neural language model for generating
given names. The QR code carries all 2,448 weights, four quantization scales
and the model configuration in a 1,174-byte payload. The website reads the
payload from the URL fragment and generates locally in JavaScript.

[Download the QR PNG](demo/qrllm.png) · [Open the full model link](demo/open.md) · [Deployment instructions](docs/deploy.md)

The link in the QR targets `https://qrllm.ahamedwajibu.com/`. It requires the
static site to be deployed at that origin. The bare page says “what are you
doing here?” A model link opens a starting-letter field and a generated name.
Press Enter to sample again, optionally with a prefix.

## Try it locally

Run the static interface with Python 3.10 or newer. Development and testing
use Python and Node. Visitors only need a modern browser.

```sh
python -m http.server 4173 --bind 127.0.0.1 --directory dist
```

Copy the URL from [demo/model-url.txt](demo/model-url.txt) and replace its
origin with `http://127.0.0.1:4173`, preserving the entire `/#m=…` suffix.
Remove the fragment to see the empty state. Restore it after switching the
tab offline to see the model initialize and generate without HTTP access.

## Model and training

The exported network embeds 27 symbols into 24 dimensions, applies a
single-layer tanh recurrence and projects to next-character logits. The
context is 16 tokens including the start symbol. Training minimizes
next-character cross entropy on distinct lowercase given names from the
2020 U.S. Social Security data. The seeded split has 17,169 train, 5,722
validation and 5,722 test names. Each name counts once regardless of
frequency. The model generates short name spellings in the fixed a–z alphabet.
It can repeat familiar names or produce unusual spellings.

The training script uses symmetric per-tensor fake quantization with a
straight-through estimator after the first third of training. The exporter
packs the learned integer codes, stores the scales as float32 and uses zlib
when it saves bytes. [Wire format](docs/format.md)

```sh
python -m pip install -r requirements.txt
python tools/train.py
python tools/export.py
python tools/baseline.py
```

The exporter selects by validation loss among candidates that fit Q error
correction at version 35 or below and a 2,048-character URL ceiling. It reads
the PNG back through ZXing-C++ and checks the full URL and binary payload
before writing the demo artifacts. For a focused rebuild, run
`python tools/train.py rnn-24-i4-long` followed by `python tools/export.py`.
For the full experiment, run training without model arguments. Exact
cross-platform hashes can depend on floating-point kernels and zlib versions.

The vendored [data file](data/yob2020.txt) is the CC0 national SSA 2020
file from mirror commit `0b1a1316457d55447d1a1f7bfe57ee15e53cc2f0`.
Its SHA-256 with LF line endings is
`efdb2e63254bfb918aeb8aa102038273f1124f4972c9fe81ca6f7bd4303ce4ca`.
[SSA data catalog](https://catalog.data.gov/dataset/baby-names-from-social-security-card-applications-national-data) · [Pinned mirror](https://github.com/hackerb9/ssa-baby-names/blob/0b1a1316457d55447d1a1f7bfe57ee15e53cc2f0/raw-data/yob2020.txt)

## Measurements

Loss is in nats per next character, including the end token. Size includes
the neural envelope and scales. The n-gram sizes count their quantized
tables. The sparse trigram stores one 16-bit key and one 8-bit log boost per
exception to its dense bigram table.

| Model | Parameters | Stored bytes | Test NLL |
| --- | ---: | ---: | ---: |
| Dense bigram | 729 table cells | 729 | 2.464 |
| Budgeted sparse trigram | 729 cells + 148 exceptions | 1,173 | 2.386 |
| Transformer 12 INT4 | 1,668 | 865 | 2.266 |
| Transformer 16 INT4 | 2,480 | 1,217 | 2.243 |
| Transformer 24 INT3 | 4,488 | 1,690 | 2.308 |
| Transformer 24 INT2 | 4,488 | 711 | 2.520 |
| RNN 24 INT4, 2,000 steps | 2,448 | 1,184 | 2.204 |
| RNN 32 INT3 | 3,776 | 1,413 | 2.231 |
| **RNN 24 INT4, 5,000 steps** | **2,448** | **1,174** | **2.194** |
| RNN 26 INT4, 5,000 steps | 2,756 | 1,307 | 2.182 |

The shipped RNN reaches **8.97 perplexity** against **10.87** for the
similarly sized trigram. The width-26 run has slightly lower loss and fits
version 37-Q. Its 360-pixel resize failed to decode in ZXing-C++, so the
release uses the version 35-Q symbol. At the same initial 2,000-step RNN
schedule and corpus size, isolated dictionary-word generation reached
2.363 test NLL at 1,177 bytes versus 2.204 for names at 1,184 bytes.
These corpora have different distributions. The result favored names for
this spelling demo. To rerun the word comparison, extract the first
whitespace-separated field of each line from the pinned CMU dictionary and
run `python tools/train.py rnn-24-i4 --text-corpus words.txt --out work/words`.
[Pinned CMUdict source](https://github.com/cmusphinx/cmudict/blob/74790861f652b15e4ac49015a90074ad62a27690/cmudict.dict)

The selected payload takes 1,254 bytes with bit packing alone, 1,174 with
zlib, and 1,147 with Brotli in the offline benchmark. The release uses zlib
for built-in browser decompression. INT2 compresses well, but the candidate
lost more in next-character quality than it saved in storage.

| Transport for the 1,174-byte payload | URL characters | L | M | Q | H |
| --- | ---: | ---: | ---: | ---: | ---: |
| Base64url, byte mode | 1,601 | 29 | 33 | 40 | — |
| Percent-escaped Base45, split | 1,917 | 26 | 30 | 36 | — |
| **URL-safe radix 43, split** | **1,796** | **25** | **29** | **35** | **40** |
| Decimal, split | 2,863 | 25 | 29 | 34 | 39 |

Numbers in L–H are the minimum actual QR versions emitted by Segno. The
release spends 10,003 data bits before terminator and padding. Decimal
saves one version at Q, at the cost of a URL beyond the release's length
ceiling. The PNG is 2,028 × 2,028 pixels with a six-module quiet zone.
[Full measurements](demo/measurements.json) · [Byte budget and prior art](docs/feasibility.md)

Headless Chromium on an Intel i5-1235U loaded the payload in 3.5 ms on the
first call and 0.9 ms median on warm calls. A warmed seven-token forward pass
plus sampling ran at about 22,300 tokens/s. This measures compute, excluding
page load and UI streaming delay. The static directory is roughly 13 KB
before transfer compression. JavaScript with typed arrays met the latency
target. WASM and GPU backends were left out of this implementation.

Seeded samples at temperature 0.85 include **Javisen**, **Gartomi**, **Nadah**,
**Rana**, and **Alj**. Prefix `el` produced **Eleh**, **Elahianna**, **Elih**,
**Elaire**, and **Elizase**. With the preview running, repeat the measurement
with `node tools/benchmark.mjs`.

## Verification

```sh
npm ci
npx playwright install chromium webkit
npm test
```

The tests compare every reconstructed weight and the logits for all
positions in six PyTorch fixtures. The maximum absolute error for the
release was 0.000000954. The browser tests use desktop Chromium, Android
Chromium emulation and iPhone WebKit emulation. They block every network
request after the bare page loads, attach the QR fragment, generate,
remove the fragment, check that the generator disappears, and restore it.
An asset allowlist and content security policy constrain the static app.

ZXing-C++ recovered the exact URL from the original PNG, a 720-pixel browser
screenshot, 1080- and 720-pixel JPEGs, 540-, 360- and 280-pixel downscales,
moderate Gaussian blur, and reduced margins. Physical cameras and X's live
image pipeline still need testing. Browser emulation covers the web app
after a camera hands off the URL. [Physical scan and posting checklist](docs/deploy.md#before-sharing-the-png)

Run `python tools/check_qr.py --matrix` to verify exact URL recovery from
all 14 fitting transport and error-correction combinations in the table.

Code is available under the [MIT license](LICENSE).
