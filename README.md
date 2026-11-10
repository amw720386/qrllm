# QRLLM

QRLLM is a pretrained character-level neural language model for generating
given names. The QR code carries all 2,448 weights, four quantization scales
and the model configuration in a 1,174-byte payload. The website reads the
payload from the URL fragment and generates locally in JavaScript.

[Open the model][model-link] · [Download the QR PNG](demo/qrllm.png) · [Deploy on Vercel](#deploy-on-vercel)

The link in the QR targets `https://qrllm.ahamedwajibu.com/`. It requires the
static site to be deployed at that origin.

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

## Deploy on Vercel

Import this repository on Vercel and set the production branch to `main`.
The website is already built and the Python tools live outside its root.

| Setting | Value |
| --- | --- |
| Root Directory | `dist` |
| Framework Preset | Other |
| Build Command | Enable Override and leave blank |
| Output Directory | Default, or `.` if a value is required |
| Include files outside the root directory | Off |

Deploy, then add `qrllm.ahamedwajibu.com` in the project's Domains settings.
In your DNS dashboard, add a CNAME named `qrllm` with the exact target Vercel
provides. Once the domain is active, [open the model][model-link] or scan the
PNG. On the same phone, you can open the saved QR image in Photos or Google
Lens. If a link wrapper drops the fragment, paste the full URL from
[`demo/model-url.txt`](demo/model-url.txt) into the browser.
[Vercel build settings](https://vercel.com/docs/builds/configure-a-build#root-directory) · [Custom domains](https://vercel.com/docs/domains/set-up-custom-domain)

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
when it saves bytes. [Wire format](#wire-format)

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

### QR byte budget

Version 40 holds at most 2,953 bytes in binary mode at L error correction,
2,331 at M, 1,663 at Q, or 1,273 at H. The HTTPS prefix and QR segment headers
also need space. [DENSO WAVE capacity tables](https://www.qrcode.com/en/about/version.html)

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
[Full measurements](demo/measurements.json)

### Runtime

Headless Chromium on an Intel i5-1235U loaded the payload in 3.5 ms on the
first call and 0.9 ms median on warm calls. A warmed seven-token forward pass
plus sampling ran at about 22,300 tokens/s. This measures compute, excluding
page load and UI streaming delay.

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
image pipeline still need testing. Before sharing, scan the full-size PNG
from a monitor with the default iPhone and Android cameras, then repeat
against the image served by X in both the feed and expanded view.

Run `python tools/check_qr.py --matrix` to verify exact URL recovery from
all 14 fitting transport and error-correction combinations in the table.

## Wire format

<details>
<summary>QRL1 envelope and encodings</summary>

The fragment starts with `#m=` and a one-character transport marker. The
fixed alphabet is `.abcdefghijklmnopqrstuvwxyz`. Token 0 marks the start and
end of a sequence.

| Offset | Bytes | Field |
| ---: | ---: | --- |
| 0 | 4 | ASCII `QRL1` |
| 4 | 1 | Architecture (1 for Transformer, 2 for RNN) |
| 5 | 1 | Model width |
| 6 | 1 | Transformer feed-forward width (0 for RNN) |
| 7 | 1 | Maximum token context |
| 8 | 1 | Quantization bits (2–4) |
| 9 | 1 | Codec (0 for raw, 1 for zlib-wrapped DEFLATE) |
| 10 | Variable | Tensor body |
| End − 4 | 4 | Little-endian CRC-32 of header and stored body |

The decompressed body starts with one little-endian float32 scale per tensor.
Row-major weight codes follow, packed from the least significant bit and
zero-padded in the last byte. Code `q` represents
`(q - (2^(bits-1)-1)) * scale`. The largest bit pattern is reserved.
CRC-32 detects corruption.

- RNN tensor order is token embeddings, input-to-state, state-to-state,
  and state-to-logit. State starts at zero and uses tanh.
- Transformer order is token embeddings, learned positions, query, key,
  value, attention output, feed-forward up, and feed-forward down. The
  format fixes one causal attention head, RMS normalization and a tied
  vocabulary projection.
- Transport `A` writes each block of up to four bytes as a zero-padded
  big-endian integer in radix 43. Block lengths 1, 2, 3 and 4 use 2, 3, 5 and 6
  digits from `0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ$*+-./:`. The digits
  occupy a separate QR alphanumeric segment.
- Transport `B` uses unpadded Base64url in QR byte mode.
- Transport `N` prepends byte 1, then writes the resulting big-endian
  integer in decimal in a QR numeric segment.

</details>

## Prior art

- PicoGPT ports MicroGPT to JavaScript and fits the compressed training and
  inference program into a QR code. It trains in the browser after scanning.
  QRLLM puts pretrained weights in the QR and serves the runtime separately.
  [PicoGPT exporter](https://github.com/Kuberwastaken/picogpt/blob/main/qrgen.py)
- The Backdooms and Snake in a QR code fit complete programs inside QR
  symbols. [The Backdooms](https://github.com/Kuberwastaken/backdooms) · [Snake in a QR code](https://mattkc.com/etc/snakeqr/)
- stream-finetune demonstrates a classifier with learned weights in a QR
  URL. [stream-finetune](https://github.com/byjameson/stream-finetune)
- MicroGPT demonstrates autoregressive name generation with a one-layer
  character Transformer and a from-scratch training loop.
  [MicroGPT](https://karpathy.github.io/2026/02/12/microgpt/)

Code is available under the [MIT license](LICENSE).

[model-link]: https://qrllm.ahamedwajibu.com/#m=A9C34A70ABZ0A0JSB$$0SAJUGP1DS23PR6507DV5:OK7PLACHK*URLEIN3$9HJ8V+.86CIPDIOL2-WXB9HH2-AJ8-/RGIC/A0OL.TNQO00*LXGDH47K1-2-T70YS18YGXISZH8KZGM2FVLS:TMF+PNVS:N/+1R:XT4XFCO5W0M4.CW42*C5I-JURIS:2NIN1A17NS101QOU-$1IUY/TA3U1RO0FW$O6I9/TW1E127D/AA6NCG5Q9I21ISR3O*0.D90HMZ.-JGK526JE60QOQBL4050L4J7PA02.:OPVASRTC6O7QIMR2BT+T17L75LA7O-ZT27L54Z11742/YF$DI4773$6CI3A+RMM014NZLJLB1RVAHVHT3B+8*6YEDNP4KGS::I:NG*RT3Q9J1AL865JPD7BKVT*KDZB3ZPZK.$13.YH727NAT9AOVJN5JI9O6Y0CWE-E5G14+J6D1*WLPA4QAX36JJH8ECBZWNI7E/*PCA0ZX:U9+ASK0GNEYWZF*5EYLOYEE55CGYGM0O/34W92VW.P4IM$*AQ38LI06N4NVMXOZWZX1I7DO2R884267N440RGA--RE0B+R125E00Q6OBE8J18F9+2HE2M.$YTQNG1L6BV9QH9BO3SZ*1$IMO$8MUW:PBIVDVGC42WH-K3N9RNJQA18JSQ/8NYHP7K-AE8+LATHL*N8EJJX/ZHJ$L.$V75IGI*24XXCQBOF34KLSMGKZR1TV.M4J9OB/0DDH16NNF$5JGKW$D7HAYO*.1X:/JMAX3O.PGGK5GYFMT3WERLA6Q-G889P6EVCOLO4+6K-6JWEL6KDYV+O.IW6BFTHF9SSFDWJZS-7UK97/HG0KEFECMH4TK1XW36C+1UYJLIPY/W3R:V+X5H+TOV4D8R.OJASIORKST+LK4W5UB14-+2A*OGSPWKBS2X./HM+KSCHN6$RIJUIVLU5P152NNB-43243UP4J7S50M93BDCH97WMBP0B05$P/LR0U4/G8LRG+NB47.HFIEYVS3U4TEQ8CJ*B4F7-*ZZDXTUAWK8BP4C8O+6I1OX.A$MQZPCG7FKRRJU8-YT226T+WZLOG$GV7JV2T+NC9KPYEB69/BD7SXSL0O8:RA60CQOVD8B0K123LGOY7BO99VZKL9V4EO0R77:KIMFFLIQDQY4A-6-V19:X02P555Q.PS6HKCG0DAZ2LEAYXE-96::XPB+UJV.A:W.YNC//+2YDZJL.NDR3Y5:IJ-O1PGWTOH6MUU.+08:LYGQK8QWNC5Q+B+DNNNOFBR*0VWL8PK$2WG/H9Z874+5VSGLPWY4220.NKBCD7B7O7B32BFQ$5R/32U77NK-$R6:AX7NK2NXD-B0D+RGJNONEP2:RJPWFFIWH+NPHITW+ZHMZLGGJ7LLDW9JM9+XDN$5W5I*QVV099H9BQJ3TTKBJIQ*HTCN5/FBSP63:54.K:WESW5R5O7*YYFR6D$K79ORG5./LMO0MLMMN*JU3UE:H:B2CH1RFQ/NPV9NQ*+LB55PYNIBXNOCS-83JV059X3738STAOO6D1-0S5L8Y086JEQP8I/EJ-JS-7U09M$34GEEGBQTD:Q4H/2NHSPIFX*.X3KA0UJXGR6BZ.ECVM2WO074NMPDN67XB088A14B.2PK9.9PCLK6RQ+2HC/SIYF2BRN:OKLDG8LBN1ZU7J9+DX9*3.SJAVL4HSRZ6L6XS1JC4WCG9JJ0GBPDKTO7Y9RL2XJV-CEM4MFIT4J0/I7SP59/57B
