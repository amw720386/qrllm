export const alphabet = ".abcdefghijklmnopqrstuvwxyz";
const radix43 = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ$*+-./:";

function decodeFragment(hash) {
  if (!hash.startsWith("#m=") || hash.length > 7200) throw new Error("Invalid model link.");
  const encoded = hash.slice(4);
  switch (hash[3]) {
    case "A": {
      const sizes = { 2: 1, 3: 2, 5: 3, 6: 4 };
      const bytes = [];
      for (let offset = 0; offset < encoded.length; offset += 6) {
        const block = encoded.slice(offset, offset + 6);
        const size = sizes[block.length];
        let value = 0;
        for (const char of block) {
          const digit = radix43.indexOf(char);
          if (digit === -1) throw new Error("Invalid radix-43 digit.");
          value = value * 43 + digit;
        }
        if (!size || value >= 256 ** size) throw new Error("Invalid radix-43 block.");
        for (let i = size - 1; i >= 0; i--) bytes.push((value >>> (i * 8)) & 255);
      }
      return Uint8Array.from(bytes);
    }
    case "B": {
      if (!/^[A-Za-z0-9_-]+$/.test(encoded)) throw new Error("Invalid Base64url model.");
      const binary = atob(encoded.replaceAll("-", "+").replaceAll("_", "/"));
      const canonical = btoa(binary).replaceAll("+", "-").replaceAll("/", "_").replace(/=+$/, "");
      if (canonical !== encoded) throw new Error("Noncanonical Base64url model.");
      return Uint8Array.from(binary, char => char.charCodeAt(0));
    }
    case "N": {
      if (!/^[1-9][0-9]*$/.test(encoded)) throw new Error("Invalid decimal model.");
      const hex = BigInt(encoded).toString(16);
      if (hex[0] !== "1" || hex.length % 2 !== 1) throw new Error("Invalid decimal model marker.");
      const body = hex.slice(1).match(/../g) ?? [];
      return Uint8Array.from(body, byte => Number.parseInt(byte, 16));
    }
    default:
      throw new Error("Unknown model link encoding.");
  }
}

function crc32(bytes) {
  let crc = 0xffffffff;
  for (const byte of bytes) {
    crc ^= byte;
    for (let bit = 0; bit < 8; bit++) crc = (crc >>> 1) ^ (0xedb88320 & -(crc & 1));
  }
  return (crc ^ 0xffffffff) >>> 0;
}

function matrixShapes(arch, width, ff, context) {
  if (arch === 1 && ff >= 4 && ff <= 64) {
    return [[27, width], [context, width], ...Array(4).fill([width, width]), [ff, width], [width, ff]];
  }
  if (arch === 2 && ff === 0) return [[27, width], [width, width], [width, width], [27, width]];
  throw new Error("Unsupported model architecture.");
}

async function inflate(bytes, expected) {
  if (!globalThis.DecompressionStream) throw new Error("Open the model in a newer browser.");
  const output = new Uint8Array(expected);
  const reader = new Blob([bytes]).stream().pipeThrough(new DecompressionStream("deflate")).getReader();
  let offset = 0;
  try {
    while (true) {
      const { value, done } = await reader.read();
      if (done) break;
      if (offset + value.length > expected) {
        await reader.cancel();
        throw new Error("Model expands beyond its declared size.");
      }
      output.set(value, offset);
      offset += value.length;
    }
  } finally {
    reader.releaseLock();
  }
  if (offset !== expected) throw new Error("Incomplete model tensors.");
  return output;
}

export async function loadModel(hash) {
  const bytes = decodeFragment(hash);
  if (bytes.length < 14 || bytes.length > 2953 || String.fromCharCode(...bytes.subarray(0, 4)) !== "QRL1") {
    throw new Error("Unrecognized model format.");
  }
  const end = bytes.length - 4;
  if (new DataView(bytes.buffer).getUint32(end, true) !== crc32(bytes.subarray(0, end))) {
    throw new Error("Model checksum failed. Rescan or paste the complete link.");
  }
  const [arch, width, ff, context, bits, codec] = bytes.subarray(4, 10);
  if (width < 4 || width > 64 || context < 2 || context > 32 || bits < 2 || bits > 4 || codec > 1) {
    throw new Error("Unsupported model settings.");
  }
  const shapes = matrixShapes(arch, width, ff, context);
  const parameters = shapes.reduce((sum, [rows, cols]) => sum + rows * cols, 0);
  const scalesLength = 4 * shapes.length;
  const expected = scalesLength + Math.ceil(parameters * bits / 8);
  const body = bytes.subarray(10, end);
  const raw = codec === 1 ? await inflate(body, expected) : body;
  if (raw.length !== expected) throw new Error("Incorrect tensor length.");
  const remainder = (parameters * bits) % 8;
  if (remainder && raw[raw.length - 1] >>> remainder) throw new Error("Invalid tensor padding.");
  const view = new DataView(raw.buffer, raw.byteOffset, raw.byteLength);
  const levels = (1 << (bits - 1)) - 1;
  let bitOffset = 0;
  const tensors = shapes.map(([rows, cols], index) => {
    const scale = view.getFloat32(4 * index, true);
    if (!Number.isFinite(scale) || scale <= 0 || scale > 10000) throw new Error("Invalid quantization scale.");
    const values = new Float32Array(rows * cols);
    for (let i = 0; i < values.length; i++, bitOffset += bits) {
      const at = scalesLength + (bitOffset >>> 3);
      const code = ((raw[at] | ((raw[at + 1] ?? 0) << 8)) >>> (bitOffset & 7)) & ((1 << bits) - 1);
      if (code > 2 * levels) throw new Error("Invalid quantization code.");
      values[i] = (code - levels) * scale;
    }
    return values;
  });
  return { arch, width, context, bits, parameters, tensors, byteLength: bytes.length, bytes };
}

function linear(matrix, x) {
  const y = new Float32Array(matrix.length / x.length);
  for (let i = 0; i < y.length; i++) {
    let sum = 0;
    for (let j = 0; j < x.length; j++) sum += matrix[i * x.length + j] * x[j];
    y[i] = sum;
  }
  return y;
}

function rms(x) {
  let squares = 0;
  for (const value of x) squares += value * value;
  const inverse = 1 / Math.sqrt(squares / x.length + 1e-5);
  return x.map(value => value * inverse);
}

function add(left, right) {
  return left.map((value, index) => value + right[index]);
}

function softmax(scores, temperature = 1) {
  let maximum = -Infinity;
  for (const score of scores) maximum = Math.max(maximum, score);
  const weights = Float64Array.from(scores, score => Math.exp((score - maximum) / temperature));
  const sum = weights.reduce((total, weight) => total + weight, 0);
  return weights.map(weight => weight / sum);
}

function transformer(model, ids) {
  const { width, tensors: [embed, pos, q, k, v, proj, up, down] } = model;
  const inputs = Array.from(ids, (id, i) => add(
    embed.subarray(id * width, (id + 1) * width),
    pos.subarray(i * width, (i + 1) * width),
  ));
  const normalized = inputs.map(rms);
  const last = inputs.length - 1;
  const query = linear(q, normalized[last]);
  const scores = normalized.map(input => {
    const key = linear(k, input);
    let score = 0;
    for (let i = 0; i < width; i++) score += query[i] * key[i];
    return score / Math.sqrt(width);
  });
  const attention = softmax(scores);
  const mix = new Float32Array(width);
  for (let t = 0; t < normalized.length; t++) {
    const value = linear(v, normalized[t]);
    for (let i = 0; i < width; i++) mix[i] += attention[t] * value[i];
  }
  const x = add(inputs[last], linear(proj, mix));
  const hidden = linear(up, rms(x)).map(value => Math.max(0, value));
  return linear(embed, rms(add(x, linear(down, hidden))));
}

function recurrent(model, ids) {
  const { width, tensors: [embed, ih, hh, output] } = model;
  let state = new Float32Array(width);
  for (const id of ids) {
    state = add(linear(ih, embed.subarray(id * width, (id + 1) * width)), linear(hh, state)).map(Math.tanh);
  }
  return linear(output, state);
}

export function logits(model, ids) {
  if (!ids.length || ids.length > model.context || Array.from(ids).some(id => !Number.isInteger(id) || id < 0 || id >= 27)) {
    throw new Error("Token sequence is outside the model context.");
  }
  return model.arch === 1 ? transformer(model, ids) : recurrent(model, ids);
}

export function nextToken(scores, temperature, random = Math.random) {
  if (!Number.isFinite(temperature) || temperature <= 0) throw new Error("Invalid sampling temperature.");
  const distribution = softmax(scores, temperature);
  let sample = random();
  for (let i = 0; i < distribution.length; i++) {
    sample -= distribution[i];
    if (sample < 0) return i;
  }
  return distribution.findLastIndex(probability => probability > 0);
}
