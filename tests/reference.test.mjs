import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";
import { loadModel, logits } from "../dist/runtime.mjs";

const demo = new URL("../demo/", import.meta.url);
const payload = readFileSync(new URL("model.qrl", demo));
const link = new URL(readFileSync(new URL("model-url.txt", demo), "ascii").trim());
const reference = JSON.parse(readFileSync(new URL("reference.json", demo), "utf8"));

test("every dequantized tensor and all prefix logits match PyTorch", async () => {
  const model = await loadModel(link.hash);
  assert.deepEqual(Buffer.from(model.bytes), payload);
  // The signed integer format has one representation of zero.
  assert.deepEqual(model.tensors.map(tensor => Array.from(tensor)), reference.matrices.map(tensor => tensor.map(value => value + 0)));
  let maxError = 0;
  for (const { ids, logits: expected } of reference.cases) {
    for (let step = 0; step < ids.length; step++) {
      const actual = logits(model, ids.slice(0, step + 1));
      for (let token = 0; token < actual.length; token++) {
        maxError = Math.max(maxError, Math.abs(actual[token] - expected[step][token]));
      }
    }
  }
  console.log("Maximum absolute logit error", maxError);
  assert.ok(maxError < 2e-5);
});

test("independent transport modes decode the same bytes", async () => {
  const base64 = "#m=B" + payload.toString("base64url");
  const decimal = "#m=N" + BigInt("0x01" + payload.toString("hex")).toString(10);
  for (const hash of [base64, decimal]) {
    const model = await loadModel(hash);
    assert.deepEqual(Buffer.from(model.bytes), payload);
  }
});

test("missing, malformed, oversized and corrupt models reject", async () => {
  const damaged = Buffer.from(payload);
  damaged[15] ^= 1;
  for (const bad of ["", "#other=abc", "#m=Aa", "#m=N001", "#m=Babc=", "#m=A" + "0".repeat(7200)]) {
    await assert.rejects(loadModel(bad));
  }
  await assert.rejects(loadModel("#m=B" + damaged.toString("base64url")), /checksum/);
});
