import { chromium } from "playwright";
import { readFileSync } from "node:fs";

const modelLink = process.argv[2] ?? "demo/model-url.txt";
const origin = process.argv[3] ?? "http://127.0.0.1:4173";
const hash = new URL(readFileSync(modelLink, "ascii").trim()).hash;
const browser = await chromium.launch();
try {
  const page = await browser.newPage();
  await page.goto(origin);
  const result = await page.evaluate(async fragment => {
    const { alphabet, loadModel, logits, nextToken } = await import("/runtime.mjs");
    const times = [];
    let model;
    for (let i = 0; i < 31; i++) {
      const start = performance.now();
      model = await loadModel(fragment);
      times.push(performance.now() - start);
    }
    const prefix = [0, 1, 12, 5, 24, 1, 14];
    const token = () => nextToken(logits(model, prefix), 0.85);
    for (let i = 0; i < 1000; i++) token();
    const speedStart = performance.now();
    for (let i = 0; i < 10000; i++) token();
    const tokensPerSecond = 1e7 / (performance.now() - speedStart);
    let seed = 39281;
    const random = () => {
      seed = (Math.imul(seed, 1664525) + 1013904223) >>> 0;
      return seed / 2 ** 32;
    };
    const samples = {};
    for (const start of ["", "el", "mar", "zo", "th"]) {
      samples[start] = Array.from({ length: 5 }, () => {
        let text = start;
        const ids = [0, ...Array.from(start, char => alphabet.indexOf(char))];
        while (ids.length < model.context) {
          const scores = logits(model, ids);
          if (!text) scores[0] = -Infinity;
          const id = nextToken(scores, 0.85, random);
          if (id === 0) break;
          ids.push(id);
          text += alphabet[id];
        }
        return text;
      });
    }
    return {
      firstLoadMs: times.shift(),
      medianWarmLoadMs: times.sort((a, b) => a - b)[15],
      tokensPerSecond, samples,
    };
  }, hash);
  console.log(JSON.stringify({ browser: browser.version(), ...result }, null, 2));
} finally {
  await browser.close();
}
