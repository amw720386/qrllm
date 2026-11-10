import { alphabet, loadModel, logits, nextToken } from "./runtime.mjs";

const message = document.getElementById("message");
const lab = document.getElementById("lab");
const prefix = document.getElementById("prefix");
const result = document.getElementById("result");
const announcement = document.getElementById("announcement");
let model;
let loadVersion = 0;
let generation = 0;

async function initialize() {
  const version = ++loadVersion;
  generation++;
  model = undefined;
  lab.hidden = true;
  message.hidden = false;
  message.dataset.state = "empty";
  message.textContent = "what are you doing here?";
  if (!location.hash) return;

  message.dataset.state = "loading";
  message.textContent = "";
  try {
    const loaded = await loadModel(location.hash);
    if (version !== loadVersion) return;
    model = loaded;
    message.hidden = true;
    lab.hidden = false;
    prefix.maxLength = model.context - 1;
    prefix.value = "";
    generate();
  } catch {
    if (version !== loadVersion) return;
    message.dataset.state = "invalid";
    message.textContent = "Couldn’t load the model.";
  }
}

async function generate() {
  const run = ++generation;
  const activeModel = model;
  let name = prefix.value.toLowerCase();
  const ids = [0, ...Array.from(name, char => alphabet.indexOf(char))];
  result.dataset.state = "generating";
  result.value = name;
  announcement.textContent = "";
  while (ids.length < activeModel.context && run === generation) {
    const scores = logits(activeModel, ids);
    if (!name) scores[0] = -Infinity;
    const token = nextToken(scores, 0.85);
    if (token === 0) break;
    name += alphabet[token];
    result.value = name[0].toUpperCase() + name.slice(1);
    ids.push(token);
    await new Promise(resolve => setTimeout(resolve, 38));
  }
  if (run !== generation) return;
  result.value = name[0].toUpperCase() + name.slice(1);
  result.dataset.state = "ready";
  announcement.textContent = result.value;
}

document.getElementById("generate-form").addEventListener("submit", event => {
  event.preventDefault();
  if (model) generate();
});
window.addEventListener("hashchange", initialize);
initialize();
