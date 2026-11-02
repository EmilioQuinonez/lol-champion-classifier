const drop = document.getElementById("drop");
const input = document.getElementById("file");
const results = document.getElementById("results");
const toolbar = document.querySelector(".toolbar");

const pretty = name => name.split("_").map(w => w[0].toUpperCase() + w.slice(1)).join(" ");
const pct = p => (p * 100).toFixed(1) + "%";

fetch("/info").then(r => r.json()).then(m => {
  document.getElementById("info").textContent =
    `Modelo ${m.version} | ${m.classes} campeones | entrada ${m.input}`;
}).catch(() => {});

function el(tag, cls, text) {
  const e = document.createElement(tag);
  if (cls) e.className = cls;
  if (text !== undefined) e.textContent = text;
  return e;
}

async function classify(file) {
  const card = el("article", "card");
  const left = el("div");
  const img = el("img");
  img.src = URL.createObjectURL(file);
  img.alt = file.name;
  left.append(img, el("div", "fname", file.name));
  const right = el("div", "status");
  right.innerHTML = '<span class="spinner"></span>Analizando...';
  card.append(left, right);
  results.prepend(card);
  toolbar.hidden = false;

  try {
    const res = await fetch("/predict", { method: "POST", body: file });
    const data = await res.json();
    if (!res.ok) throw new Error(data.error || "Error del servidor");

    right.className = "";
    right.replaceChildren();
    const best = data.top[0];
    const name = el("div", "winner-name", pretty(best.champion));
    name.append(el("span", null, pct(best.prob)));
    const list = el("ol");
    for (const { champion, prob } of data.top) {
      const li = el("li");
      const row = el("div", "row");
      row.append(el("b", null, pretty(champion)), el("span", "pct", pct(prob)));
      const track = el("div", "track");
      const fill = el("div", "fill");
      track.append(fill);
      li.append(row, track);
      list.append(li);
      requestAnimationFrame(() => requestAnimationFrame(() => fill.style.width = (prob * 100) + "%"));
    }
    right.append(el("div", "winner", "Predicción"), name, el("div", "winner", "Top 5"), list);
    list.style.marginTop = "8px";
  } catch (err) {
    right.className = "error";
    right.textContent = err.message === "Failed to fetch"
      ? "No hay conexión con el servidor (¿sigue corriendo app.py?)"
      : err.message;
  }
}

async function handle(files) {
  for (const f of files) {
    if (f.type.startsWith("image/")) await classify(f);
  }
}

input.addEventListener("change", () => { handle([...input.files]); input.value = ""; });
["dragenter", "dragover"].forEach(ev => drop.addEventListener(ev, e => {
  e.preventDefault(); drop.classList.add("over");
}));
["dragleave", "drop"].forEach(ev => drop.addEventListener(ev, e => {
  e.preventDefault(); drop.classList.remove("over");
}));
drop.addEventListener("drop", e => handle([...e.dataTransfer.files]));
window.addEventListener("dragover", e => e.preventDefault());
window.addEventListener("drop", e => e.preventDefault());
document.addEventListener("paste", e => {
  handle([...e.clipboardData.items].filter(i => i.kind === "file").map(i => i.getAsFile()));
});
document.querySelector(".clear").addEventListener("click", () => {
  results.replaceChildren(); toolbar.hidden = true;
});
