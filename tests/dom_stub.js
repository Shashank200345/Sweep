// Minimal DOM stub to execute the replay dashboard script in node (test helper).
// Usage: node dom_stub.js <script.js> ; exits non-zero on any runtime error.
const fs = require("fs");
const src = fs.readFileSync(process.argv[2], "utf8");
const g2d = new Proxy({}, { get: (t, k) => (k in t ? t[k] : () => {}), set: (t, k, v) => { t[k] = v; return true; } });
const els = {};
function el(id) {
  if (!els[id]) {
    els[id] = { id, style: {}, textContent: "", innerHTML: "", value: "0", max: "0", clientWidth: 800,
      getAttribute: () => "200", getContext: () => g2d, classList: { toggle() {} }, set onclick(f) { this._click = f; },
      get onclick() { return this._click; } };
  }
  return els[id];
}
global.document = { getElementById: el, documentElement: {}, createElement: () => el("tmp" + Math.random()) };
global.window = { devicePixelRatio: 1 };
global.getComputedStyle = () => ({ getPropertyValue: () => "#fff" });
global.setInterval = () => 1; global.clearInterval = () => {};
new Function(src)();
const pos = el("pos");
const n = +pos.max + 1;
const show = (i) => { pos.value = String(i); pos.oninput(); };
for (const i of [0, Math.floor(n / 3), Math.floor(n / 2), n - 1]) show(i);
if (!el("pi").textContent.includes("/")) throw new Error("dashboard did not render");
console.log("ok", n, "bars;", el("hdr").textContent);
