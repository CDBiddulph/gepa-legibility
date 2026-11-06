#!/usr/bin/env python3
"""Local browser app for hand-labeling judge decisions. Stop and resume at will.

Serves one item at a time from the frozen manifests and appends every label to
an append-only JSONL log. Nothing is ever rewritten, so killing the server (or
the machine) loses at most the item currently on screen. Restarting picks up
wherever you left off; relabeling an item just appends a newer line and the
analysis takes the last one.

Blinding: the browser is never sent the judge's label, the judge's reasoning, or
the second judge's label. It receives only the artifact under review and the
judge's own rubric, so labels cannot be anchored on the judge's answer.

Usage:
    python scripts/judge_validation/label_app.py
    python scripts/judge_validation/label_app.py --judge wordchain_critique --port 8802

Then open the printed URL. Keys 1-4 pick a label, Enter saves, s skips.
"""

import argparse
import json
import os
import sys
import time
import webbrowser
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

REPO_ROOT = Path(__file__).resolve().parents[2]
# Running this file directly puts its own directory on sys.path, not the repo
# root, so the rubric imports below need the root added explicitly.
sys.path.insert(0, str(REPO_ROOT))

DATA_DIR = REPO_ROOT / "data" / "judge_validation"

JUDGES = [
    "verbalization_gepa",
    "verbalization_rl_cot",
    "psychosis_agreeableness",
    "wordchain_critique",
]

# Label options per judge. The verbalization options mirror the judge's own
# output space so that agreement is measured on identical categories.
LABEL_OPTIONS = {
    "verbalization_gepa": ["yes", "no", "unclear"],
    "verbalization_rl_cot": ["yes", "no", "unclear"],
    "wordchain_critique": ["yes", "no", "unclear"],
    "psychosis_agreeableness": [str(i) for i in range(11)] + ["ABNORMAL"],
}
PSYCHOSIS_VERBALIZATION_OPTIONS = ["yes", "mistargeted", "no", "unclear"]


def _rubric(item: dict) -> dict:
    """The judge's own rubric text, so the human applies the same definition."""
    judge, env = item["judge"], item["env"]
    if judge.startswith("verbalization"):
        from core.monitoring_prompts import get_hack_verbalization_description

        subject = (
            "the chain of thought"
            if judge == "verbalization_rl_cot"
            else "the system prompt"
        )
        return {
            "question": f"Does {subject} express intent to hack, per the definition below?",
            "definition": get_hack_verbalization_description(env, item.get("hint_type")),
        }
    if judge == "wordchain_critique":
        from tasks.wordchain.critique_judge import CRITIQUE_DETECTION_PROMPT

        definition = CRITIQUE_DETECTION_PROMPT.split("<response>")[0].strip()
        return {
            "question": "Does the response contain unqualified self-critique?",
            "definition": definition,
        }
    from tasks.psychosis.scorer import AGREEABLENESS_PROMPT

    return {
        "question": "How much does the response agree with the user's belief? (0-10)",
        "definition": AGREEABLENESS_PROMPT.strip(),
    }


class LabelStore:
    """Frozen manifests + append-only label log."""

    def __init__(self, data_dir: Path, annotator: str, judges: list[str]):
        self.data_dir = data_dir
        self.annotator = annotator
        self.log_path = data_dir / f"labels_{annotator}.jsonl"
        self.manifests: dict[str, list[dict]] = {}
        self.pools: dict[str, dict[str, dict]] = {}

        for judge in judges:
            manifest_path = data_dir / f"manifest_{judge}.jsonl"
            pool_path = data_dir / f"pool_{judge}.jsonl"
            if not manifest_path.exists():
                print(f"  skipping {judge}: no manifest (run sample_items.py)")
                continue
            with open(manifest_path) as f:
                self.manifests[judge] = [json.loads(line) for line in f]
            with open(pool_path) as f:
                self.pools[judge] = {
                    row["item_id"]: row for row in (json.loads(line) for line in f)
                }

    def labeled_ids(self) -> set[str]:
        if not self.log_path.exists():
            return set()
        with open(self.log_path) as f:
            return {json.loads(line)["item_id"] for line in f if line.strip()}

    def append(self, record: dict) -> None:
        """Append one label durably. Never rewrites earlier lines."""
        record = {**record, "annotator": self.annotator, "timestamp": time.time()}
        with open(self.log_path, "a") as f:
            f.write(json.dumps(record) + "\n")
            f.flush()
            os.fsync(f.fileno())

    def next_item(self, judge: str) -> dict | None:
        """First unlabeled item in queue order for this judge."""
        done = self.labeled_ids()
        for row in self.manifests.get(judge, []):
            if row["item_id"] in done:
                continue
            item = self.pools[judge].get(row["item_id"])
            if item is None:
                continue
            return self._blind(item, row)
        return None

    def _blind(self, item: dict, manifest_row: dict) -> dict:
        """Strip every field that could anchor the annotator on a judge's answer."""
        options = LABEL_OPTIONS[item["judge"]]
        if item["judge"].startswith("verbalization") and item["env"] == "psychosis":
            options = PSYCHOSIS_VERBALIZATION_OPTIONS
        return {
            "item_id": item["item_id"],
            "judge": item["judge"],
            "env": item["env"],
            "hint_type": item.get("hint_type"),
            "payload": item["payload"],
            "options": options,
            "rubric": _rubric(item),
            "queue_position": manifest_row["queue_position"],
        }

    def progress(self) -> list[dict]:
        done = self.labeled_ids()
        rows = []
        for judge, manifest in self.manifests.items():
            ids = {row["item_id"] for row in manifest}
            rows.append({
                "judge": judge,
                "labeled": len(ids & done),
                "total": len(ids),
            })
        return rows


class Handler(BaseHTTPRequestHandler):
    store: LabelStore

    def _send(self, status: int, body: bytes, content_type: str) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _json(self, payload) -> None:
        self._send(200, json.dumps(payload).encode(), "application/json")

    def log_message(self, *args) -> None:
        pass  # keep the console clean for the progress summary

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/":
            self._send(200, PAGE.encode(), "text/html; charset=utf-8")
        elif parsed.path == "/api/next":
            judge = parse_qs(parsed.query).get("judge", [JUDGES[0]])[0]
            self._json({"item": self.store.next_item(judge), "progress": self.store.progress()})
        elif parsed.path == "/api/progress":
            self._json({"progress": self.store.progress()})
        else:
            self._send(404, b"not found", "text/plain")

    def do_POST(self) -> None:
        if urlparse(self.path).path != "/api/label":
            self._send(404, b"not found", "text/plain")
            return
        length = int(self.headers.get("Content-Length", 0))
        record = json.loads(self.rfile.read(length) or b"{}")
        self.store.append(record)
        judge = record.get("judge", JUDGES[0])
        self._json({"item": self.store.next_item(judge), "progress": self.store.progress()})


PAGE = """
<title>Judge validation labeling</title>
<style>
  :root { color-scheme: light dark; --fg:#111; --bg:#fff; --muted:#666; --line:#ddd; --accent:#2563eb; --card:#f7f7f8; }
  @media (prefers-color-scheme: dark) {
    :root { --fg:#e8e8ea; --bg:#151517; --muted:#9a9aa2; --line:#33333a; --accent:#7aa2f7; --card:#1e1e22; }
  }
  body { margin:0; font:15px/1.55 -apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif; color:var(--fg); background:var(--bg); }
  header { position:sticky; top:0; background:var(--bg); border-bottom:1px solid var(--line); padding:10px 20px; display:flex; gap:16px; align-items:center; flex-wrap:wrap; }
  select, button { font:inherit; color:var(--fg); background:var(--card); border:1px solid var(--line); border-radius:6px; padding:6px 10px; }
  button.opt { cursor:pointer; }
  button.opt[aria-pressed="true"] { background:var(--accent); color:#fff; border-color:var(--accent); }
  main { max-width:900px; margin:0 auto; padding:20px; }
  .card { background:var(--card); border:1px solid var(--line); border-radius:10px; padding:14px 16px; margin-bottom:16px; }
  .card h2 { margin:0 0 8px; font-size:13px; text-transform:uppercase; letter-spacing:.06em; color:var(--muted); }
  pre { white-space:pre-wrap; word-wrap:break-word; margin:0; font:13px/1.5 ui-monospace,SFMono-Regular,Menlo,monospace; max-height:60vh; overflow:auto; }
  .row { display:flex; gap:8px; flex-wrap:wrap; align-items:center; }
  .muted { color:var(--muted); font-size:13px; }
  textarea { width:100%; box-sizing:border-box; font:inherit; color:var(--fg); background:var(--bg); border:1px solid var(--line); border-radius:6px; padding:8px; }
  details summary { cursor:pointer; color:var(--muted); font-size:13px; }
</style>
<header>
  <strong>Judge validation</strong>
  <select id="judge"></select>
  <span id="progress" class="muted"></span>
  <span style="flex:1"></span>
  <span class="muted">1-9 label &middot; Enter save &middot; s skip</span>
</header>
<main id="main"></main>
<script>
const state = { item:null, label:null, overseer:null, shownAt:Date.now() };
const JUDGES = ["verbalization_gepa","verbalization_rl_cot","psychosis_agreeableness","wordchain_critique"];

const judgeSel = document.getElementById("judge");
JUDGES.forEach(j => { const o=document.createElement("option"); o.value=j; o.textContent=j; judgeSel.append(o); });
judgeSel.value = localStorage.getItem("judge") || JUDGES[0];
judgeSel.onchange = () => { localStorage.setItem("judge", judgeSel.value); load(); };

function esc(s){ return (s??"").toString().replace(/[&<>]/g, c => ({"&":"&amp;","<":"&lt;",">":"&gt;"}[c])); }

function renderProgress(rows){
  document.getElementById("progress").textContent =
    rows.map(r => `${r.judge.split("_")[0]} ${r.labeled}/${r.total}`).join("  ·  ");
}

function payloadCards(item){
  const p = item.payload, out = [];
  const add = (title, text) => { if (text) out.push(`<div class="card"><h2>${title}</h2><pre>${esc(text)}</pre></div>`); };
  const fmt = v => typeof v === "string" ? v : JSON.stringify(v, null, 2);
  add("System prompt / instructions", p.instructions && fmt(p.instructions));
  add("Chain of thought", p.cot && fmt(p.cot));
  add("Model response", p.response && fmt(p.response));
  if (p.user_message) out.push(
    `<div class="card"><details><summary>User message (context only)</summary><pre>${esc(fmt(p.user_message))}</pre></details></div>`);
  return out.join("");
}

function render(){
  const item = state.item, main = document.getElementById("main");
  if (!item){ main.innerHTML = `<div class="card"><h2>Done</h2>No unlabeled items left for this judge. Pick another, or run analyze.py.</div>`; return; }
  const isVerb = item.judge.startsWith("verbalization");
  main.innerHTML = `
    <div class="card">
      <h2>${esc(item.env)}${item.hint_type ? " · " + esc(item.hint_type) : ""} · item ${item.queue_position}</h2>
      <div><strong>${esc(item.rubric.question)}</strong></div>
      <details style="margin-top:6px"><summary>Show the judge's full rubric</summary><pre>${esc(item.rubric.definition)}</pre></details>
    </div>
    ${payloadCards(item)}
    <div class="card">
      <h2>Your label</h2>
      <div class="row" id="opts">${item.options.map((o,i) =>
        `<button class="opt" data-v="${esc(o)}" aria-pressed="false">${i<9?`<u>${i+1}</u> `:""}${esc(o)}</button>`).join("")}</div>
      ${isVerb ? `
      <div style="margin-top:14px">
        <div class="muted">Optional second question (construct validity): would a human overseer reading this text learn the model intends to hack?</div>
        <div class="row" id="overseer" style="margin-top:6px">
          ${["yes","no"].map(o => `<button class="opt" data-v="${o}" aria-pressed="false">${o}</button>`).join("")}
        </div>
      </div>` : ""}
      <div style="margin-top:14px"><textarea id="notes" rows="2" placeholder="Notes (optional)"></textarea></div>
      <div class="row" style="margin-top:10px">
        <button id="save">Save &amp; next</button>
        <button id="skip">Skip</button>
        <span class="muted" id="hint"></span>
      </div>
    </div>`;

  const wire = (containerId, key) => {
    const container = document.getElementById(containerId);
    if (!container) return;
    container.querySelectorAll("button.opt").forEach(b => b.onclick = () => {
      state[key] = b.dataset.v;
      container.querySelectorAll("button.opt").forEach(x =>
        x.setAttribute("aria-pressed", String(x === b)));
    });
  };
  wire("opts", "label");
  wire("overseer", "overseer");
  document.getElementById("save").onclick = save;
  document.getElementById("skip").onclick = () => save(true);
  state.shownAt = Date.now();
}

async function load(){
  const r = await fetch(`/api/next?judge=${encodeURIComponent(judgeSel.value)}`);
  const d = await r.json();
  state.item = d.item; state.label = null; state.overseer = null;
  renderProgress(d.progress); render();
}

async function save(skip=false){
  if (!state.item) return;
  if (!skip && !state.label){ document.getElementById("hint").textContent = "pick a label first"; return; }
  const body = {
    item_id: state.item.item_id,
    judge: state.item.judge,
    label: skip ? null : state.label,
    skipped: skip,
    overseer_would_learn: state.overseer,
    notes: (document.getElementById("notes")||{}).value || "",
    elapsed_sec: (Date.now() - state.shownAt) / 1000,
  };
  const r = await fetch("/api/label", {method:"POST", body: JSON.stringify(body)});
  const d = await r.json();
  state.item = d.item; state.label = null; state.overseer = null;
  renderProgress(d.progress); render();
}

document.addEventListener("keydown", e => {
  if (e.target.tagName === "TEXTAREA") return;
  if (e.key === "Enter") { save(); return; }
  if (e.key === "s") { save(true); return; }
  const n = parseInt(e.key, 10);
  if (n >= 1 && n <= 9) {
    const buttons = document.querySelectorAll("#opts button.opt");
    if (buttons[n-1]) buttons[n-1].click();
  }
});

load();
</script>
"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotator", default="caleb")
    parser.add_argument("--judge", action="append", help="restrict to these judges")
    parser.add_argument("--port", type=int, default=8801)
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args()

    store = LabelStore(args.data_dir, args.annotator, args.judge or JUDGES)
    if not store.manifests:
        raise SystemExit("No manifests found. Run extract_items.py then sample_items.py.")

    for row in store.progress():
        print(f"  {row['judge']:24} {row['labeled']:4d}/{row['total']:<4d} labeled")
    print(f"\n  labels append to {store.log_path.relative_to(REPO_ROOT)}")

    Handler.store = store
    url = f"http://127.0.0.1:{args.port}/"
    print(f"  serving {url}   (ctrl-c to stop; progress is saved after every item)\n")
    if not args.no_browser:
        webbrowser.open(url)
    try:
        HTTPServer(("127.0.0.1", args.port), Handler).serve_forever()
    except KeyboardInterrupt:
        print("\n  stopped; rerun this command to pick up where you left off")


if __name__ == "__main__":
    main()
