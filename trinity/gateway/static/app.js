"use strict";
// Trinity-C2C workflow debugger — fetch-stream SSE client + DOM rendering.
// SECURITY: every piece of model/LLM-derived text is rendered with textContent
// (never innerHTML) so a crafted role output cannot inject markup.

const $ = (id) => document.getElementById(id);
const timelineEl = $("timeline");
const finalEl = $("final");
const finalBadge = $("final-badge");
const errorBanner = $("error-banner");
const rawLog = $("raw-log");
const rawCount = $("raw-count");
const runMeta = $("run-meta");
const healthDot = $("health-dot");
const healthText = $("health-text");

const ROLE_NAME = { thinker: "Thinker", worker: "Worker", verifier: "Verifier" };
const roleName = (r) => ROLE_NAME[r] || r || "?";

let steps = {};      // step number -> refs
let rawN = 0;
let controller = null;

// ---------------------------------------------------------------- controls
const queryEl = $("query"), mockEl = $("mock"), promptsEl = $("prompts");
const maxTurnsEl = $("max-turns"), mockDelayEl = $("mock-delay");
const runBtn = $("run"), stopBtn = $("stop");

runBtn.addEventListener("click", run);
stopBtn.addEventListener("click", stop);
window.addEventListener("beforeunload", () => controller && controller.abort());

function setRunning(on) {
  runBtn.disabled = on;
  stopBtn.disabled = !on;
}

function resetUI() {
  steps = {};
  rawN = 0;
  timelineEl.replaceChildren();
  rawLog.textContent = "";
  rawCount.textContent = "0";
  finalEl.textContent = "—";
  finalBadge.textContent = "";
  finalBadge.className = "badge";
  errorBanner.classList.add("hidden");
  errorBanner.textContent = "";
}

// ---------------------------------------------------------------- run / stream
async function run() {
  const payload = {
    query: queryEl.value || "",
    mock: mockEl.checked,
    include_prompts: promptsEl.checked,
    max_turns: parseInt(maxTurnsEl.value, 10) || 5,
    mock_delay: parseFloat(mockDelayEl.value) || 0,
  };
  resetUI();
  setRunning(true);
  runMeta.textContent = "running…";
  controller = new AbortController();
  try {
    const resp = await fetch("/debug/runs/stream", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
      signal: controller.signal,
    });
    if (!resp.ok || !resp.body) throw new Error("HTTP " + resp.status);
    const reader = resp.body.getReader();
    const dec = new TextDecoder();
    let buf = "";
    for (;;) {
      const { done, value } = await reader.read();
      if (done) break;
      buf += dec.decode(value, { stream: true });
      let i;
      while ((i = buf.indexOf("\n\n")) >= 0) {
        const frame = buf.slice(0, i);
        buf = buf.slice(i + 2);
        const ev = parseFrame(frame);
        if (ev) dispatch(ev);
      }
    }
  } catch (e) {
    if (e.name !== "AbortError") showError("Stream error: " + e.message);
    else runMeta.textContent = "stopped";
  } finally {
    setRunning(false);
    controller = null;
  }
}

function stop() {
  if (controller) controller.abort();
}

function parseFrame(frame) {
  let data = "";
  for (const ln of frame.split("\n")) {
    if (ln.startsWith("data:")) data += ln.slice(5).trim();
  }
  if (!data) return null;
  try { return JSON.parse(data); } catch { return null; }
}

// ---------------------------------------------------------------- dispatch
function dispatch(ev) {
  logRaw(ev);
  switch (ev.type) {
    case "run_start": onRunStart(ev); break;
    case "decision": onDecision(ev); break;
    case "turn_start": onTurnStart(ev); break;
    case "turn_end": onTurnEnd(ev); break;
    case "verdict": onVerdict(ev); break;
    case "error": onError(ev); break;
    case "final": onFinal(ev); break;
  }
}

function onRunStart(ev) {
  runMeta.textContent = `running · max_turns=${ev.max_turns}`;
}

function onDecision(ev) {
  if (!ev.role) { // coordinator chose to stop
    const note = document.createElement("div");
    note.className = "empty";
    note.textContent = `Coordinator stopped (${ev.reason || "no action"}).`;
    timelineEl.append(note);
    return;
  }
  createStepCard(ev);
}

function createStepCard(ev) {
  const card = document.createElement("div");
  card.className = `step role-${ev.role}`;
  card.dataset.step = ev.step;

  const head = document.createElement("div");
  head.className = "step-head";

  const no = document.createElement("span");
  no.className = "step-no";
  no.textContent = `Step ${ev.step}`;

  const flow = document.createElement("span");
  flow.className = "flow";
  const c = document.createElement("span"); c.textContent = "Coordinator";
  const arrow = document.createElement("span"); arrow.className = "arrow"; arrow.textContent = "→";
  const r = document.createElement("span"); r.className = "role"; r.textContent = roleName(ev.role);
  const model = document.createElement("span"); model.className = "model";
  flow.append(c, arrow, r, document.createTextNode(" "), model);

  const spacer = document.createElement("span"); spacer.className = "spacer";
  const badge = document.createElement("span"); badge.className = "badge running"; badge.textContent = "running";
  const dur = document.createElement("span"); dur.className = "dur";

  head.append(no, flow, spacer, badge, dur);
  card.append(head);

  // learned-coordinator meta (role logits etc.), if any
  if (ev.meta && Object.keys(ev.meta).length) {
    const meta = document.createElement("div");
    meta.className = "meta-line";
    meta.textContent = "coordinator meta: " + JSON.stringify(ev.meta);
    card.append(meta);
  }

  const outWrap = document.createElement("div"); outWrap.className = "output hidden";
  const outPre = document.createElement("pre");
  outWrap.append(outPre);
  card.append(outWrap);

  timelineEl.append(card);
  card.scrollIntoView({ behavior: "smooth", block: "nearest" });
  steps[ev.step] = { card, model, badge, dur, outWrap, outPre };
}

function onTurnStart(ev) {
  const s = steps[ev.step];
  if (!s) return;
  if (ev.model_name) s.model.textContent = `(${ev.model_name})`;
  if (promptsEl.checked && (ev.system || ev.user)) addPrompt(s, ev);
}

function addPrompt(s, ev) {
  const d = document.createElement("details"); d.className = "prompt";
  const sum = document.createElement("summary"); sum.textContent = "Prompt sent to this role";
  d.append(sum);
  for (const [label, val] of [["system", ev.system], ["user", ev.user]]) {
    if (!val) continue;
    const blk = document.createElement("div"); blk.className = "p-block";
    const lab = document.createElement("div"); lab.className = "p-label"; lab.textContent = label;
    const pre = document.createElement("pre"); pre.textContent = val;
    blk.append(lab, pre);
    d.append(blk);
  }
  s.card.insertBefore(d, s.outWrap);
}

function onTurnEnd(ev) {
  const s = steps[ev.step];
  if (!s) return;
  s.badge.className = "badge done";
  s.badge.textContent = "done";
  s.dur.textContent = (ev.duration_ms != null ? ev.duration_ms + " ms" : "") +
    (ev.output_chars != null ? ` · ${ev.output_chars} chars` : "");
  s.outPre.textContent = ev.output || "";
  s.outWrap.classList.remove("hidden");
}

function onVerdict(ev) {
  const s = steps[ev.step];
  if (!s) return;
  const row = document.createElement("div"); row.className = "verdict-row";
  const label = document.createElement("span"); label.className = "label"; label.textContent = "Verifier verdict:";
  const badge = document.createElement("span");
  const accept = ev.verdict === "ACCEPT";
  badge.className = "badge " + (accept ? "accept" : "revise");
  badge.textContent = ev.verdict + (accept && ev.accepted ? " · accepted" : (accept ? " (no artifact)" : " · loop back to Worker"));
  row.append(label, badge);
  s.card.insertBefore(row, s.outWrap);
}

function onError(ev) {
  if (ev.step && steps[ev.step]) {
    const s = steps[ev.step];
    s.badge.className = "badge error";
    s.badge.textContent = "error";
    s.outPre.textContent = ev.message || "error";
    s.outWrap.classList.remove("hidden");
  }
  showError(ev.message || "error");
}

function onFinal(ev) {
  finalEl.textContent = ev.final || "(no artifact)";
  const accepted = !!ev.accepted;
  finalBadge.className = "badge " + (accepted ? "accept" : "revise");
  finalBadge.textContent = accepted ? "ACCEPTED" : "not accepted";
  runMeta.textContent = `${ev.turns} turn(s) · ${accepted ? "accepted" : "stopped"}`;
  if (ev.error) showError(ev.error);
}

function showError(msg) {
  errorBanner.textContent = msg;
  errorBanner.classList.remove("hidden");
}

function logRaw(ev) {
  rawN += 1;
  rawCount.textContent = String(rawN);
  rawLog.textContent += JSON.stringify(ev) + "\n";
}

// ---------------------------------------------------------------- health
async function pollHealth() {
  try {
    const r = await fetch("/healthz", { cache: "no-store" });
    const j = await r.json();
    healthDot.className = "dot ok";
    healthText.textContent = `online · ${j.model || ""}${j.mock_default ? " · mock-default" : ""}`;
  } catch {
    healthDot.className = "dot down";
    healthText.textContent = "offline";
  }
}
pollHealth();
setInterval(pollHealth, 5000);
