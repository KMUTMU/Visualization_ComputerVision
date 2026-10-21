/*
 * AI 이미지 판단 라이브 대시보드
 * precompute.py 가 만든 data/results.json 을 읽어
 * 이미지를 한 장씩 넘기며 공격 전/후 판단을 보여준다.
 */

const els = {
  meta: document.getElementById("meta"),
  playBtn: document.getElementById("playBtn"),
  prevBtn: document.getElementById("prevBtn"),
  nextBtn: document.getElementById("nextBtn"),
  speed: document.getElementById("speed"),
  speedVal: document.getElementById("speedVal"),
  attackToggle: document.getElementById("attackToggle"),
  counter: document.getElementById("counter"),
  progress: document.getElementById("progress"),
  origImg: document.getElementById("origImg"),
  advImg: document.getElementById("advImg"),
  noiseImg: document.getElementById("noiseImg"),
  origVerdict: document.getElementById("origVerdict"),
  advVerdict: document.getElementById("advVerdict"),
  cleanBars: document.getElementById("cleanBars"),
  advBars: document.getElementById("advBars"),
  advPanel: document.getElementById("advPanel"),
  arrow: document.getElementById("arrow"),
  summary: document.getElementById("summary"),
  error: document.getElementById("error"),
};

let state = {
  data: null,
  idx: 0,
  playing: false,
  timer: null,
  attackOn: false,
};

const DATA_DIR = "data/";

init();

async function init() {
  try {
    const res = await fetch(DATA_DIR + "results.json", { cache: "no-store" });
    if (!res.ok) throw new Error("results.json 을 불러올 수 없습니다 (status " + res.status + ")");
    state.data = await res.json();
  } catch (e) {
    showError(e);
    return;
  }

  if (!state.data.records || state.data.records.length === 0) {
    showError(new Error("results.json 에 레코드가 없습니다."));
    return;
  }

  renderMeta();
  renderSummary();
  bindEvents();
  render();
}

function renderMeta() {
  const d = state.data;
  const p = d.attack_params || {};
  els.meta.innerHTML = `
    <div>모델: <b>${esc(d.model)}</b></div>
    <div>공격: <b>${esc((d.attack || "").toUpperCase())}</b>
      (ε=${p.epsilon_255 ?? "?"}/255, T=${p.steps ?? "?"})</div>
    <div>장치: <b>${esc(d.device || "-")}</b> · 이미지 <b>${d.count}</b>장</div>
  `;
}

function bindEvents() {
  els.playBtn.addEventListener("click", togglePlay);
  els.prevBtn.addEventListener("click", () => { pause(); step(-1); });
  els.nextBtn.addEventListener("click", () => { pause(); step(1); });
  els.speed.addEventListener("input", () => {
    els.speedVal.textContent = (els.speed.value / 1000).toFixed(1) + "s";
    if (state.playing) { pause(); play(); }
  });
  els.attackToggle.addEventListener("change", () => {
    state.attackOn = els.attackToggle.checked;
    render();
  });
  document.addEventListener("keydown", (e) => {
    if (e.key === "ArrowRight") { pause(); step(1); }
    else if (e.key === "ArrowLeft") { pause(); step(-1); }
    else if (e.key === " ") { e.preventDefault(); togglePlay(); }
    else if (e.key.toLowerCase() === "a") {
      els.attackToggle.checked = !els.attackToggle.checked;
      state.attackOn = els.attackToggle.checked; render();
    }
  });
}

function togglePlay() { state.playing ? pause() : play(); }

function play() {
  state.playing = true;
  els.playBtn.textContent = "⏸ 일시정지";
  els.playBtn.classList.remove("primary");
  const interval = parseInt(els.speed.value, 10);
  state.timer = setInterval(() => {
    if (state.idx >= state.data.records.length - 1) { pause(); return; }
    step(1);
  }, interval);
}

function pause() {
  state.playing = false;
  els.playBtn.textContent = "▶ 재생";
  els.playBtn.classList.add("primary");
  if (state.timer) { clearInterval(state.timer); state.timer = null; }
}

function step(delta) {
  const n = state.data.records.length;
  state.idx = Math.max(0, Math.min(n - 1, state.idx + delta));
  render();
}

function render() {
  const d = state.data;
  const rec = d.records[state.idx];
  const n = d.records.length;

  els.counter.textContent = `${state.idx + 1} / ${n}`;
  els.progress.style.width = ((state.idx + 1) / n * 100) + "%";

  els.origImg.src = DATA_DIR + rec.orig_img;
  els.advImg.src = DATA_DIR + rec.adv_img;
  els.noiseImg.src = DATA_DIR + rec.noise_img;

  // 원본 판정
  els.origVerdict.className = "verdict " + (rec.clean.correct ? "correct" : "wrong");
  els.origVerdict.innerHTML = `
    <div class="pred">${esc(rec.clean.pred_name)}</div>
    <div class="conf">확신도 ${(rec.clean.topk[0].prob * 100).toFixed(1)}%
      · 정답: ${esc(rec.true_name)}
      <span class="badge ${rec.clean.correct ? "ok" : "fail"}">
        ${rec.clean.correct ? "정답 ✓" : "오답 ✗"}</span>
    </div>`;

  renderBars(els.cleanBars, rec.clean.topk, "clean", rec.true_label);

  // 공격 패널 (토글 꺼지면 흐리게)
  if (state.attackOn) {
    els.advPanel.classList.remove("hidden-panel");
    els.arrow.classList.remove("hidden-arrow");
  } else {
    els.advPanel.classList.add("hidden-panel");
    els.arrow.classList.add("hidden-arrow");
  }

  // 공격 후 판정
  const a = rec.adv;
  els.advVerdict.className = "verdict " + (a.correct ? "correct" : "wrong");
  let badge;
  if (a.fooled) badge = `<span class="badge fail">속았음 ✗</span>`;
  else if (a.correct) badge = `<span class="badge ok">버팀 ✓</span>`;
  else badge = `<span class="badge fail">오답 ✗</span>`;
  els.advVerdict.innerHTML = `
    <div class="pred">${esc(a.pred_name)}</div>
    <div class="conf">확신도 ${(a.topk[0].prob * 100).toFixed(1)}%
      · 정답: ${esc(rec.true_name)} ${badge}</div>`;

  renderBars(els.advBars, a.topk, "adv", rec.true_label);
}

function renderBars(container, topk, kind, trueLabel) {
  container.innerHTML = topk.map((t) => {
    const pct = (t.prob * 100).toFixed(1);
    const star = t.index === trueLabel ? " ★" : "";
    return `
      <div class="bar-row">
        <div class="bar-label" title="${esc(t.label)}">${esc(t.label)}${star}</div>
        <div class="bar-track">
          <div class="bar-fill ${kind}" style="width:${pct}%"></div>
        </div>
        <div class="bar-val">${pct}%</div>
      </div>`;
  }).join("");
}

function renderSummary() {
  const s = state.data.summary;
  els.summary.innerHTML = `
    <div class="stat">
      <span class="lbl">공격 전 정확도</span>
      <b class="c-clean">${s.clean_accuracy}%</b>
    </div>
    <div class="stat">
      <span class="lbl">공격 후 정확도</span>
      <b class="c-adv">${s.adv_accuracy}%</b>
    </div>
    <div class="stat">
      <span class="lbl">속은 이미지 수</span>
      <b class="c-warn">${s.fooled}장</b>
    </div>
    <div class="stat">
      <span class="lbl">원래 맞힌 것 중 속은 비율</span>
      <b class="c-warn">${s.fool_rate_of_correct}%</b>
    </div>
  `;
}

function showError(e) {
  els.error.classList.remove("hidden");
  els.error.innerHTML = `
    <b>데이터를 불러오지 못했습니다.</b><br>${esc(e.message)}<br><br>
    먼저 로컬에서 데이터를 생성하고, <code>web/</code> 폴더를 로컬 서버로 여세요:<br>
    <code>python precompute.py --cv-repo C:\\GITHUB\\ComputerVision --attack ifgsm --num 30</code><br>
    <code>cd web &amp;&amp; python -m http.server 8000</code> → 브라우저에서 <code>http://localhost:8000</code><br>
    (file:// 로 직접 열면 브라우저 보안정책 때문에 JSON 로드가 막힐 수 있습니다.)
  `;
}

function esc(s) {
  return String(s).replace(/[&<>"']/g, (c) => ({
    "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;",
  }[c]));
}
