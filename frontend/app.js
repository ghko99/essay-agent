// 글결 · AI 에세이 채점 — 학습자용 프런트엔드
const $ = (id) => document.getElementById(id);

// ── State ──────────────────────────────────────────────────────────
let rubric = [];
let radarChart = null;
let currentTopic = null;
let currentKeywords = "";
let currentCriteria = null;   // criteria from the last scoring (fed to the audit)
let currentEssay = "";
let currentReport = null;      // last verify report (agent scores + evidence)
let linguisticLoaded = false;  // lazy-load guard for the 언어 분석 section

// topic picker state
let topicsData = null;          // { groups: [...], subjects, levels }
let activeTypeIdx = 0;
let selectedSubjects = new Set();
let selectedLevels = new Set();
let topicQuery = "";

// ── Utils ──────────────────────────────────────────────────────────
function toast(msg) {
  const t = $("toast");
  t.textContent = msg;
  t.hidden = false;
  clearTimeout(toast._h);
  toast._h = setTimeout(() => { t.hidden = true; }, 3500);
}

let wizardStep = 1;   // current wizard step, survives route changes (guide ↔ home)

function setStep(n) {
  wizardStep = n;
  $("view-guide").classList.remove("active");
  ["view-topic", "view-write", "view-result"].forEach((id, i) => {
    $(id).classList.toggle("active", i + 1 === n);
  });
  document.querySelectorAll(".step-pill").forEach((el) => {
    const s = parseInt(el.dataset.step, 10);
    el.classList.toggle("active", s === n);
    el.classList.toggle("done", s < n);
  });
  window.scrollTo({ top: 0, behavior: "smooth" });
}

// ── Router (#/ = 채점 위저드, #/guide = 채점 기준) ─────────────────
function currentRoute() {
  return location.hash.startsWith("#/guide") ? "guide" : "home";
}

function renderRoute() {
  const route = currentRoute();
  document.querySelectorAll(".nav-link").forEach((a) => {
    a.classList.toggle("active", a.dataset.route === route);
  });
  document.querySelector(".progress-bar").style.display = route === "guide" ? "none" : "";
  if (route === "guide") {
    ["view-topic", "view-write", "view-result"].forEach((id) => $(id).classList.remove("active"));
    $("view-guide").classList.add("active");
    window.scrollTo({ top: 0 });
    loadGuide();
  } else {
    setStep(wizardStep);
  }
}

function countEssay() {
  const v = $("essay").value;
  const chars = v.length;
  const words = v.trim().split(/\s+/).filter(Boolean).length;
  $("essay-counter").textContent = `${chars.toLocaleString()}자 · ${words.toLocaleString()}단어`;
}

// ── Step 1: Topic selection ────────────────────────────────────────
async function loadTopics() {
  const r = await fetch("/api/topics");
  topicsData = await r.json();
  renderTypeTabs();
  renderFilters();
  renderTopicGrid();
}

function renderTypeTabs() {
  const el = $("type-tabs");
  el.innerHTML = topicsData.groups.map((g, i) => `
    <button class="type-tab ${i === activeTypeIdx ? "active" : ""}" type="button" data-idx="${i}" role="tab">
      <span>${g.short}</span>
      <span class="type-tab-count">${g.count}</span>
    </button>
  `).join("");
  el.querySelectorAll(".type-tab").forEach((btn) => {
    btn.addEventListener("click", () => {
      activeTypeIdx = parseInt(btn.dataset.idx, 10);
      selectedSubjects.clear();
      selectedLevels.clear();
      renderTypeTabs();
      renderFilters();
      renderTopicGrid();
    });
  });
  $("type-desc").textContent = topicsData.groups[activeTypeIdx].description;
}

function renderFilters() {
  const group = topicsData.groups[activeTypeIdx];
  const subjectSet = [...new Set(group.topics.map((t) => t.subject).filter(Boolean))].sort();
  const levelSet = ["상", "중", "하"].filter((l) => group.topics.some((t) => t.level === l));

  const sEl = $("filter-subjects");
  sEl.innerHTML = subjectSet.map((s) =>
    `<button class="chip-filter ${selectedSubjects.has(s) ? "active" : ""}" data-v="${s}" type="button">${s}</button>`
  ).join("");
  sEl.querySelectorAll(".chip-filter").forEach((btn) => {
    btn.addEventListener("click", () => {
      const v = btn.dataset.v;
      if (selectedSubjects.has(v)) selectedSubjects.delete(v);
      else selectedSubjects.add(v);
      renderFilters();
      renderTopicGrid();
    });
  });

  const lEl = $("filter-levels");
  lEl.innerHTML = levelSet.map((l) =>
    `<button class="chip-filter ${selectedLevels.has(l) ? "active" : ""}" data-v="${l}" type="button">난이도 ${l}</button>`
  ).join("");
  lEl.querySelectorAll(".chip-filter").forEach((btn) => {
    btn.addEventListener("click", () => {
      const v = btn.dataset.v;
      if (selectedLevels.has(v)) selectedLevels.delete(v);
      else selectedLevels.add(v);
      renderFilters();
      renderTopicGrid();
    });
  });
}

function renderTopicGrid() {
  const group = topicsData.groups[activeTypeIdx];
  const q = topicQuery.trim().toLowerCase();
  const filtered = group.topics.filter((t) => {
    if (selectedSubjects.size && !selectedSubjects.has(t.subject)) return false;
    if (selectedLevels.size && !selectedLevels.has(t.level)) return false;
    if (q) {
      const hay = `${t.prompt} ${t.topic || ""} ${t.subject || ""} ${t.keyword || ""}`.toLowerCase();
      if (!hay.includes(q)) return false;
    }
    return true;
  });

  $("topic-count").textContent = `${filtered.length} / ${group.count}개`;
  $("topic-empty").hidden = filtered.length > 0;

  $("topic-grid").innerHTML = filtered.map((t, i) => {
    const tags = [];
    if (t.subject) tags.push(`<span class="tag tag-subject">${t.subject}</span>`);
    if (t.level)   tags.push(`<span class="tag tag-level" data-level="${t.level}">난이도 ${t.level}</span>`);
    if (t.grade)   tags.push(`<span class="tag tag-grade">${t.grade}</span>`);
    const kw = (t.keyword || "").split(",").map(s => s.trim()).filter(Boolean);
    return `
      <button class="topic-card" data-idx="${i}" type="button">
        <div class="topic-card-meta">${tags.join("")}</div>
        <div class="topic-card-title">${escapeHtml(t.prompt)}</div>
        ${t.topic ? `<div class="topic-card-topic">${escapeHtml(t.topic)}</div>` : ""}
        ${kw.length ? `<div class="topic-card-kw">${kw.map(k => `<span>#${escapeHtml(k)}</span>`).join("")}</div>` : ""}
      </button>
    `;
  }).join("");

  $("topic-grid").querySelectorAll(".topic-card").forEach((el) => {
    el.addEventListener("click", () => {
      const row = filtered[Number(el.dataset.idx)];
      if (row) pickTopic(row.prompt, row.keyword || "");
    });
  });
}

function escapeHtml(s) {
  return String(s ?? "").replace(/[&<>"']/g, (c) => ({
    "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;",
  })[c]);
}

function pickTopic(topic, keywords) {
  currentTopic = topic;
  currentKeywords = keywords || "";
  $("selected-topic").textContent = topic;
  const kwEl = $("selected-keywords");
  if (currentKeywords.trim()) {
    kwEl.innerHTML = currentKeywords
      .split(",")
      .map((k) => `<span>#${k.trim()}</span>`)
      .join("");
    kwEl.style.display = "";
  } else {
    kwEl.innerHTML = "";
    kwEl.style.display = "none";
  }
  $("write-meta").textContent = "AI가 8가지 기준으로 채점해 줄 거예요.";
  setStep(2);
  $("essay").focus();
}

// ── Step 3: Result rendering ──────────────────────────────────────
async function loadRubric() {
  const r = await fetch("/api/rubric");
  const d = await r.json();
  rubric = d.rubric;
}

// ── 채점 기준 페이지 (#/guide) ─────────────────────────────────────
let guideData = null;      // /api/rubric/guide response (lazy)
let guideTypeIdx = 0;

async function loadGuide() {
  if (guideData) return;
  try {
    const r = await fetch("/api/rubric/guide");
    if (!r.ok) throw new Error(`요청 실패 (${r.status})`);
    guideData = await r.json();
    $("guide-loading").hidden = true;
    renderGuideLevels();
    renderGuideTabs();
    renderGuideGrid();
  } catch (e) {
    $("guide-loading").hidden = true;
    toast(`채점 기준을 불러오지 못했어요: ${e.message || e}`);
  }
}

// score_to_level ({"1":"evaluation_1",…}) → {evaluation_1:"1~2점",…}
function guideLevelRanges() {
  const by = {};
  Object.entries(guideData.score_to_level || {}).forEach(([score, lv]) => {
    (by[lv] = by[lv] || []).push(Number(score));
  });
  const out = {};
  Object.entries(by).forEach(([lv, arr]) => {
    arr.sort((a, b) => a - b);
    out[lv] = arr.length > 1 ? `${arr[0]}~${arr[arr.length - 1]}점` : `${arr[0]}점`;
  });
  return out;
}

const GUIDE_LEVELS = ["evaluation_1", "evaluation_2", "evaluation_3", "evaluation_4", "evaluation_5"];

function renderGuideLevels() {
  const ranges = guideLevelRanges();
  $("guide-levels").innerHTML = `
    <div class="guide-levels-title">점수 ↔ 평가 단계</div>
    <div class="guide-levels-row">
      ${GUIDE_LEVELS.map((lv, i) => `
        <div class="guide-level-cell lv${i + 1}">
          <div class="guide-level-num">${i + 1}단계</div>
          <div class="guide-level-range">${escapeHtml(ranges[lv] || "")}</div>
        </div>`).join("")}
    </div>
    <p class="guide-levels-note">1단계가 가장 낮고 5단계가 가장 높아요. 아래에서 글 유형을 고르면 8개 기준의 단계별 설명을 볼 수 있어요.</p>`;
}

function renderGuideTabs() {
  const el = $("guide-type-tabs");
  el.innerHTML = (guideData.types || []).map((t, i) => `
    <button class="type-tab ${i === guideTypeIdx ? "active" : ""}" type="button" data-idx="${i}" role="tab">
      <span>${escapeHtml(t.label || t.key)}</span>
      <span class="type-tab-count">${escapeHtml(t.purpose || "")}</span>
    </button>
  `).join("");
  el.querySelectorAll(".type-tab").forEach((btn) => {
    btn.addEventListener("click", () => {
      guideTypeIdx = parseInt(btn.dataset.idx, 10);
      renderGuideTabs();
      renderGuideGrid();
    });
  });
  const t = (guideData.types || [])[guideTypeIdx];
  $("guide-type-desc").textContent = t ? (t.description || "") : "";
}

function renderGuideGrid() {
  const t = (guideData.types || [])[guideTypeIdx];
  if (!t) { $("guide-grid").innerHTML = ""; return; }
  const ranges = guideLevelRanges();
  $("guide-grid").innerHTML = (t.slots || []).map((s) => `
    <div class="guide-card" data-cat="${escapeAttr(s.category || "")}">
      <div class="guide-card-head">
        <span class="guide-card-cat">${escapeHtml(s.category || "")}</span>
        <span class="guide-card-name">${escapeHtml(s.name || s.slot)}</span>
      </div>
      <div class="guide-ladder">
        ${GUIDE_LEVELS.map((lv, i) => `
          <div class="guide-ladder-row">
            <span class="guide-ladder-lv lv${i + 1}">${i + 1}단계 <em>${escapeHtml(ranges[lv] || "")}</em></span>
            <span class="guide-ladder-txt">${escapeHtml((s.criteria || {})[lv] || "—")}</span>
          </div>`).join("")}
      </div>
    </div>
  `).join("");
}

// percentile (top_pct = % scoring at-or-above) → friendly 상위/하위 label
function pctLabel(p) {
  if (p == null) return "";
  return p <= 50 ? `상위 ${p}%` : `하위 ${Math.max(1, 100 - p)}%`;
}

function gradeBand(total) {
  if (total == null) return { label: "—", cls: "none", hint: "" };
  if (total >= 56) return { label: "우수", cls: "good", hint: "전반적으로 훌륭한 글이에요." };
  if (total >= 40) return { label: "양호", cls: "mid",  hint: "조금만 다듬으면 더 좋아질 거예요." };
  return { label: "보완 필요", cls: "low", hint: "몇 가지 부분을 고쳐 써 볼까요?" };
}

function categorySummary(criteria) {
  const cats = {};
  for (const c of criteria) {
    if (!cats[c.category]) cats[c.category] = { sum: 0, count: 0, max: 0 };
    if (c.score != null) cats[c.category].sum += c.score;
    cats[c.category].count += 1;
    cats[c.category].max += 9;
  }
  return cats;
}

function renderCategoryBars(criteria) {
  const cats = categorySummary(criteria);
  const order = ["과제", "내용", "조직", "표현"];
  $("category-bars").innerHTML = order
    .filter((k) => cats[k])
    .map((name) => {
      const v = cats[name];
      const pct = v.max ? (v.sum / v.max) * 100 : 0;
      return `
        <div class="cat-bar" data-cat="${name}">
          <div class="cat-bar-head">
            <span class="cat-name">${name}</span>
            <span class="cat-score">${v.sum} / ${v.max}</span>
          </div>
          <div class="cat-progress"><div class="cat-progress-fill" style="width:${pct}%"></div></div>
        </div>`;
    }).join("");
}

function renderDistBar(dist) {
  if (!Array.isArray(dist) || dist.length !== 9) return "";
  const maxP = Math.max(...dist);
  return `<div class="prob-row">${dist.map((p, i) => {
    const h = Math.max(2, Math.round((p / Math.max(maxP, 0.001)) * 28));
    const hot = p > 0 && p === maxP ? " hot" : "";
    return `<span class="prob-bar${hot}" style="height:${h}px" title="${i + 1}점: ${(p * 100).toFixed(1)}%"></span>`;
  }).join("")}<div class="prob-axis">${[1,2,3,4,5,6,7,8,9].map(n => `<span>${n}</span>`).join("")}</div></div>`;
}

function renderConfidence(conf) {
  if (!conf) return "";
  const scalar = (conf.confidence != null) ? conf.confidence : (conf.peak_prob || 0);
  const pct = Math.round(scalar * 100);
  const exp = (conf.expected_score != null) ? conf.expected_score.toFixed(2) : "—";
  return `
    <div class="crit-confidence">
      <span class="conf-label">모델 확신도</span>
      <span class="conf-bar"><span class="conf-bar-fill" style="width:${pct}%"></span></span>
      <span class="conf-value">${pct}%</span>
      <span class="conf-exp">기대값 ${exp}점</span>
    </div>`;
}

function renderCriteriaGrid(criteria, targetId = "criteria-grid") {
  $(targetId).innerHTML = criteria.map((c) => {
    const pct = c.score ? (c.score / 9) * 100 : 0;
    const meanTag = (c.expected_score != null)
      ? `<span class="crit-mean">기대값 ${c.expected_score.toFixed(2)}</span>`
      : "";
    // AI 보정: 보정 전(LoRA) 점수가 다르면 배지로 표시
    const adjusted = (c.lora_score != null && c.lora_score !== c.score);
    const adjustBadge = adjusted
      ? `<span class="crit-adjust ${c.score > c.lora_score ? "up" : "down"}">보정 전 ${c.lora_score}점</span>`
      : "";
    // 보정한 항목은 근거를 반드시, 도드라지게 노출한다(보정 방향 포함).
    // adjust_reason = "기존 채점과 왜 다르게 봤는지" (리포트 단계에서 생성),
    // agent_reasoning = 독립 판정 자체의 근거. 보정 카드엔 전자를 우선한다.
    const reasonLabel = adjusted
      ? `🔧 보정 근거 ${c.lora_score}→${c.score}점`
      : "🧠 AI 검증";
    const reasonText = adjusted
      ? (c.adjust_reason || c.agent_reasoning)
      : c.agent_reasoning;
    const reasonBlock = reasonText
      ? `<div class="crit-reason${adjusted ? " adjusted" : ""}"><span class="crit-reason-label">${reasonLabel}</span>${escapeHtml(reasonText)}</div>`
      : (adjusted
          ? `<div class="crit-reason adjusted"><span class="crit-reason-label">${reasonLabel}</span>근거가 제공되지 않았습니다.</div>`
          : "");
    const distBlock = Array.isArray(c.score_probs)
      ? `<div class="crit-probs"><div class="crit-probs-label">점수별 확률 (앙상블)</div>${renderDistBar(c.score_probs)}</div>`
      : "";
    const confBlock = renderConfidence(c.confidence);
    let body = "";
    if (Array.isArray(c.feedbacks) && c.feedbacks.length) {
      body = `
        <div class="crit-feedbacks">
          ${c.feedbacks.map((f, i) => `
            <div class="crit-fb-item">
              <div class="crit-fb-head">
                <span class="crit-fb-tag">샘플 ${i + 1}</span>
                <span class="crit-fb-score">${f.score ?? "—"}점</span>
              </div>
              <div class="crit-fb-text">${escapeHtml(f.feedback)}</div>
            </div>
          `).join("")}
        </div>`;
    } else if (c.feedback) {
      body = `<div class="crit-feedback">${escapeHtml(c.feedback)}</div>`;
    } else {
      body = `<div class="crit-feedback crit-feedback-empty">(피드백 없음)</div>`;
    }
    return `
      <div class="criterion" data-cat="${c.category}">
        <div class="crit-head">
          <div class="crit-title-wrap">
            <div class="crit-title">${escapeHtml(c.full)}</div>
            <div class="crit-cat">${escapeHtml(c.category)}</div>
          </div>
          <div class="crit-score-block">
            <div class="crit-score-num">${c.score ?? "—"}<span class="crit-score-max"> /9</span></div>
            ${c.percentile != null ? `<span class="crit-pct${c.percentile > 50 ? " bottom" : ""}">${pctLabel(c.percentile)}</span>` : ""}
            ${adjustBadge || meanTag}
          </div>
        </div>
        <div class="crit-bar"><div class="crit-bar-fill" style="width:${pct}%"></div></div>
        ${confBlock}
        ${reasonBlock}
        ${distBlock}
        ${body}
      </div>`;
  }).join("");
}

function renderRadar(values, labels) {
  const ctx = $("radar").getContext("2d");
  if (radarChart) radarChart.destroy();
  radarChart = new Chart(ctx, {
    type: "radar",
    data: {
      labels,
      datasets: [{
        label: "점수",
        data: values,
        backgroundColor: "rgba(44, 107, 237, 0.15)",
        borderColor: "rgba(44, 107, 237, 1)",
        borderWidth: 2,
        pointBackgroundColor: "rgba(127, 106, 240, 1)",
        pointBorderColor: "#fff",
        pointBorderWidth: 2,
        pointRadius: 4,
        pointHoverRadius: 6,
      }],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: { display: false },
        tooltip: { callbacks: { label: (c) => ` ${c.parsed.r ?? "—"} / 9점` } },
      },
      scales: {
        r: {
          suggestedMin: 0, suggestedMax: 9,
          grid: { color: "rgba(15, 27, 46, 0.06)" },
          angleLines: { color: "rgba(15, 27, 46, 0.08)" },
          pointLabels: {
            color: "#3F526B",
            font: { family: "Pretendard", size: 12, weight: "600" },
          },
          ticks: {
            stepSize: 3,
            color: "#9BA9BF",
            backdropColor: "transparent",
            font: { family: "Pretendard", size: 10 },
          },
        },
      },
    },
  });
}

function applyResult(data) {
  currentCriteria = data.criteria;
  $("result-topic").textContent = currentTopic || "—";

  $("total-score").textContent = data.total ?? "—";
  const band = gradeBand(data.total);
  const badge = $("grade-badge");
  badge.textContent = band.label;
  badge.className = "grade-badge " + band.cls;
  const ensembleNote = data.n_samples
    ? ` · ${data.n_samples}회 앙상블 (유효 ${data.n_valid})`
    : "";
  $("total-hint").textContent = band.hint + ensembleNote;

  renderCategoryBars(data.criteria);
  renderCriteriaGrid(data.criteria);
  renderRadar(
    data.criteria.map((c) => c.score ?? 0),
    data.criteria.map((c) => c.short || c.full),
  );

  const low = [...data.criteria].filter(c => c.score != null).sort((a, b) => a.score - b.score)[0];
  if (low) {
    $("hero-note").textContent = `가장 보완이 필요한 영역은 "${low.full}"이에요. 아래 상세 피드백을 참고해 보세요.`;
  } else {
    $("hero-note").textContent = "";
  }
}

// Re-render the result with the AI Agent's verified (corrected) scores as final.
// The original LoRA scores move into the collapsible "초기 채점 결과" reference.
function applyAgentResult(report) {
  if (!report || !Array.isArray(report.slots)) return;
  // Merge LoRA criteria (feedbacks/probs) with the agent's corrected score + reasoning.
  const agentCriteria = (currentCriteria || []).map((c, i) => {
    const s = report.slots[i] || {};
    const ag = s.agent || {};
    return {
      ...c,
      score: ag.score != null ? ag.score : c.score,
      lora_score: (s.lora && s.lora.score != null) ? s.lora.score : c.score,
      agent_reasoning: ag.reasoning || "",
      adjust_reason: ag.adjust_reason || "",
      percentile: ag.percentile != null ? ag.percentile : null,
    };
  });

  const agentTotal = report.totals ? report.totals.agent : null;
  const loraTotal = report.totals ? report.totals.lora : null;
  $("total-score").textContent = agentTotal ?? "—";
  const band = gradeBand(agentTotal);
  const badge = $("grade-badge");
  badge.textContent = band.label;
  badge.className = "grade-badge " + band.cls;
  $("total-hint").textContent = band.hint;

  // 논제별 백분위 (전체 점수)
  const tp = $("total-pct");
  if (report.totals && report.totals.percentile != null) {
    const p = report.totals.percentile;
    tp.textContent = pctLabel(p);
    tp.classList.toggle("bottom", p > 50);
    tp.title = report.percentile_n
      ? `이 논제 응시 ${report.percentile_n.toLocaleString()}편 기준` : "";
    tp.hidden = false;
  } else {
    tp.hidden = true;
  }

  // 보정 전/후 비교 라벨
  const adj = $("hero-adjust");
  if (loraTotal != null && agentTotal != null) {
    const d = agentTotal - loraTotal;
    const dtxt = d === 0 ? "보정 없음" : `${d < 0 ? "▼" : "▲"} ${Math.abs(d)}점`;
    const cls = d < 0 ? "down" : (d > 0 ? "up" : "same");
    adj.innerHTML = `<span class="ha-label">AI Agent 보정 점수</span>`
      + `<span class="ha-delta ${cls}">${dtxt}</span>`
      + `<span class="ha-ref">보정 전(LoRA) ${loraTotal}점</span>`;
    adj.hidden = false;
  }

  renderCategoryBars(agentCriteria);
  renderCriteriaGrid(agentCriteria, "criteria-grid");
  renderRadar(
    agentCriteria.map((c) => c.score ?? 0),
    agentCriteria.map((c) => c.short || c.full),
  );
  $("criteria-badge").hidden = false;

  // 하단 접이식 탭들 노출
  $("audit-section").hidden = false;       // 도구 호출 결과
  renderCriteriaGrid(currentCriteria || [], "ref-criteria-grid");
  $("ref-summary").textContent = loraTotal != null ? `총점 ${loraTotal} / 72점` : "";
  $("ref-section").hidden = false;         // 초기 채점(LoRA)
  $("linguistic-section").hidden = false;  // 언어 분석 (펼치면 온디맨드 로드)
}

// ── Agent verify ──────────────────────────────────────────────────
// Covers both the agent-facing tool names (what the model calls) and the
// underlying evidence tool names (what the evidence objects carry).
const TOOL_INFO = {
  // agent-facing names
  rubric_criteria:       { name: "루브릭 기준 조회",        src: "data/rubric_criteria.json",          ico: "📋" },
  check_orthography:     { name: "맞춤법·띄어쓰기 교차검증", src: "BAREUN · KIWI · ETRI · 어문규범",     ico: "🔤" },
  check_keywords:        { name: "핵심 키워드 충족",        src: "data/topics.json",                   ico: "🎯" },
  check_vocabulary:      { name: "어휘 등급 확인",          src: "한국어기초사전 · 우리말샘",          ico: "📖" },
  check_terminology:     { name: "전문어 확인",            src: "국립국어원 온용어(K-term)",          ico: "🔬" },
  statistical_prior:     { name: "통계적 사전(참고)",       src: "scoring_rules · 학습데이터 분석",     ico: "📊" },
  norm_search:           { name: "어문규범 검색",          src: "국립국어원 어문 규범",               ico: "⚖️" },
  perplexity_probe:      { name: "문장 자연스러움(PPL)",    src: "Kanana base · perplexity",          ico: "📈" },
  reparse_request:       { name: "JSON 재요청",            src: "agent",                              ico: "↻" },
  verdict_parse:         { name: "검증 결과 파싱",          src: "agent",                              ico: "🧩" },
  final_report:          { name: "종합 리포트 작성",        src: "agent",                              ico: "📝" },
  // evidence-object names (carried in report.slots[].evidence)
  rubric_retrieve:       { name: "루브릭 기준 조회",        src: "data/rubric_criteria.json",          ico: "📋" },
  orthography_probe:     { name: "맞춤법·띄어쓰기 교차검증", src: "BAREUN · KIWI · ETRI · 어문규범",     ico: "🔤" },
  lexical_grounding:     { name: "어휘 등급 확인",          src: "한국어기초사전 · 우리말샘",          ico: "📖" },
  terminology_grounding: { name: "전문어 확인",            src: "국립국어원 온용어(K-term)",          ico: "🔬" },
  keyword_coverage:      { name: "핵심 키워드 충족",        src: "data/topics.json",                   ico: "🎯" },
};

function toolInfo(tool) {
  const base = tool.split(":")[0];
  return TOOL_INFO[base] || { name: tool, src: "", ico: "•" };
}

// ── Live activity feed (streamed tool calls) ──────────────────────
function resetActivity() {
  $("activity-feed").innerHTML = "";
  const think = $("activity-think");
  think.hidden = true;
  think.textContent = "";
}

function activityCall(tool, args) {
  const info = toolInfo(tool);
  const argTxt = args && Object.keys(args).length
    ? ` <span class="act-args">${escapeHtml(JSON.stringify(args))}</span>` : "";
  const row = document.createElement("div");
  row.className = "act-row pending";
  row.innerHTML = `
    <span class="act-dot"></span>
    <span class="tool-ico">${info.ico}</span>
    <div class="act-main">
      <div class="act-name">${info.name}${argTxt}</div>
      <div class="act-src">${escapeHtml(info.src)}</div>
      <div class="act-result"></div>
    </div>
    <span class="act-ms"></span>`;
  $("activity-feed").appendChild(row);
  row.scrollIntoView({ block: "nearest" });
  return row;
}

function activityResult(row, ev) {
  if (!row) return;
  const ok = ev.status === "ok";
  row.className = `act-row ${ok ? "ok" : "err"}`;
  row.querySelector(".act-ms").textContent = ev.ms != null ? `${ev.ms}ms` : "";
  const res = row.querySelector(".act-result");
  res.textContent = ev.summary || (ok ? "완료" : "오류");
  if (!ok) res.classList.add("err");
}

function setActStatus(txt) {
  const el = $("activity-status");
  if (txt == null) { el.hidden = true; el.textContent = ""; return; }
  el.hidden = false;
  el.textContent = txt;
}

async function runVerify() {
  if (!currentCriteria) return false;
  $("processing").hidden = false;
  $("proc-title").textContent = "AI 검증관이 도구를 호출하는 중…";
  resetActivity();
  setActStatus("도구 실행 준비 중…");

  let pendingRow = null;
  let doneReport = null;
  let toolsDone = 0;

  try {
    const r = await fetch("/api/agent/verify/stream", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        topic: currentTopic,
        essay: currentEssay,
        keywords: currentKeywords || null,
        criteria: currentCriteria,
      }),
    });
    if (!r.ok || !r.body) throw new Error(`요청 실패 (${r.status})`);

    const reader = r.body.getReader();
    const decoder = new TextDecoder("utf-8");
    let carry = "";

    while (true) {
      const { value, done } = await reader.read();
      if (done) break;
      carry += decoder.decode(value, { stream: true });
      let idx;
      while ((idx = carry.indexOf("\n\n")) >= 0) {
        const frame = carry.slice(0, idx);
        carry = carry.slice(idx + 2);
        const ev = parseSSE(frame);
        if (!ev) continue;
        if (ev.event === "start") {
          setActStatus("도구 실행 중…");
        } else if (ev.event === "token") {
          // tools are done; the agent is now writing its verdict
          setActStatus("🧠 검증 결과를 정리하고 있어요…");
          const think = $("activity-think");
          think.hidden = false;
          think.textContent += ev.data.text || "";
          think.scrollTop = think.scrollHeight;
        } else if (ev.event === "tool_call") {
          pendingRow = activityCall(ev.data.tool, ev.data.args);
        } else if (ev.event === "tool_result") {
          activityResult(pendingRow, ev.data);
          pendingRow = null;
          toolsDone += 1;
          setActStatus(`🔧 도구 실행 중 · ${toolsDone}개 완료`);
        } else if (ev.event === "done") {
          doneReport = ev.data;
        } else if (ev.event === "error") {
          throw new Error(ev.data.message || "검증 오류");
        }
      }
    }
    if (!doneReport) throw new Error("검증 응답이 완결되지 않았어요.");

    setActStatus(null);
    renderAudit(doneReport);   // builds the agent-final report + the 도구 호출 결과 tab
    return true;
  } catch (e) {
    toast(`AI 검증 오류: ${e.message || e}`);
    return false;
  } finally {
    setActStatus(null);
    $("processing").hidden = true;
  }
}

// agent-facing tool name → evidence-object tool name (for modal lookup)
const AGENT_TO_EVIDENCE = {
  check_orthography: "orthography_probe",
  check_keywords: "keyword_coverage",
  check_vocabulary: "lexical_grounding",
  check_terminology: "terminology_grounding",
  rubric_criteria: "rubric_retrieve",
  statistical_prior: "statistical_prior",
  norm_search: "norm_search",
  perplexity_probe: "perplexity_probe",
};

function buildEvidenceIndex(report) {
  const idx = {};
  for (const s of (report.slots || [])) {
    for (const e of (s.evidence || [])) {
      (idx[e.tool] = idx[e.tool] || []).push(e);
    }
  }
  return idx;
}

function renderAudit(report) {
  currentReport = report;
  report._evidenceByTool = buildEvidenceIndex(report);
  renderAuditCompare(report.totals, report.n_flagged, report.overall);
  renderAuditTools(report.trace);
  renderAuditFlags(report.slots);
  renderOverallReport(report);   // 종합 리포트 (총평·강점·개선·보정 내역)
  applyAgentResult(report);   // promote the agent's corrected scores to the final result
}

// ── AI 검증관 종합 리포트 (result page, prominent) ─────────────────
function renderOverallReport(report) {
  const el = $("overall-report");
  const ov = report.overall || {};
  const adjusted = (report.slots || []).filter((s) => s.agent && s.agent.flag);

  const summary = ov.summary
    ? `<p class="report-summary">${escapeHtml(ov.summary)}</p>` : "";
  const li = (arr) => arr.map((x) => `<li>${escapeHtml(x)}</li>`).join("");
  const strengths = (ov.strengths || []).length
    ? `<div class="report-col strengths"><h4>👍 강점</h4><ul>${li(ov.strengths)}</ul></div>` : "";
  const improvements = (ov.improvements || []).length
    ? `<div class="report-col improvements"><h4>✏️ 개선 제안</h4><ul>${li(ov.improvements)}</ul></div>` : "";
  const cols = (strengths || improvements)
    ? `<div class="report-cols">${strengths}${improvements}</div>` : "";

  const adjRows = adjusted.map((s) => {
    const d = s.agent.score - s.lora.score;
    return `
      <div class="adj-row">
        <span class="adj-name">${escapeHtml(s.rubric_name)}</span>
        <span class="adj-score ${d < 0 ? "down" : "up"}">${s.lora.score}점 → <b>${s.agent.score}점</b></span>
        <span class="adj-reason">${escapeHtml(s.agent.adjust_reason || s.agent.reasoning || "")}</span>
      </div>`;
  }).join("");
  const adjBlock = adjusted.length
    ? `<div class="report-adjust"><h4>🔧 점수 보정 내역 · ${adjusted.length}건</h4>${adjRows}</div>`
    : `<div class="report-adjust none">외부 도구 증거가 모든 항목에서 초기 채점과 부합해 점수를 그대로 유지했어요.</div>`;

  const body = summary + cols + adjBlock;
  if (!body.trim()) { el.hidden = true; return; }
  $("report-body").innerHTML = body;
  el.hidden = false;
}

const CONF_LABEL = { high: "높음", medium: "보통", low: "낮음" };

function renderAuditCompare(totals, nFlagged, overall) {
  const delta = totals.delta;
  const deltaCls = delta < 0 ? "down" : (delta > 0 ? "up" : "same");
  const deltaTxt = delta === 0 ? "동일" : `${delta < 0 ? "▼" : "▲"} ${Math.abs(delta)}점`;
  const conf = overall && overall.confidence ? CONF_LABEL[overall.confidence] || overall.confidence : null;
  const summaryBlock = (overall && overall.summary)
    ? `<div class="audit-summary">🧠 <strong>검증관 총평${conf ? ` · 확신도 ${conf}` : ""}</strong> ${escapeHtml(overall.summary)}</div>`
    : "";
  $("audit-compare").innerHTML = summaryBlock + `
    <div class="cmp-card">
      <div class="cmp-label">LoRA 채점</div>
      <div class="cmp-num">${totals.lora}<span>/ ${totals.max}</span></div>
      <div class="cmp-sub">채점 모델의 가설</div>
    </div>
    <div class="cmp-arrow ${deltaCls}">
      <div class="cmp-delta ${deltaCls}">${deltaTxt}</div>
      <div class="cmp-arrow-line"></div>
      <div class="cmp-flag">${nFlagged}개 항목 의심</div>
    </div>
    <div class="cmp-card cmp-card-agent">
      <div class="cmp-label">Agent 감사</div>
      <div class="cmp-num">${totals.agent}<span>/ ${totals.max}</span></div>
      <div class="cmp-sub">외부 근거로 교차 검증</div>
    </div>`;
}

function renderAuditTools(trace) {
  // Aggregate repeated calls (rubric_criteria/statistical_prior run per slot) into one row.
  const agg = {};
  const order = [];
  for (const t of (trace || [])) {
    if (t.tool === "reparse_request" || t.tool === "verdict_parse") continue;
    if (!agg[t.tool]) { agg[t.tool] = { tool: t.tool, ms: 0, n: 0, status: "ok", message: "" }; order.push(t.tool); }
    agg[t.tool].ms += t.ms || 0; agg[t.tool].n += 1;
    if (t.status !== "ok") { agg[t.tool].status = "error"; agg[t.tool].message = t.message || ""; }
  }

  const chips = order.map((key) => {
    const t = agg[key];
    const info = toolInfo(t.tool);
    const evName = AGENT_TO_EVIDENCE[t.tool];
    const hasDetail = !!(evName && currentReport && currentReport._evidenceByTool[evName]);
    const ok = t.status === "ok";
    const cnt = t.n > 1 ? `<span class="tool-cnt">×${t.n}</span>` : "";
    const msg = t.message ? `<div class="tool-err">${escapeHtml(t.message)}</div>` : "";
    return `
      <div class="tool-row ${ok ? "ok" : "err"} ${hasDetail ? "clickable" : ""}" ${hasDetail ? `data-tool="${evName}"` : ""}>
        <span class="tool-dot"></span>
        <span class="tool-ico">${info.ico}</span>
        <div class="tool-main">
          <div class="tool-name">${info.name}${cnt}</div>
          <div class="tool-src">${escapeHtml(info.src)}</div>
          ${msg}
        </div>
        ${hasDetail ? `<span class="tool-detail-hint">상세 ›</span>` : ""}
        <span class="tool-ms">${t.ms != null ? t.ms + "ms" : ""}</span>
      </div>`;
  }).join("");

  $("audit-tools").innerHTML = `
    <div class="audit-block-head"><h4>도구 검증 내역</h4>
      <span class="audit-block-sub">행을 클릭하면 도구별 상세 결과를 볼 수 있어요.</span></div>
    <div class="tool-list">${chips}</div>`;

  $("audit-tools").querySelectorAll(".tool-row.clickable").forEach((el) => {
    el.addEventListener("click", () => openToolModal(el.dataset.tool));
  });
}

function findEvidence(slots, slotKey, tool) {
  const s = slots.find((x) => x.slot === slotKey);
  if (!s) return null;
  return (s.evidence || []).find((e) => e.tool === tool) || null;
}

// ── Inline-highlight tool detail: full essay text with clickable highlights ──
// Generic: render `text` with `spans` ({start,end,cls,detail}) marked; clicking a
// mark shows its detail HTML in the panel below. Used by the tool-detail modal.
let _hlDetails = [];

function escapeAttr(s) { return escapeHtml(s).replace(/"/g, "&quot;"); }

function occurrences(text, word) {
  const out = [];
  if (!word) return out;
  let i = 0;
  while ((i = text.indexOf(word, i)) !== -1) { out.push([i, i + word.length]); i += word.length; }
  return out;
}

function highlightDoc(text, spans, emptyMsg) {
  const clean = (spans || []).filter((s) => s.start >= 0 && s.end > s.start)
    .sort((a, b) => a.start - b.start || b.end - a.end);
  let html = "", pos = 0;
  _hlDetails = [];
  for (const s of clean) {
    if (s.start < pos) continue;            // drop overlaps
    if (s.start > pos) html += escapeHtml(text.slice(pos, s.start));
    const i = _hlDetails.length; _hlDetails.push(s.detail || "");
    html += `<mark class="hl ${s.cls}" data-hi="${i}">${escapeHtml(text.slice(s.start, s.end))}</mark>`;
    pos = s.end;
  }
  if (pos < text.length) html += escapeHtml(text.slice(pos));
  const detail = `<div class="hl-detail" id="hl-detail"><span class="md-muted">${emptyMsg || "강조된 부분을 클릭하면 상세가 표시돼요."}</span></div>`;
  return `<div class="hl-doc">${html || escapeHtml(text)}</div>${detail}`;
}

// Norm article shown inline (full body + examples), no external link.
function normDetailHtml(norm) {
  if (!norm) return `<div class="hl-norm none">연결된 어문규범 조항 없음</div>`;
  const head = `${escapeHtml((norm.category || "") + " " + (norm.article || ""))} ${escapeHtml(norm.title || "")}`.trim();
  const ex = (norm.examples && norm.examples.length)
    ? `<div class="hl-norm-ex">예: ${norm.examples.slice(0, 3).map(escapeHtml).join(" / ")}</div>` : "";
  const via = norm.matched_by === "search" ? ` <span class="hl-tag">검색</span>` : "";
  return `<div class="hl-norm"><div class="hl-norm-head">${head}${via}</div>`
    + `<div class="hl-norm-body">${escapeHtml(norm.body || "")}</div>${ex}</div>`;
}

function strengthBadge(strength) {
  const map = { strong: ["강한 근거", "strong"], weak: ["약한 근거", "weak"], none: ["참고", "none"] };
  const [txt, cls] = map[strength] || map.none;
  return `<span class="ev-strength ${cls}">${txt}</span>`;
}

function renderAuditFlags(slots) {
  const flagged = slots.filter((s) => s.agent.flag);
  if (!flagged.length) {
    $("audit-flags").innerHTML = `
      <div class="audit-block-head"><h4>감사 결과</h4></div>
      <div class="mk-clean">외부 도구가 채점 결과와 충돌하는 신호를 찾지 못했어요. 채점이 신뢰할 만해요.</div>`;
    return;
  }
  const cards = flagged.map((s) => {
    const a = s.agent, l = s.lora;
    const changed = a.score !== l.score;
    const scoreLine = changed
      ? `<span class="fl-lora">${l.score}</span><span class="fl-arrow">→</span><span class="fl-agent">${a.score}</span><span class="fl-unit">/9</span>`
      : `<span class="fl-agent">${a.score}</span><span class="fl-unit">/9</span> <span class="fl-keep">유지</span>`;
    const sev = { high: "높음", medium: "중간", low: "낮음", none: "" }[a.severity] || "";
    const sevBadge = sev ? `<span class="fl-sev ${a.severity}">의심도 ${sev}</span>` : "";

    const evRows = (s.evidence || [])
      .filter((e) => e.tool !== "rubric_retrieve")
      .map((e) => {
        const info = toolInfo(e.tool);
        let norms = "";
        if (e.tool === "orthography_probe") {
          const cites = (e.spans || []).filter((x) => x.norm)
            .map((x) => `<a class="ev-norm" href="${escapeAttr(x.norm.source_url || "")}" target="_blank" rel="noopener">${escapeHtml(x.norm.category)} ${escapeHtml(x.norm.article)} · ${escapeHtml(x.norm.body)}</a>`);
          if (cites.length) norms = `<div class="ev-norms">${[...new Set(cites)].join("")}</div>`;
        }
        return `
          <div class="ev-row">
            <span class="ev-ico">${info.ico}</span>
            <div class="ev-main">
              <div class="ev-summary">${escapeHtml(e.summary)} ${strengthBadge(e.strength)}</div>
              <div class="ev-src">${escapeHtml(info.src)}</div>
              ${norms}
            </div>
          </div>`;
      }).join("");

    return `
      <div class="flag-card">
        <div class="flag-head">
          <div class="flag-title-wrap">
            <span class="flag-dot"></span>
            <div>
              <div class="flag-title">${escapeHtml(s.rubric_name)}</div>
              <div class="flag-cat">${escapeHtml(s.category || "")}</div>
            </div>
          </div>
          <div class="flag-score">${scoreLine} ${sevBadge}</div>
        </div>
        ${a.deterministic_note ? `<div class="flag-note">📐 ${escapeHtml(a.deterministic_note)}</div>` : ""}
        ${a.adjust_reason ? `<div class="flag-reason"><span class="flag-reason-label">🔧 보정 이유</span>${escapeHtml(a.adjust_reason)}</div>` : ""}
        ${a.reasoning ? `<div class="flag-reason"><span class="flag-reason-label">Agent 판정 근거</span>${escapeHtml(a.reasoning)}</div>` : ""}
        <div class="ev-list">${evRows}</div>
      </div>`;
  }).join("");

  $("audit-flags").innerHTML = `
    <div class="audit-block-head"><h4>의심 항목 상세 <span class="audit-block-sub">${flagged.length}개</span></h4></div>
    <div class="flag-grid">${cards}</div>`;
}

// Clicking a highlighted span updates the detail panel below the document.
document.addEventListener("click", (e) => {
  const m = e.target.closest("mark.hl[data-hi]");
  if (!m) return;
  const panel = document.getElementById("hl-detail");
  if (!panel) return;
  const idx = parseInt(m.getAttribute("data-hi"), 10);
  panel.innerHTML = (_hlDetails && _hlDetails[idx]) || "";
  m.closest(".md-detail, .modal-body, body")
    ?.querySelectorAll("mark.hl.active").forEach((x) => x.classList.remove("active"));
  m.classList.add("active");
});

// ── Tool detail modal ─────────────────────────────────────────────
function openToolModal(evName) {
  const list = currentReport && currentReport._evidenceByTool[evName];
  if (!list || !list.length) return;
  const info = toolInfo(evName);
  $("modal-ico").textContent = info.ico;
  $("modal-title").textContent = info.name;
  $("modal-body").innerHTML = renderToolDetail(evName, list);
  $("tool-modal").hidden = false;
  document.body.classList.add("modal-open");
}
function closeToolModal() {
  $("tool-modal").hidden = true;
  document.body.classList.remove("modal-open");
}

function kv(label, value) {
  return `<div class="md-kv"><span class="md-k">${escapeHtml(label)}</span><span class="md-v">${value}</span></div>`;
}

function renderToolDetail(evName, list) {
  const ev = list[0];
  const sig = ev.signals || {};
  if (evName === "orthography_probe") {
    const rate = sig.spacing_error_rate != null ? (sig.spacing_error_rate * 100).toFixed(2) + "%" : "—";
    const stats = `<div class="md-stats">
      ${kv("교정 후보", (sig.n_corrections ?? 0) + "건")}
      ${kv("띄어쓰기", (sig.n_spacing ?? 0) + "건")}
      ${kv("맞춤법", (sig.n_typo ?? 0) + "건")}
      ${kv("KIWI 교차검증", (sig.n_strong ?? 0) + "건")}
      ${kv("띄어쓰기 오류율", rate)}
    </div>`;
    const marks = (ev.spans || []).map((s) => {
      const kind = s.category === "SPACING" ? "띄어쓰기" : "맞춤법";
      const cross = (s.kiwi_agrees || s.kiwi_morp_agrees) ? " · KIWI 교차검증 ✓" : "";
      const detail = `<div class="hl-d-head"><span class="md-bad">${escapeHtml(s.origin || "")}</span>`
        + ` → <span class="md-good">${escapeHtml(s.revised || "")}</span></div>`
        + `<div class="hl-d-row">유형: ${kind}${cross}</div>`
        + normDetailHtml(s.norm);
      return { start: s.begin, end: s.end,
               cls: s.category === "SPACING" ? "hl-spacing" : "hl-typo", detail };
    });
    const body = marks.length
      ? highlightDoc(currentEssay, marks, "강조된 오류를 클릭하면 교정안·어문규범 조항이 표시돼요.")
      : `<div class="md-empty">맞춤법·띄어쓰기 의심 구간이 없어요.</div>`;
    return stats + `<p class="md-muted">밑줄 친 구간이 의심 오류예요. 클릭하면 상세가 보여요.</p>` + body;
  }
  if (evName === "keyword_coverage") {
    const covered = sig.covered || [], missing = sig.missing || [];
    const ratio = sig.coverage_ratio != null ? Math.round(sig.coverage_ratio * 100) + "%" : "—";
    const marks = [];
    covered.forEach((k) => occurrences(currentEssay, k).forEach(([a, b]) =>
      marks.push({ start: a, end: b, cls: "hl-kw",
                   detail: `<div class="hl-d-head">${escapeHtml(k)}</div><div class="hl-d-row">논제 핵심 키워드 — 충족</div>` })));
    const missChips = missing.map((k) => `<span class="md-chip miss">${escapeHtml(k)}</span>`).join("")
      || "<span class='md-muted'>없음</span>";
    return `<div class="md-stats">${kv("충족률", ratio)}${kv("핵심어 수", sig.n_total ?? 0)}</div>`
      + `<p class="md-muted">본문에서 충족된 키워드를 표시했어요.</p>`
      + highlightDoc(currentEssay, marks, "충족된 키워드가 본문에 표시돼요.")
      + `<div class="md-group"><div class="md-group-title">누락된 키워드</div><div class="md-chips">${missChips}</div></div>`;
  }
  if (evName === "lexical_grounding") {
    const gradeCls = { "초급": "hl-g1", "중급": "hl-g2", "고급": "hl-g3" };
    const marks = [];
    (sig.graded || []).forEach((g) => {
      const grade = g.word_grade || (g.in_opendict ? "우리말샘 등재" : "등재");
      const cls = gradeCls[g.word_grade] || "hl-g0";
      const detail = `<div class="hl-d-head">${escapeHtml(g.word || "")} <span class="hl-tag">${escapeHtml(grade)}</span></div>`
        + (g.pos ? `<div class="hl-d-row">품사: ${escapeHtml(g.pos)}</div>` : "")
        + `<div class="hl-d-body">${escapeHtml(g.definition || "")}</div>`;
      occurrences(currentEssay, g.word).forEach(([a, b]) => marks.push({ start: a, end: b, cls, detail }));
    });
    (sig.unverified || []).forEach((w) => occurrences(currentEssay, w).forEach(([a, b]) =>
      marks.push({ start: a, end: b, cls: "hl-unv",
                   detail: `<div class="hl-d-head">${escapeHtml(w)}</div><div class="hl-d-row">사전 미확인 — 신조어·고유명사이거나 오기일 수 있어요.</div>` })));
    const legend = `<div class="hl-legend">
      <span><i class="hl hl-g1"></i>초급</span><span><i class="hl hl-g2"></i>중급</span>
      <span><i class="hl hl-g3"></i>고급</span><span><i class="hl hl-g0"></i>등재(등급없음)</span>
      <span><i class="hl hl-unv"></i>사전 미확인</span></div>`;
    return `<div class="md-stats">${kv("등급 분포", escapeHtml(JSON.stringify(sig.grade_counts || {})))}</div>`
      + legend + highlightDoc(currentEssay, marks, "어휘를 클릭하면 등급·뜻풀이가 보여요.");
  }
  if (evName === "terminology_grounding") {
    const marks = [];
    (sig.grounded || []).forEach((t) => {
      const detail = `<div class="hl-d-head">${escapeHtml(t.word || "")} ${t.origin ? `<span class="md-muted">${escapeHtml(t.origin)}</span>` : ""}`
        + `${t.category_sub ? ` <span class="hl-tag">${escapeHtml(t.category_sub)}</span>` : ""}</div>`
        + `<div class="hl-d-body">${escapeHtml(t.definition || "")}</div>`;
      occurrences(currentEssay, t.word).forEach(([a, b]) => marks.push({ start: a, end: b, cls: "hl-term", detail }));
    });
    const not = (sig.not_terms || []).map((w) => `<span class="md-chip">${escapeHtml(w)}</span>`).join("")
      || "<span class='md-muted'>없음</span>";
    return `<div class="md-stats">${kv("전문어 인정", (sig.n_grounded ?? 0) + "개")}${kv("분야", (sig.categories || []).join(", ") || "—")}</div>`
      + `<p class="md-muted">전문어로 확인된 어휘를 표시했어요. 클릭하면 뜻풀이가 보여요.</p>`
      + highlightDoc(currentEssay, marks, "전문어를 클릭하면 상세가 보여요.")
      + `<div class="md-group"><div class="md-group-title">전문어 아님</div><div class="md-chips">${not}</div></div>`;
  }
  if (evName === "rubric_retrieve") {
    // one evidence per slot — show each slot's 5-level ladder with current highlighted
    return list.map((e) => {
      const s = e.signals || {};
      const ladder = s.ladder || {};
      const levels = ["evaluation_1", "evaluation_2", "evaluation_3", "evaluation_4", "evaluation_5"];
      const rows = levels.map((lv, i) => `<div class="md-ladder-row ${lv === s.level ? "cur" : ""}">
        <span class="md-ladder-lv">${i + 1}단계</span>
        <span class="md-ladder-txt">${escapeHtml(ladder[lv] || "")}</span></div>`).join("");
      return `<div class="md-group"><div class="md-group-title">${escapeHtml(s.name || e.slot)} · ${s.score ?? "—"}점</div>${rows}</div>`;
    }).join("");
  }
  if (evName === "statistical_prior") {
    return list.map((e) => {
      const s = e.signals || {};
      const notes = (s.notes || []).map((n) => `<li>${escapeHtml(n)}</li>`).join("") || "<li class='md-muted'>규칙 없음</li>";
      const sug = s.suggested_score != null ? ` · 제안 ${s.suggested_score}점` : "";
      return `<div class="md-group"><div class="md-group-title">${escapeHtml(e.slot)} (LoRA ${s.lora_score ?? "—"}점${sug})</div><ul class="md-notes">${notes}</ul></div>`;
    }).join("") + `<p class="md-muted" style="margin-top:10px">※ 학습 데이터 통계 기반 참고치이며 점수를 강제하지 않아요.</p>`;
  }
  if (evName === "perplexity_probe") {
    const sents = (sig.sentences || []).slice();
    // 흐름 문장(첫 문장 제외)의 중앙값 대비 상대값으로 거칠기를 판단 → outlier가 전체 스케일을 먹지 않게.
    const vals = sents.map((s) => s.ppl).filter((v) => v != null && v > 0).sort((a, b) => a - b);
    const median = vals.length ? vals[Math.floor(vals.length / 2)] : 0;
    const heat = (p) => {
      if (!median || p == null) return 0;
      const r = p / median;
      if (r >= 2.0) return 2;   // 매우 거칢
      if (r >= 1.4) return 1;   // 다소 거칢
      return 0;                 // 매끄러움
    };
    // 원문 전체를 이어서 보여주고, 각 문장 끝에 흐름 PPL을 인라인으로 붙인다.
    // 첫 문장은 앞 문맥이 없어 흐름 PPL이 없으므로(ppl=null) '시작'으로만 표시한다.
    const flow = sents.map((s) => {
      if (s.ppl == null) {
        return `<span class="ppl-seg seg-anchor">${escapeHtml(s.text || "")}` +
          `<sup class="ppl-tag" title="앞 문맥이 없는 시작 문장이라 흐름 PPL을 재지 않아요">시작</sup></span>`;
      }
      const lvl = heat(s.ppl);
      const cls = lvl === 2 ? "seg-hot" : (lvl === 1 ? "seg-warm" : "");
      return `<span class="ppl-seg ${cls}">${escapeHtml(s.text || "")}` +
        `<sup class="ppl-tag" title="앞 문맥에 이어질 때의 perplexity">${s.ppl.toFixed(0)}</sup></span>`;
    }).join(" ") || `<span class="md-empty">문장이 분석되지 않았어요.</span>`;
    return `<div class="md-stats">${kv("흐름 perplexity 평균", sig.overall_ppl ?? "—")}${kv("분석 토큰", sig.n_tokens ?? "—")}</div>
      <p class="md-muted">base 모델(LoRA OFF)이 잰 <b>문장 흐름</b> perplexity예요. 각 문장이 <b>앞 문맥에 얼마나 자연스럽게 이어지는지</b>를 뜻하고, 숫자가 높을수록 연결이 어색합니다. 첫 문장은 앞 문맥이 없어 측정하지 않아요. 노란/빨간 표시는 같은 글 안에서 상대적으로 거친 문장이에요.</p>
      <div class="ppl-flow">${flow}</div>`;
  }
  if (evName === "norm_search") {
    // agent may have searched several times — show each query and its top articles
    return list.map((e) => {
      const s = e.signals || {};
      const matches = s.matches || [];
      const rows = matches.map((m) => `<div class="hl-norm">
        <div class="hl-norm-head">${escapeHtml((m.category || "") + " " + (m.article || ""))} ${escapeHtml(m.title || "")}</div>
        <div class="hl-norm-body">${escapeHtml(m.body || "")}</div>
        ${(m.examples && m.examples.length) ? `<div class="hl-norm-ex">예: ${m.examples.slice(0,3).map(escapeHtml).join(" / ")}</div>` : ""}</div>`).join("")
        || `<div class="md-empty">일치하는 조항이 없어요.</div>`;
      return `<div class="md-group"><div class="md-group-title">🔎 검색어: "${escapeHtml(s.query || "")}"</div></div>${rows}`;
    }).join("");
  }
  return `<pre class="md-raw">${escapeHtml(JSON.stringify(sig, null, 2))}</pre>`;
}

// ── 언어 분석 (형태소 · 의존구문 · 의미역) ─────────────────────────
async function loadLinguistic() {
  if (linguisticLoaded || !currentEssay) return;
  linguisticLoaded = true;
  $("ling-loading").hidden = false;
  $("ling-error").hidden = true;
  $("ling-body").innerHTML = "";
  try {
    const r = await fetch("/api/analyze/linguistic", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ essay: currentEssay }),
    });
    if (!r.ok) throw new Error(await r.text());
    const data = await r.json();
    renderLinguistic(data);
    $("ling-loading").hidden = true;
  } catch (e) {
    $("ling-loading").hidden = true;
    linguisticLoaded = false;   // allow retry on re-open
    const el = $("ling-error");
    el.hidden = false;
    el.textContent = `언어 분석 중 오류: ${e.message || e}`;
  }
}

function posCategory(tag) {
  if (!tag) return "other";
  const c = tag[0];
  if (c === "N") return "noun";
  if (c === "V") return "verb";
  if (c === "J") return "particle";
  if (c === "E") return "ending";
  if (c === "M") return "modifier";
  return "other";
}

function renderLinguistic(data) {
  const sents = data.sentences || [];
  if (!sents.length) { $("ling-body").innerHTML = `<div class="md-empty">분석 결과가 없어요.</div>`; return; }
  $("ling-body").innerHTML = sents.map((s, i) => `
    <div class="ling-sent">
      <div class="ling-sent-head"><span class="ling-sent-no">문장 ${i + 1}</span>${escapeHtml(s.text || "")}</div>
      <div class="ling-block"><div class="ling-block-title">형태소 분석</div>${renderMorphemes(s.words)}</div>
      <div class="ling-block"><div class="ling-block-title">의존 구문 분석</div>${renderDependency(s.words, s.dependency)}</div>
      <div class="ling-block"><div class="ling-block-title">의미역 분석 (SRL)</div>${renderSRL(s.words, s.srl)}</div>
    </div>`).join("");
}

function renderMorphemes(words) {
  if (!words || !words.length) return `<div class="md-empty">—</div>`;
  return `<div class="morph-row">` + words.map((w) => {
    const chips = (w.morphemes || []).map((m) =>
      `<span class="morph-chip ${posCategory(m.tag)}">${escapeHtml(m.lemma)}<span class="morph-tag">${escapeHtml(m.tag)}</span></span>`
    ).join("");
    return `<div class="morph-word"><div class="morph-surface">${escapeHtml(w.text)}</div><div class="morph-chips">${chips}</div></div>`;
  }).join("") + `</div>`;
}

function renderDependency(words, deps) {
  if (!words || !words.length || !deps || !deps.length) return `<div class="md-empty">—</div>`;
  const COL = 96, nodeY = 84, wordY = 104, height = 116;
  const width = Math.max(words.length * COL, 200);
  const cx = (i) => i * COL + COL / 2;
  let arcs = "";
  for (const d of deps) {
    const from = d.id;
    const head = d.head;
    if (head == null || head < 0 || head === from) {
      arcs += `<text x="${cx(from)}" y="22" class="dep-root">ROOT</text>
        <line x1="${cx(from)}" y1="28" x2="${cx(from)}" y2="${nodeY}" class="dep-rootline"/>`;
      continue;
    }
    const x1 = cx(from), x2 = cx(head);
    const dist = Math.abs(from - head);
    const apex = nodeY - 16 - Math.min(dist, 5) * 11;
    const mx = (x1 + x2) / 2;
    arcs += `<path d="M ${x2} ${nodeY} C ${x2} ${apex}, ${x1} ${apex}, ${x1} ${nodeY}" class="dep-arc"/>
      <path d="M ${x1} ${nodeY} l -4 -7 l 8 0 z" class="dep-arrow"/>
      <text x="${mx}" y="${apex - 3}" class="dep-label">${escapeHtml(d.label)}</text>`;
  }
  const wtxt = words.map((w, i) =>
    `<text x="${cx(i)}" y="${wordY}" class="dep-word">${escapeHtml(w.text)}</text>`).join("");
  return `<div class="dep-scroll"><svg viewBox="0 0 ${width} ${height}" width="${width}" height="${height}" class="dep-svg">${arcs}${wtxt}</svg></div>`;
}

function srlRoleClass(role) {
  if (!role) return "argm";
  if (role === "ARG0") return "arg0";
  if (role === "ARG1") return "arg1";
  if (role.startsWith("ARGM")) return "argm";
  return "arg2";
}

function renderSRL(words, srl) {
  if (!srl || !srl.length) return `<div class="md-empty">술어-논항 구조가 식별되지 않았어요.</div>`;
  return srl.map((fr) => {
    const pred = fr.predicate || {};
    const args = (fr.args || []).map((a) =>
      `<span class="srl-arg ${srlRoleClass(a.role)}"><span class="srl-role">${escapeHtml(a.role)}</span>${escapeHtml(a.text)}</span>`
    ).join("");
    return `<div class="srl-frame">
      <span class="srl-pred">술어 · ${escapeHtml(pred.verb || "")}</span>
      <span class="srl-args">${args || "<span class='md-muted'>논항 없음</span>"}</span>
    </div>`;
  }).join("");
}

// ── Scoring call ──────────────────────────────────────────────────
const ENSEMBLE_N = 30;

let livePreview = "";
let sampleCompleted = 0;

async function submitEssay() {
  const essay = $("essay").value.trim();
  if (!currentTopic) { toast("논제를 먼저 선택해 주세요."); setStep(1); return; }
  if (essay.length < 20) { toast("에세이를 조금 더 작성해 주세요 (최소 20자)."); return; }
  currentEssay = essay;
  // reset previous verify/result-augmentation state for the new essay
  $("audit-section").hidden = true;
  $("audit-section").removeAttribute("open");
  $("ref-section").hidden = true;
  $("ref-section").removeAttribute("open");
  $("linguistic-section").hidden = true;
  $("linguistic-section").removeAttribute("open");
  $("hero-adjust").hidden = true;
  $("total-pct").hidden = true;
  $("criteria-badge").hidden = true;
  $("overall-report").hidden = true;
  $("report-body").innerHTML = "";
  $("ling-body").innerHTML = "";
  currentReport = null;
  linguisticLoaded = false;

  const body = {
    topic: currentTopic,
    essay,
    keywords: currentKeywords || null,
    n_samples: ENSEMBLE_N,
    temperature: 0.7,
    top_p: 0.9,
    repetition_penalty: 1.1,
    max_new_tokens: 1024,
    top_feedback_k: 3,
  };

  $("btn-submit").disabled = true;
  $("status-card").hidden = false;
  $("status-sub").textContent = `AI가 ${ENSEMBLE_N}개 샘플 병렬 생성 중이에요… (0/${ENSEMBLE_N})`;
  livePreview = "";
  sampleCompleted = 0;

  try {
    if ($("stream-toggle").checked) {
      await streamEnsemble(body);
    } else {
      const r = await fetch("/api/score/ensemble", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      });
      if (!r.ok) throw new Error(await r.text());
      const d = await r.json();
      applyResult(d);
    }
    // 채점만 하고 바로 넘어가지 않는다 — AI 검증(도구 호출)까지 모두 끝낸 뒤 결과로 이동
    $("status-card").hidden = true;
    await runVerify();   // builds the agent-final report; on failure, the LoRA result stands in
    setStep(3);
  } catch (e) {
    toast(`오류: ${e.message || e}`);
  } finally {
    $("btn-submit").disabled = false;
    $("status-card").hidden = true;
    $("processing").hidden = true;
  }
}

async function streamEnsemble(body) {
  const r = await fetch("/api/score/ensemble/stream", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!r.ok || !r.body) throw new Error(`요청 실패 (${r.status})`);

  const reader = r.body.getReader();
  const decoder = new TextDecoder("utf-8");
  let carry = "";
  let doneData = null;
  let total = body.n_samples;

  while (true) {
    const { value, done } = await reader.read();
    if (done) break;
    carry += decoder.decode(value, { stream: true });
    let idx;
    while ((idx = carry.indexOf("\n\n")) >= 0) {
      const frame = carry.slice(0, idx);
      carry = carry.slice(idx + 2);
      const ev = parseSSE(frame);
      if (!ev) continue;
      if (ev.event === "start") {
        total = ev.data.n_samples || total;
        $("status-sub").textContent = `병렬 생성 시작 · 0/${total} 샘플`;
      } else if (ev.event === "token") {
        livePreview += ev.data.text || "";
        const trimmed = livePreview.length > 280
          ? "…" + livePreview.slice(-280)
          : livePreview;
        $("status-sub").textContent = `대표 샘플 미리보기: ${trimmed.replace(/\n/g, " ⏎ ")}`;
      } else if (ev.event === "sample") {
        sampleCompleted += 1;
        const valid = (ev.data.scores || []).filter(v => v != null).length;
        $("status-sub").textContent = `샘플 ${sampleCompleted}/${total} 완료 (${valid}/8 파싱됨)`;
      } else if (ev.event === "score_probs") {
        $("status-sub").textContent = `확률 분포 수신 완료 · 집계 중…`;
      } else if (ev.event === "done") {
        doneData = ev.data;
      } else if (ev.event === "error") {
        throw new Error(ev.data.message || "모델 오류");
      }
    }
  }
  if (!doneData) throw new Error("응답이 완결되지 않았어요.");
  applyResult(doneData);
}

function parseSSE(frame) {
  let event = "message", data = "";
  for (const line of frame.split("\n")) {
    if (line.startsWith("event:")) event = line.slice(6).trim();
    else if (line.startsWith("data:")) data += line.slice(5).trim();
  }
  if (!data) return null;
  try { return { event, data: JSON.parse(data) }; }
  catch { return { event, data: { text: data } }; }
}

// ── Health check ──────────────────────────────────────────────────
async function pollHealth() {
  try {
    const r = await fetch("/api/health");
    const d = await r.json();
    if (!d.ready) setTimeout(pollHealth, 2500);
  } catch { setTimeout(pollHealth, 3000); }
}

// ── Boot ──────────────────────────────────────────────────────────
window.addEventListener("DOMContentLoaded", async () => {
  // 라우팅: 로고 → 홈(논제 선택), 채점 기준 → #/guide, 뒤로가기 지원
  window.addEventListener("hashchange", renderRoute);
  document.querySelector(".brand").addEventListener("click", () => {
    // hash가 이미 "#/"면 hashchange가 안 나므로 직접 홈 1단계로 이동한다.
    if (currentRoute() === "home") setStep(1);
    else wizardStep = 1;   // guide → home 전환 시 renderRoute가 1단계를 띄운다
  });
  // 진행 스텝 클릭으로 이전 단계 되돌아가기
  document.querySelectorAll(".step-pill").forEach((el) => {
    el.addEventListener("click", () => {
      const s = parseInt(el.dataset.step, 10);
      if (s >= wizardStep) return;               // 앞 단계로만 이동
      if (s === 2 && !currentTopic) return;      // 논제 없이 작성 단계 금지
      setStep(s);
    });
  });
  renderRoute();

  try {
    await Promise.all([loadTopics(), loadRubric()]);
  } catch (e) {
    toast(`서버에서 논제·기준을 불러오지 못했어요. 새로고침해 주세요. (${e.message || e})`);
  }
  pollHealth();

  $("essay").addEventListener("input", countEssay);
  countEssay();

  $("topic-search").addEventListener("input", (e) => {
    topicQuery = e.target.value;
    renderTopicGrid();
  });

  $("btn-custom-next").addEventListener("click", () => {
    const t = $("custom-topic").value.trim();
    const k = $("custom-keywords").value.trim();
    if (!t) { toast("논제를 입력해 주세요."); return; }
    pickTopic(t, k);
  });

  $("btn-back-topic").addEventListener("click", () => setStep(1));
  $("btn-submit").addEventListener("click", submitEssay);

  $("btn-rewrite").addEventListener("click", () => setStep(2));
  $("btn-new-topic").addEventListener("click", () => {
    $("essay").value = "";
    countEssay();
    setStep(1);
  });

  // tool detail modal
  $("modal-close").addEventListener("click", closeToolModal);
  $("tool-modal").addEventListener("click", (e) => {
    if (e.target.id === "tool-modal") closeToolModal();
  });
  document.addEventListener("keydown", (e) => {
    if (e.key === "Escape" && !$("tool-modal").hidden) closeToolModal();
  });

  // 언어 분석: 펼칠 때 온디맨드 로드
  $("linguistic-section").addEventListener("toggle", (e) => {
    if (e.target.open) loadLinguistic();
  });
});
