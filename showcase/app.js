const docSeeds = [
  {
    hash: "482e9c487c608f1bbeaceef35bc3c0933e8b35443cfff447e4279d590468364a",
    fallbackSource: "Document/irs_i1040gi_2025.pdf",
    tagline: "IRS filing guidance",
  },
  {
    hash: "728293cfbfeb672979115ab969cf6ec85cd62977f643eaee397334c6a9534ebd",
    fallbackSource: "Document/fed_monetary_policy_report_2025_02.pdf",
    tagline: "Federal Reserve report",
  },
  {
    hash: "7576edb531d9848825814ee88e28b1795d3a84b435b4b797d3670eafdc4a89f1",
    fallbackSource: "Document/nist_ai_rmf_1_0.pdf",
    tagline: "AI governance framework",
  },
  {
    hash: "8eeb4887a4dc57a23049da7dd2ed556833cf98e214b240468d987873164ff688",
    fallbackSource: "Document/nasa_systems_engineering_handbook_rev2.pdf",
    tagline: "Aerospace systems engineering",
  },
  {
    hash: "faedd8a0625b4f558ec5303a9eea047edf1a3a3c0c2efde577b3e63d1d1d4d21",
    fallbackSource: "Document/osha_field_operations_manual_2019.pdf",
    tagline: "OSHA field operations",
  },
];

const state = {
  data: null,
  documents: [],
  filter: "all",
};

const recommendedQueries = [
  "What is the purpose of the NIST AI Risk Management Framework 1.0?",
  "Name the four core AI RMF functions.",
  "In the Federal Reserve report, how did PCE inflation change?",
  "From IRS 1040 instructions, when should taxpayers use the Tax Table versus the Tax Computation Worksheet?",
  "Give two concrete claims from OSHA manual and cite each source.",
];

const els = {
  heroMetrics: document.getElementById("heroMetrics"),
  summaryGrid: document.getElementById("summaryGrid"),
  barList: document.getElementById("barList"),
  docGrid: document.getElementById("docGrid"),
  questionList: document.getElementById("questionList"),
  filterRow: document.getElementById("filterRow"),
  detailDialog: document.getElementById("detailDialog"),
  dialogBody: document.getElementById("dialogBody"),
  exampleSelect: document.getElementById("exampleSelect"),
  demoInput: document.getElementById("demoInput"),
  demoAnswer: document.getElementById("demoAnswer"),
  demoMeta: document.getElementById("demoMeta"),
  runDemo: document.getElementById("runDemo"),
  surpriseMe: document.getElementById("surpriseMe"),
  liveQuestion: document.getElementById("liveQuestion"),
  liveTopK: document.getElementById("liveTopK"),
  liveDbDir: document.getElementById("liveDbDir"),
  runLiveQuery: document.getElementById("runLiveQuery"),
  useSampleQuery: document.getElementById("useSampleQuery"),
  liveAnswer: document.getElementById("liveAnswer"),
  liveMeta: document.getElementById("liveMeta"),
  liveSources: document.getElementById("liveSources"),
};

async function loadJson(path) {
  const res = await fetch(path);
  if (!res.ok) {
    throw new Error(`Failed to load ${path}`);
  }
  return res.json();
}

async function loadEvaluation() {
  return loadJson("../evaluation/golden_eval_vlm.json");
}

async function loadDocuments() {
  const docs = await Promise.all(
    docSeeds.map(async (seed) => {
      const metadataPath = `../outputs/${seed.hash}/metadata.json`;
      const fallback = {
        ...seed,
        source_path: seed.fallbackSource,
        pages: null,
        enable_vlm: true,
        ocr_engine: "unknown",
      };

      try {
        const metadata = await loadJson(metadataPath);
        return { ...seed, ...metadata };
      } catch {
        return fallback;
      }
    })
  );

  return docs;
}

function pct(v, digits = 1) {
  if (typeof v !== "number") return "-";
  return `${(v * 100).toFixed(digits)}%`;
}

function truncate(text, maxLen = 210) {
  if (!text) return "";
  if (text.length <= maxLen) return text;
  return `${text.slice(0, maxLen).trim()}...`;
}

function unique(arr) {
  return [...new Set((arr || []).filter(Boolean))];
}

function esc(s) {
  return String(s ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#39;");
}

function renderHero(summary, docs) {
  const totalPages = docs.reduce((sum, d) => sum + (Number(d.pages) || 0), 0);
  const cards = [
    { label: "Source PDFs", value: docs.length },
    { label: "Total Pages", value: totalPages },
    {
      label: "Deterministic Pass",
      value: `${summary.deterministic_pass_count}/${summary.total_questions}`,
    },
    { label: "Pass Rate", value: pct(summary.deterministic_pass_rate, 0) },
    { label: "LLM Judge Mean", value: pct(summary.llm_judge_mean_overall, 1) },
  ];

  els.heroMetrics.innerHTML = cards
    .map(
      (card) => `
        <article class="metric-pill">
          <strong>${esc(card.value)}</strong>
          <span>${esc(card.label)}</span>
        </article>
      `
    )
    .join("");
}

function renderSummary(summary) {
  const cards = [
    { label: "Expected Doc Hit", value: pct(summary.expected_doc_hit_rate, 1) },
    { label: "Expected Type Hit", value: pct(summary.expected_type_hit_rate, 1) },
    { label: "Mean Term Recall", value: pct(summary.mean_term_recall, 1) },
    { label: "Citation Valid Mean", value: pct(summary.mean_citation_valid_rate, 1) },
    {
      label: "Abstentions",
      value: `${summary.abstain_count}/${summary.total_questions}`,
    },
  ];

  els.summaryGrid.innerHTML = cards
    .map(
      (card) => `
        <article class="summary-card">
          <p class="label">${esc(card.label)}</p>
          <p class="value">${esc(card.value)}</p>
        </article>
      `
    )
    .join("");

  const bars = [
    { label: "Deterministic Pass Rate", value: summary.deterministic_pass_rate },
    { label: "Expected Doc Hit Rate", value: summary.expected_doc_hit_rate },
    { label: "Expected Type Hit Rate", value: summary.expected_type_hit_rate },
    { label: "Mean Term Recall", value: summary.mean_term_recall },
    { label: "Citation Valid Mean", value: summary.mean_citation_valid_rate },
    { label: "LLM Judge Mean", value: summary.llm_judge_mean_overall },
  ];

  els.barList.innerHTML = bars
    .map(
      (b) => `
        <article class="bar-item">
          <div class="bar-item-head">
            <span>${esc(b.label)}</span>
            <strong>${esc(pct(b.value, 1))}</strong>
          </div>
          <div class="bar-track"><div class="bar-fill" data-width="${Math.max(0, Math.min(100, b.value * 100))}"></div></div>
        </article>
      `
    )
    .join("");

  requestAnimationFrame(() => {
    document.querySelectorAll(".bar-fill").forEach((el) => {
      el.style.width = `${el.dataset.width}%`;
    });
  });
}

function renderDocuments(docs) {
  const cards = docs
    .map((doc) => {
      const filename = doc.source_path?.split("/").pop() || "unknown.pdf";
      const pagePreview = `../outputs/${doc.hash}/pages/page_1.png`;
      const regionPreview = `../outputs/${doc.hash}/regions/page_1_1.png`;
      const docJson = `../outputs/${doc.hash}/document.json`;
      const docMd = `../outputs/${doc.hash}/document.md`;

      return `
        <article class="doc-card">
          <div class="doc-images">
            <img src="${esc(pagePreview)}" alt="${esc(filename)} first page" loading="lazy" onerror="this.style.display='none'" />
            <img src="${esc(regionPreview)}" alt="${esc(filename)} region sample" loading="lazy" onerror="this.style.display='none'" />
          </div>
          <div class="doc-body">
            <h3 class="doc-title">${esc(filename)}</h3>
            <p class="doc-subtitle">${esc(doc.tagline || "Processed source document")}</p>
            <div class="doc-stats">
              <span class="stat-chip">pages:${esc(doc.pages ?? "?")}</span>
              <span class="stat-chip">ocr:${esc(doc.ocr_engine || "-")}</span>
              <span class="stat-chip">vlm:${doc.enable_vlm === false ? "off" : "on"}</span>
            </div>
            <div class="hero-actions" style="margin-top:0.7rem; margin-bottom:0">
              <a class="btn btn-ghost" href="${esc(docJson)}">document.json</a>
              <a class="btn btn-ghost" href="${esc(docMd)}">document.md</a>
            </div>
          </div>
        </article>
      `;
    })
    .join("");

  els.docGrid.innerHTML = cards;
}

function resultMatchesFilter(result, filter) {
  if (filter === "all") return true;
  if (filter === "pass") return result.deterministic_pass;
  if (filter === "fail") return !result.deterministic_pass;
  return result.category === filter;
}

function renderQuestions(results) {
  const filtered = results.filter((r) => resultMatchesFilter(r, state.filter));

  els.questionList.innerHTML = filtered
    .map(
      (r) => `
        <article class="q-card" data-id="${esc(r.id)}">
          <div class="q-top">
            <div>
              <p class="q-id">${esc(r.id)} | ${esc(r.category)}</p>
              <p class="q-text">${esc(r.question)}</p>
            </div>
            <span class="status ${r.deterministic_pass ? "pass" : "fail"}">${r.deterministic_pass ? "PASS" : "FAIL"}</span>
          </div>
          <p class="q-text">${esc(truncate(r.answer, 220))}</p>
          <div class="q-meta">
            <span class="q-badge">term:${esc(pct(r.term_recall, 0))}</span>
            <span class="q-badge">citation:${esc(pct(r.citation_stats?.citation_valid_rate ?? 0, 0))}</span>
            <span class="q-badge">abstain:${r.abstained ? "yes" : "no"}</span>
          </div>
        </article>
      `
    )
    .join("");

  const idToResult = new Map(filtered.map((r) => [r.id, r]));
  els.questionList.querySelectorAll(".q-card").forEach((card) => {
    card.addEventListener("click", () => {
      const q = idToResult.get(card.dataset.id);
      if (q) openDetail(q);
    });
  });
}

function openDetail(q) {
  const docs = unique(q.retrieved_docs).join(", ");
  const types = unique(q.retrieved_types).join(", ");
  const judge = q.llm_judge || {};

  els.dialogBody.innerHTML = `
    <p class="q-id">${esc(q.id)} | ${esc(q.category)}</p>
    <h3>${esc(q.question)}</h3>
    <p>${esc(q.answer)}</p>
    <div class="q-meta">
      <span class="q-badge">pass:${q.deterministic_pass ? "true" : "false"}</span>
      <span class="q-badge">term_recall:${esc((q.term_recall ?? 0).toFixed(2))}</span>
      <span class="q-badge">citation_valid:${esc((q.citation_stats?.citation_valid_rate ?? 0).toFixed(2))}</span>
      <span class="q-badge">abstained:${q.abstained ? "true" : "false"}</span>
    </div>
    <p><strong>Retrieved docs:</strong> ${esc(docs || "-")}</p>
    <p><strong>Retrieved types:</strong> ${esc(types || "-")}</p>
    <p><strong>LLM judge:</strong> overall=${esc((judge.overall ?? 0).toFixed(3))}, correctness=${esc(String(judge.correctness ?? "-"))}, citation_support=${esc(String(judge.citation_support ?? "-"))}</p>
    <p><strong>Judge notes:</strong> ${esc(judge.notes || "-")}</p>
  `;

  els.detailDialog.showModal();
}

function setupFilters(results) {
  els.filterRow.addEventListener("click", (e) => {
    const btn = e.target.closest("button[data-filter]");
    if (!btn) return;

    state.filter = btn.dataset.filter;
    els.filterRow.querySelectorAll("button").forEach((b) => b.classList.remove("active"));
    btn.classList.add("active");
    renderQuestions(results);
  });
}

function setupDemo(results) {
  els.exampleSelect.innerHTML = results
    .map((r, i) => `<option value="${i}">${esc(r.id)} | ${esc(r.category)} | ${esc(truncate(r.question, 70))}</option>`)
    .join("");

  const setDemoFromResult = (r) => {
    els.demoInput.value = r.question;
    els.demoAnswer.textContent = r.answer;

    const metaBits = [
      `deterministic_pass:${r.deterministic_pass}`,
      `term_recall:${(r.term_recall ?? 0).toFixed(2)}`,
      `citation_valid:${(r.citation_stats?.citation_valid_rate ?? 0).toFixed(2)}`,
      `retrieved:${unique(r.retrieved_docs).slice(0, 3).join(", ") || "-"}`,
    ];

    els.demoMeta.innerHTML = metaBits.map((m) => `<span class="q-badge">${esc(m)}</span>`).join("");
  };

  const findBestMatch = () => {
    const query = els.demoInput.value.trim().toLowerCase();
    if (!query) return results[0];

    const exact = results.find((r) => r.question.toLowerCase() === query);
    if (exact) return exact;

    const keywords = query.split(/\s+/).filter((w) => w.length > 3);
    let best = results[0];
    let bestScore = -1;

    results.forEach((r) => {
      const text = `${r.question} ${r.answer}`.toLowerCase();
      const score = keywords.reduce((s, kw) => (text.includes(kw) ? s + 1 : s), 0);
      if (score > bestScore) {
        bestScore = score;
        best = r;
      }
    });

    return best;
  };

  els.exampleSelect.addEventListener("change", () => {
    const idx = Number(els.exampleSelect.value);
    setDemoFromResult(results[idx] || results[0]);
  });

  els.runDemo.addEventListener("click", () => {
    setDemoFromResult(findBestMatch());
  });

  els.surpriseMe.addEventListener("click", () => {
    const idx = Math.floor(Math.random() * results.length);
    els.exampleSelect.value = String(idx);
    setDemoFromResult(results[idx]);
  });

  setDemoFromResult(results[0]);
}

function setupRevealAnimation() {
  const sections = document.querySelectorAll(".reveal");
  const observer = new IntersectionObserver(
    (entries) => {
      entries.forEach((entry) => {
        if (entry.isIntersecting) {
          entry.target.classList.add("visible");
          observer.unobserve(entry.target);
        }
      });
    },
    { threshold: 0.12 }
  );

  sections.forEach((el) => observer.observe(el));
}

function showLoadError(message) {
  const escaped = esc(message);
  els.heroMetrics.innerHTML = `<article class="metric-pill"><strong>Data load error</strong><span>${escaped}</span></article>`;
  els.summaryGrid.innerHTML = `<article class="summary-card"><p class="label">Could not load showcase data</p><p class="value" style="font-size:1rem">${escaped}</p></article>`;
}

function renderLiveSources(sources) {
  if (!Array.isArray(sources) || sources.length === 0) {
    els.liveSources.innerHTML = "";
    return;
  }

  els.liveSources.innerHTML = sources
    .map(
      (s) => `
        <article class="source-card">
          <div class="source-head">
            <strong>${esc(s.source_name || "unknown")}</strong>
            <span>p${esc(s.page ?? "?")} | ${esc(s.chunk_type || "?")} | sim=${esc(
              typeof s.similarity === "number" ? s.similarity.toFixed(3) : "-"
            )}</span>
          </div>
          <p class="source-preview">${esc(s.preview || "")}</p>
        </article>
      `
    )
    .join("");
}

function setLiveBusy(isBusy) {
  els.runLiveQuery.disabled = isBusy;
  els.runLiveQuery.textContent = isBusy ? "Running..." : "Run Live Query";
}

function setupLiveQuery() {
  const setRecommended = () => {
    const q = recommendedQueries[Math.floor(Math.random() * recommendedQueries.length)];
    els.liveQuestion.value = q;
  };

  els.useSampleQuery.addEventListener("click", () => {
    setRecommended();
  });

  els.runLiveQuery.addEventListener("click", async () => {
    const question = els.liveQuestion.value.trim();
    const topK = Number(els.liveTopK.value || 6);
    const dbDir = (els.liveDbDir.value || "chroma_db").trim();

    if (!question) {
      els.liveAnswer.textContent = "Please enter a question.";
      return;
    }

    setLiveBusy(true);
    els.liveAnswer.textContent = "Running live query...";
    els.liveMeta.innerHTML = "";
    els.liveSources.innerHTML = "";

    try {
      const res = await fetch("/api/ask", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          question: question,
          top_k: topK,
          db_dir: dbDir,
        }),
      });

      const data = await res.json();
      if (!res.ok) {
        throw new Error(data.error || data.detail || `HTTP ${res.status}`);
      }

      els.liveAnswer.textContent = data.answer || "(no answer returned)";
      const sourceCount = Array.isArray(data.sources) ? data.sources.length : 0;
      els.liveMeta.innerHTML = [
        `top_k:${data.top_k}`,
        `sources:${sourceCount}`,
        `db:${data.db_dir || dbDir}`,
      ]
        .map((m) => `<span class="q-badge">${esc(m)}</span>`)
        .join("");
      if (sourceCount === 0) {
        els.liveMeta.innerHTML += `<span class="q-badge">hint:no indexed chunks in this db</span>`;
      }
      renderLiveSources(data.sources || []);
    } catch (err) {
      const msg = err instanceof Error ? err.message : "Unknown live query error";
      els.liveAnswer.textContent = `Live query failed: ${msg}`;
      els.liveMeta.innerHTML = `<span class="q-badge">hint: run "uv run python showcase/server.py"</span>`;
    } finally {
      setLiveBusy(false);
    }
  });

  setRecommended();
}

async function init() {
  setupRevealAnimation();

  try {
    const [evaluation, documents] = await Promise.all([loadEvaluation(), loadDocuments()]);
    state.data = evaluation;
    state.documents = documents;

    renderHero(evaluation.summary, documents);
    renderSummary(evaluation.summary);
    renderDocuments(documents);
    renderQuestions(evaluation.results);
    setupFilters(evaluation.results);
    setupDemo(evaluation.results);
    setupLiveQuery();
  } catch (err) {
    const msg =
      err instanceof Error
        ? `${err.message}. Serve the repo root with: python3 -m http.server`
        : "Unknown error while loading data.";
    showLoadError(msg);
  }
}

init();
