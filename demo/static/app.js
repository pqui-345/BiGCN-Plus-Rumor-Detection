(function () {
  "use strict";

  const STATE = {
    meta: null,
    benchmarks: null,
    dataset: null,
    tid: null,
    eventData: null,
    earlyData: null,
    whatIfData: null,
    playTimer: null,
  };

  const CLS_COLORS = ["#16a34a", "#dc2626", "#2563eb", "#f59e0b"];
  const CLS_NAMES = ["NR", "FR", "TR", "UR"];
  const DEADLINE_SHORT = ["0p", "10p", "30p", "1g", "2g", "4g", "12g", "Full"];

  function fmtNum(v, d = 4) {
    if (v === null || v === undefined || Number.isNaN(Number(v))) return "—";
    return Number(v).toFixed(d).replace(".", ",");
  }

  async function apiGet(url) {
    const res = await fetch(url);
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    return res.json();
  }

  async function apiPost(url, body) {
    const res = await fetch(url, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    return res.json();
  }

  // Interactive Tree Canvas Renderer (Pan + Zoom + Hover Tooltip, zero external libs)
  class TreeCanvas {
    constructor(canvasId, tooltipId) {
      this.canvas = document.getElementById(canvasId);
      this.tooltip = document.getElementById(tooltipId);
      this.ctx = this.canvas.getContext("2d");
      this.tree = null;
      this.layoutNodes = [];
      this.scale = 1.0;
      this.tx = 0;
      this.ty = 0;
      this.dragging = false;
      this.lastX = 0;
      this.lastY = 0;
      this._bindEvents();
    }

    _bindEvents() {
      this.canvas.addEventListener("mousedown", (e) => {
        this.dragging = true;
        this.lastX = e.offsetX;
        this.lastY = e.offsetY;
      });
      window.addEventListener("mouseup", () => {
        this.dragging = false;
      });
      this.canvas.addEventListener("mousemove", (e) => {
        if (this.dragging) {
          this.tx += e.offsetX - this.lastX;
          this.ty += e.offsetY - this.lastY;
          this.lastX = e.offsetX;
          this.lastY = e.offsetY;
          this.draw();
          return;
        }
        this._handleHover(e.offsetX, e.offsetY);
      });
      this.canvas.addEventListener("mouseleave", () => {
        this.tooltip.style.display = "none";
      });
      this.canvas.addEventListener(
        "wheel",
        (e) => {
          e.preventDefault();
          const factor = e.deltaY < 0 ? 1.12 : 0.89;
          const newScale = Math.min(4.0, Math.max(0.45, this.scale * factor));
          const ratio = newScale / this.scale;
          this.tx = e.offsetX - (e.offsetX - this.tx) * ratio;
          this.ty = e.offsetY - (e.offsetY - this.ty) * ratio;
          this.scale = newScale;
          this.draw();
        },
        { passive: false }
      );
    }

    resetView() {
      this.scale = 1.0;
      this.tx = 0;
      this.ty = 0;
      this.draw();
    }

    setTree(treeViz) {
      this.tree = treeViz;
      this.scale = 1.0;
      this.tx = 0;
      this.ty = 0;
      this._computeLayout();
      this.draw();
    }

    _computeLayout() {
      if (!this.tree || !this.tree.nodes.length) {
        this.layoutNodes = [];
        return;
      }
      const w = this.canvas.width;
      const h = this.canvas.height;
      const nodes = this.tree.nodes;
      const edges = this.tree.edges;

      // Radial hierarchical tree layout centered at Root (id = 0)
      const byDepth = new Map();
      let maxD = 0;
      nodes.forEach((nd) => {
        maxD = Math.max(maxD, nd.depth);
        if (!byDepth.has(nd.depth)) byDepth.set(nd.depth, []);
        byDepth.get(nd.depth).push(nd);
      });

      const cx = w / 2;
      const cy = h / 2;
      const maxR = Math.min(w, h) * 0.43;
      const posMap = new Map();

      if (maxD === 0) {
        posMap.set(0, { x: cx, y: cy, nd: nodes[0] });
      } else {
        // Assign angular span based on order for clean non-cluttered rings
        const childMap = new Map();
        edges.forEach((e) => {
          if (!childMap.has(e.from)) childMap.set(e.from, []);
          childMap.get(e.from).push(e.to);
        });
        const nodeMap = new Map(nodes.map((n) => [n.id, n]));

        // BFS deterministic angular ordering
        const order = [];
        const placeSub = (uid, a0, a1) => {
          const nd = nodeMap.get(uid);
          if (!nd) return;
          const mid = (a0 + a1) / 2;
          const r = nd.depth === 0 ? 0 : (nd.depth / Math.max(maxD, 1)) * maxR;
          posMap.set(uid, {
            x: cx + r * Math.cos(mid),
            y: cy + r * Math.sin(mid),
            nd,
          });
          const ch = childMap.get(uid) || [];
          if (!ch.length) return;
          const span = (a1 - a0) / ch.length;
          ch.forEach((cid, i) => {
            placeSub(cid, a0 + i * span, a0 + (i + 1) * span);
          });
        };
        placeSub(0, 0, Math.PI * 2);

        // Ensure any disconnected viz nodes still appear cleanly
        nodes.forEach((nd, i) => {
          if (!posMap.has(nd.id)) {
            const ang = (i / nodes.length) * Math.PI * 2;
            const r = (nd.depth / Math.max(maxD, 1)) * maxR;
            posMap.set(nd.id, { x: cx + r * Math.cos(ang), y: cy + r * Math.sin(ang), nd });
          }
        });
      }

      this.layoutNodes = Array.from(posMap.values());
      this.posMap = posMap;
    }

    _nodeColor(nd) {
      if (nd.is_root) return "#f59e0b";
      if (nd.delay <= 15) return "#16a34a";
      if (nd.delay <= 120) return "#2563eb";
      return "#dc2626";
    }

    draw() {
      const ctx = this.ctx;
      const w = this.canvas.width;
      const h = this.canvas.height;
      ctx.clearRect(0, 0, w, h);

      if (!this.tree || !this.layoutNodes.length) return;

      ctx.save();
      ctx.translate(this.tx, this.ty);
      ctx.scale(this.scale, this.scale);

      // Draw edges
      ctx.strokeStyle = "rgba(100, 116, 139, 0.35)";
      ctx.lineWidth = Math.max(0.8, 1.2 / this.scale);
      this.tree.edges.forEach((e) => {
        const p1 = this.posMap.get(e.from);
        const p2 = this.posMap.get(e.to);
        if (!p1 || !p2) return;
        ctx.beginPath();
        ctx.moveTo(p1.x, p1.y);
        ctx.lineTo(p2.x, p2.y);
        ctx.stroke();
      });

      // Draw nodes
      this.layoutNodes.forEach(({ x, y, nd }) => {
        const radius = nd.is_root ? 11 : 4.2 + nd.centrality * 5.0;
        ctx.beginPath();
        ctx.arc(x, y, radius, 0, Math.PI * 2);
        ctx.fillStyle = this._nodeColor(nd);
        ctx.fill();
        ctx.lineWidth = nd.is_root ? 2.6 : 1.0;
        ctx.strokeStyle = nd.is_root ? "#92400e" : "#ffffff";
        ctx.stroke();
      });
      ctx.restore();

      if (this.tree.truncated_for_viz) {
        ctx.fillStyle = "rgba(15,23,42,0.75)";
        ctx.font = "11px Segoe UI";
        ctx.fillText(`Hiển thị ${this.tree.n_shown}/${this.tree.n_full} nút đầu theo BFS để giữ mượt`, 10, h - 10);
      }
    }

    _handleHover(mx, my) {
      const worldX = (mx - this.tx) / this.scale;
      const worldY = (my - this.ty) / this.scale;
      let hit = null;
      let minDist = 14 / this.scale;
      for (const item of this.layoutNodes) {
        const dx = item.x - worldX;
        const dy = item.y - worldY;
        const dist = Math.hypot(dx, dy);
        if (dist < minDist) {
          minDist = dist;
          hit = item;
        }
      }
      if (!hit) {
        this.tooltip.style.display = "none";
        return;
      }
      const nd = hit.nd;
      this.tooltip.innerHTML = `
        <strong>${nd.is_root ? "Nút Gốc (Source)" : "Phản hồi #" + nd.id}</strong><br/>
        Độ sâu: ${nd.depth} &bull; Độ trễ: ${nd.delay} phút<br/>
        PageRank: ${nd.centrality} &bull; Nhánh con: ${nd.children_count}
      `;
      this.tooltip.style.left = `${Math.min(this.canvas.clientWidth - 180, mx + 12)}px`;
      this.tooltip.style.top = `${Math.max(8, my - 12)}px`;
      this.tooltip.style.display = "block";
    }
  }

  let canvas1, canvas2, canvas3;

  function renderProbBar(probs) {
    const segs = probs
      .map((p, idx) => `<div class="seg-${idx}" style="width:${(p * 100).toFixed(1)}%" title="${CLS_NAMES[idx]}: ${(p * 100).toFixed(1)}%"></div>`)
      .join("");
    return `<div class="mini-probs"><div class="mini-bar">${segs}</div><span>${(Math.max(...probs) * 100).toFixed(1)}%</span></div>`;
  }

  function renderShowdownTable(tableId, modelsObj, yTrue = null) {
    const tbody = document.querySelector(`#${tableId} tbody`);
    const order = ["BiGCN-Plus-Clean", "RAGCL", "BiGCN-Base", "MLP-BoW", "TF-IDF-Ridge"];
    tbody.innerHTML = order
      .filter((m) => modelsObj[m])
      .map((m) => {
        const r = modelsObj[m];
        const conf = Math.max(...r.probs) * 100;
        const isPlus = m === "BiGCN-Plus-Clean";
        let statusTd = "";
        if (yTrue !== null) {
          const ok = r.pred === yTrue;
          statusTd = `<td><span class="${ok ? "pill-ok" : "pill-err"}">${ok ? "✓ Đúng" : "✗ Sai"}</span></td>`;
        }
        return `
          <tr class="${isPlus ? "row-highlight" : ""}">
            <td><code>${m}</code></td>
            <td><span class="lbl-chip lbl-${r.label}">${r.label}</span></td>
            <td class="right">${conf.toFixed(2).replace(".", ",")}%</td>
            <td>${renderProbBar(r.probs)}</td>
            ${statusTd}
          </tr>
        `;
      })
      .join("");
  }

  function renderXaiSection(xai, stripId, barsId, tokensId, statsTableId = null, autoLooId = null) {
    const fw = xai.fusion_weights;
    document.getElementById(stripId).innerHTML = [
      { title: "Nhánh 1: Exact BiGCN", w: fw[0] },
      { title: "Nhánh 2: Dual-SAGE", w: fw[1] },
      { title: "Nhánh 3: Auto-LOO", w: fw[2] },
    ]
      .map(
        (b) => `
        <div class="fusion-box">
          <div class="f-title">${b.title}</div>
          <div class="f-val">w = ${(b.w * 100).toFixed(1).replace(".", ",")}%</div>
        </div>`
      )
      .join("");

    if (autoLooId && xai.auto_loo_params) {
      document.getElementById(autoLooId).textContent =
        `Auto-LOO: g_weight = ${xai.auto_loo_params.g_weight}, lambda = ${xai.auto_loo_params.ridge_lam}`;
    }

    const branches = [
      { name: "Nhánh 1 (BiGCN TD+BU)", p: xai.branch1_probs },
      { name: "Nhánh 2 (Dual-View SAGE)", p: xai.branch2_probs },
      { name: "Nhánh 3 (TF-IDF + Graph)", p: xai.branch3_probs },
    ];
    document.getElementById(barsId).innerHTML = branches
      .map((b) => {
        const bestIdx = b.p.indexOf(Math.max(...b.p));
        const pct = (b.p[bestIdx] * 100).toFixed(1);
        return `
          <div class="b-row">
            <span>${b.name}</span>
            <div class="b-track"><div class="b-fill" style="width:${pct}%;background:${CLS_COLORS[bestIdx]}"></div></div>
            <span class="lbl-chip lbl-${CLS_NAMES[bestIdx]}">${CLS_NAMES[bestIdx]} ${pct}%</span>
          </div>`;
      })
      .join("");

    document.getElementById(tokensId).innerHTML = (xai.top_tokens || [])
      .map(
        (t) => `
        <li>
          <span><code>${t.token}</code><span class="tok-kind">(${t.kind})</span></span>
          <span class="tok-score">${t.score >= 0 ? "+" : ""}${t.score.toFixed(4).replace(".", ",")}</span>
        </li>`
      )
      .join("");

    if (statsTableId) {
      document.querySelector(`#${statsTableId} tbody`).innerHTML = xai.graph_stats
        .map((g) => `<tr><td title="${g.key}">${g.name}</td><td class="right">${g.value.toFixed(4).replace(".", ",")}</td></tr>`)
        .join("");
    }
  }

  function populateEventSelect() {
    const ds = STATE.dataset;
    const filter = document.getElementById("filter-select").value;
    const all = STATE.meta.events[ds] || [];
    let filtered = all;
    if (filter === "highlight") filtered = all.filter((e) => e.win_tag === "highlight");
    else if (filter === "sparse") filtered = all.filter((e) => e.n_nodes <= 180);
    else if (filter === "correct") filtered = all.filter((e) => e.win_tag !== "wrong");
    if (!filtered.length) filtered = all;

    document.getElementById("event-count").textContent = filtered.length;
    const sel = document.getElementById("event-select");
    sel.innerHTML = filtered
      .map((e) => {
        const tag = e.win_tag === "highlight" ? "★ " : "";
        return `<option value="${e.tid}">${tag}[${e.true_label}] #${e.tid} (${e.n_nodes} nút) — ${e.snippet.slice(0, 46)}...</option>`;
      })
      .join("");
    STATE.tid = sel.value;
  }

  async function loadCurrentEvent() {
    if (!STATE.dataset || !STATE.tid) return;
    const [ev, early] = await Promise.all([
      apiGet(`/api/event?dataset=${encodeURIComponent(STATE.dataset)}&tid=${encodeURIComponent(STATE.tid)}&max_delay=-1`),
      apiGet(`/api/early_trajectory?dataset=${encodeURIComponent(STATE.dataset)}&tid=${encodeURIComponent(STATE.tid)}`),
    ]);
    STATE.eventData = ev;
    STATE.earlyData = early;

    document.getElementById("ev-meta").textContent = `Dataset: ${ev.dataset} • Event ID: ${ev.tid}`;
    document.getElementById("ev-true-badge").textContent = `Nhãn thật: ${ev.true_label_vi}`;
    document.getElementById("ev-tweet-text").textContent = ev.text;
    document.getElementById("ev-tree-stats").textContent = `Số nút: ${ev.tree_viz.n_full} | Số cạnh: ${ev.tree_viz.e_full} | Độ sâu tối đa: ${ev.tree_viz.max_depth}`;

    canvas1.setTree(ev.tree_viz);
    renderShowdownTable("showdown-table", ev.models, ev.y_true);
    renderXaiSection(ev.xai, "fusion-strip", "branch-bars", "top-tokens-list", "graph-stats-table", "xai-auto-loo");

    renderEarlyTrajectoryChart(early);
    document.getElementById("early-slider").value = 7;
    await updateEarlyStep(7);
  }

  function renderEarlyTrajectoryChart(early) {
    const canvas = document.getElementById("early-chart-canvas");
    const ctx = canvas.getContext("2d");
    const w = canvas.width;
    const h = canvas.height;
    ctx.clearRect(0, 0, w, h);

    document.getElementById("early-true-note").textContent = `Nhãn thật sự kiện: ${early.true_label}`;
    const pad = { l: 46, r: 18, t: 22, b: 34 };
    const pw = w - pad.l - pad.r;
    const ph = h - pad.t - pad.b;

    // Grid
    ctx.strokeStyle = "#e2e8f0";
    ctx.lineWidth = 1;
    ctx.font = "11px Segoe UI";
    ctx.fillStyle = "#64748b";
    for (let g = 0; g <= 4; g++) {
      const y = pad.t + (ph * g) / 4;
      ctx.beginPath();
      ctx.moveTo(pad.l, y);
      ctx.lineTo(w - pad.r, y);
      ctx.stroke();
      ctx.fillText((1 - g * 0.25).toFixed(2).replace(".", ","), 8, y + 4);
    }

    const pts = early.points;
    const xPos = (i) => pad.l + (pw * i) / Math.max(pts.length - 1, 1);
    const yPos = (v) => pad.t + ph * (1 - v);

    // Draw 4 class probability curves of BiGCN-Plus-Clean
    for (let c = 0; c < 4; c++) {
      ctx.strokeStyle = CLS_COLORS[c];
      ctx.lineWidth = c === early.y_true ? 3.0 : 1.7;
      ctx.setLineDash([]);
      ctx.beginPath();
      pts.forEach((pt, i) => {
        const x = xPos(i);
        const y = yPos(pt.plus_probs[c]);
        if (i === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
      });
      ctx.stroke();

      pts.forEach((pt, i) => {
        ctx.fillStyle = CLS_COLORS[c];
        ctx.beginPath();
        ctx.arc(xPos(i), yPos(pt.plus_probs[c]), c === early.y_true ? 4 : 2.8, 0, Math.PI * 2);
        ctx.fill();
      });
    }

    // Also draw BiGCN-Base true-class probability dashed curve
    ctx.strokeStyle = "#334155";
    ctx.lineWidth = 1.8;
    ctx.setLineDash([5, 4]);
    ctx.beginPath();
    pts.forEach((pt, i) => {
      const x = xPos(i);
      const y = yPos(pt.base_probs[early.y_true]);
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    });
    ctx.stroke();
    ctx.setLineDash([]);

    // X labels & legend
    pts.forEach((pt, i) => {
      ctx.fillStyle = "#334155";
      ctx.fillText(DEADLINE_SHORT[i] || pt.minutes, xPos(i) - 10, h - 10);
    });

    // Legend strip
    const legItems = [
      ...CLS_NAMES.map((nm, idx) => ({ nm: `BiGCN+ (${nm})`, c: CLS_COLORS[idx] })),
      { nm: `BiGCN-Base (${early.true_label})`, c: "#334155" },
    ];
    let lx = pad.l;
    legItems.forEach((it) => {
      ctx.fillStyle = it.c;
      ctx.fillRect(lx, 7, 12, 4);
      ctx.fillStyle = "#1e293b";
      ctx.fillText(it.nm, lx + 16, 13);
      lx += 116;
    });

    // Step table
    document.querySelector("#early-step-table tbody").innerHTML = pts
      .map(
        (pt) => `
      <tr>
        <td>${pt.label}</td>
        <td class="right">${pt.n_nodes}</td>
        <td class="right">${pt.max_depth}</td>
        <td><span class="lbl-chip lbl-${pt.plus_pred}">${pt.plus_pred}</span> (${(Math.max(...pt.plus_probs) * 100).toFixed(1)}%)</td>
        <td><span class="lbl-chip lbl-${pt.base_pred}">${pt.base_pred}</span></td>
        <td><span class="lbl-chip lbl-${pt.ragcl_pred}">${pt.ragcl_pred}</span></td>
      </tr>`
      )
      .join("");
  }

  async function updateEarlyStep(stepIdx) {
    if (!STATE.earlyData) return;
    const pt = STATE.earlyData.points[stepIdx];
    if (!pt) return;
    document.getElementById("early-time-label").textContent = pt.label;
    document.getElementById("early-kpi").innerHTML = `
      <div class="kpi-chip">Số nút: <strong>${pt.n_nodes}</strong></div>
      <div class="kpi-chip">Độ sâu: <strong>${pt.max_depth}</strong></div>
      <div class="kpi-chip">BiGCN-Plus: <span class="lbl-chip lbl-${pt.plus_pred}">${pt.plus_pred}</span></div>
      <div class="kpi-chip">BiGCN-Base: <span class="lbl-chip lbl-${pt.base_pred}">${pt.base_pred}</span></div>
    `;

    const evStep = await apiGet(
      `/api/event?dataset=${encodeURIComponent(STATE.dataset)}&tid=${encodeURIComponent(STATE.tid)}&max_delay=${pt.minutes}`
    );
    document.getElementById("early-tree-meta").textContent =
      `Số nút hiện có: ${evStep.tree_viz.n_full} | Số cạnh: ${evStep.tree_viz.e_full} | Độ sâu: ${evStep.tree_viz.max_depth}`;
    canvas2.setTree(evStep.tree_viz);
  }

  async function runWhatIf() {
    const text = document.getElementById("wi-text").value;
    const template_mode = document.getElementById("wi-template").value;
    const keep_edge_ratio = Number(document.getElementById("wi-keep-ratio").value) / 100.0;
    const res = await apiPost("/api/what_if", {
      dataset: STATE.dataset,
      text,
      template_mode,
      base_tid: STATE.tid,
      keep_edge_ratio,
      max_delay: -1,
    });
    STATE.whatIfData = res;
    document.getElementById("wi-tree-summary").textContent =
      `Nút: ${res.tree_viz.n_full} | Cạnh: ${res.tree_viz.e_full} | Độ sâu: ${res.tree_viz.max_depth}`;
    canvas3.setTree(res.tree_viz);
    renderShowdownTable("wi-model-table", res.models, null);
    renderXaiSection(res.xai, "wi-fusion-strip", "wi-branch-bars", "wi-tokens-list");
  }

  function renderBenchmarksTab() {
    if (!STATE.benchmarks) return;
    const ds = document.getElementById("bench-ds").value;
    const evalChoice = document.getElementById("bench-eval").value;
    const B = STATE.benchmarks;

    let rows = [];
    if (evalChoice === "primary") {
      rows = (B.master || []).filter((r) => r.Dataset === ds);
    } else {
      rows = (B.same_eval || []).filter((r) => r.Dataset === ds && r.EvalMode === evalChoice);
    }
    document.querySelector("#bench-master-table tbody").innerHTML = rows
      .map((r) => {
        const isPlus = String(r.Model).includes("BiGCN-Plus-Clean") || String(r.Model).includes("BiGCN-Plus-UserID");
        return `
        <tr class="${isPlus ? "row-highlight" : ""}">
          <td><code>${r.Model}</code></td>
          <td>${r.EvalMode}</td>
          <td class="right">${r.Runs}</td>
          <td class="right">${fmtNum(r.Acc_Mean, 5)} &plusmn; ${fmtNum(r.Acc_Std, 5)}</td>
          <td class="right">${fmtNum(r.F1_Mean, 5)} &plusmn; ${fmtNum(r.F1_Std, 5)}</td>
          <td class="right">${fmtNum(r.Acc_SeedStd, 5)}</td>
          <td class="right">${fmtNum(r.Kappa, 5)}</td>
          <td class="right">${fmtNum(r.AUC_ROC, 5)}</td>
          <td class="right">${fmtNum(r.Sec_Per_Ep, 4)}</td>
        </tr>`;
      })
      .join("");

    const spRows = (B.tree_sparsity || []).filter((r) => r.Dataset === ds);
    document.querySelector("#bench-sparsity-table tbody").innerHTML = spRows.length
      ? spRows
          .map(
            (r) => `
          <tr class="${r.Model === "BiGCN-Plus-Clean" ? "row-highlight" : ""}">
            <td><code>${r.Model}</code></td>
            <td>${r.Stratum}</td>
            <td class="right">${r.N_Events}</td>
            <td class="right">${fmtNum(r.Accuracy, 5)}</td>
            <td class="right">${fmtNum(r.Macro_F1, 5)}</td>
          </tr>`
          )
          .join("")
      : `<tr><td colspan="5">Chưa có tệp tree_sparsity_analysis.csv cho ${ds}. Hãy chạy Colab &amp; aggregate_results.py.</td></tr>`;

    const view = document.getElementById("bench-stat-view").value;
    const thead = document.querySelector("#bench-extra-table thead");
    const tbody = document.querySelector("#bench-extra-table tbody");

    if (view === "early") {
      const edRows = (B.early_detection || []).filter((r) => r.Dataset === ds);
      thead.innerHTML = `<tr><th>Mốc</th><th>Mô hình</th><th class="right">Acc Mean</th><th class="right">F1 Mean</th><th class="right">p vs BiGCN</th></tr>`;
      tbody.innerHTML = edRows.length
        ? edRows
            .map(
              (r) => `
            <tr class="${r.Model === "BiGCN-Plus-Clean" ? "row-highlight" : ""}">
              <td>${r.Deadline_Label}</td>
              <td><code>${r.Model}</code></td>
              <td class="right">${fmtNum(r.Acc_Mean, 5)}</td>
              <td class="right">${fmtNum(r.F1_Mean, 5)}</td>
              <td class="right">${r.Wilcoxon_vs_BiGCNBase !== null ? Number(r.Wilcoxon_vs_BiGCNBase).toExponential(2) : "—"}</td>
            </tr>`
            )
            .join("")
        : `<tr><td colspan="5">Chưa có tệp early_detection_summary.csv cho ${ds}.</td></tr>`;
    } else {
      const stRows = (B.statistical_tests || []).filter((r) => r.Dataset === ds && r.Ref_B === "BiGCN-Plus-Clean");
      thead.innerHTML = `<tr><th>Model A</th><th>Ref B</th><th class="right">&Delta;Acc (A-B)</th><th class="right">Paired-t p</th><th class="right">Wilcoxon p</th></tr>`;
      tbody.innerHTML = stRows
        .map(
          (r) => `
          <tr>
            <td><code>${r.Model_A}</code></td>
            <td><code>${r.Ref_B}</code></td>
            <td class="right">${Number(r.Acc_Diff_A_minus_B) >= 0 ? "+" : ""}${fmtNum(r.Acc_Diff_A_minus_B, 5)}</td>
            <td class="right">${Number(r.Acc_t_pval).toExponential(2)}</td>
            <td class="right">${Number(r.Acc_Wilcoxon_pval).toExponential(2)}</td>
          </tr>`
        )
        .join("");
    }
  }

  function setupListeners() {
    document.querySelectorAll(".tab-btn").forEach((btn) => {
      btn.addEventListener("click", () => {
        document.querySelectorAll(".tab-btn").forEach((b) => b.classList.remove("active"));
        document.querySelectorAll(".tab-panel").forEach((p) => p.classList.remove("active"));
        btn.classList.add("active");
        document.getElementById(`tab-${btn.dataset.tab}`).classList.add("active");
        if (btn.dataset.tab === "whatif" && !STATE.whatIfData && STATE.eventData) {
          document.getElementById("wi-text").value = STATE.eventData.text;
          runWhatIf();
        }
      });
    });

    document.getElementById("dataset-select").addEventListener("change", (e) => {
      STATE.dataset = e.target.value;
      populateEventSelect();
      loadCurrentEvent();
    });

    document.getElementById("filter-select").addEventListener("change", () => {
      populateEventSelect();
      loadCurrentEvent();
    });

    document.getElementById("event-select").addEventListener("change", (e) => {
      STATE.tid = e.target.value;
      loadCurrentEvent();
    });

    document.getElementById("btn-reset-zoom-1").addEventListener("click", () => canvas1.resetView());

    document.getElementById("early-slider").addEventListener("input", (e) => {
      updateEarlyStep(Number(e.target.value));
    });

    document.getElementById("btn-play-early").addEventListener("click", () => {
      if (STATE.playTimer) {
        clearInterval(STATE.playTimer);
        STATE.playTimer = null;
        document.getElementById("btn-play-early").textContent = "▶ Tự động phát quá trình lan truyền";
        return;
      }
      const slider = document.getElementById("early-slider");
      let cur = 0;
      slider.value = 0;
      updateEarlyStep(0);
      document.getElementById("btn-play-early").textContent = "⏸ Tạm dừng";
      STATE.playTimer = setInterval(() => {
        cur += 1;
        if (cur > 7) {
          clearInterval(STATE.playTimer);
          STATE.playTimer = null;
          document.getElementById("btn-play-early").textContent = "▶ Tự động phát quá trình lan truyền";
          return;
        }
        slider.value = cur;
        updateEarlyStep(cur);
      }, 1100);
    });

    document.getElementById("wi-keep-ratio").addEventListener("input", (e) => {
      document.getElementById("wi-ratio-val").textContent = `${e.target.value}%`;
    });

    document.querySelectorAll("[data-preset]").forEach((btn) => {
      btn.addEventListener("click", () => {
        const p = btn.dataset.preset;
        const box = document.getElementById("wi-text");
        if (p === "current" && STATE.eventData) box.value = STATE.eventData.text;
        else if (p === "rumor") box.value = "BREAKING: Unconfirmed reports say explosion hits major bridge, multiple casualties suspected #rumor";
        else if (p === "news") box.value = "Official NASA statement confirms new solar probe launch scheduled tomorrow morning at 7 AM EST via URL";
      });
    });

    document.getElementById("btn-run-whatif").addEventListener("click", runWhatIf);
    ["bench-ds", "bench-eval", "bench-stat-view"].forEach((id) => {
      document.getElementById(id).addEventListener("change", renderBenchmarksTab);
    });
  }

  async function init() {
    canvas1 = new TreeCanvas("tree-canvas-1", "tree-tooltip-1");
    canvas2 = new TreeCanvas("tree-canvas-2", "tree-tooltip-2");
    canvas3 = new TreeCanvas("tree-canvas-3", "tree-tooltip-3");
    setupListeners();

    const [meta, bench] = await Promise.all([apiGet("/api/meta"), apiGet("/api/benchmarks")]);
    STATE.meta = meta;
    STATE.benchmarks = bench;

    const dsSel = document.getElementById("dataset-select");
    if (!meta.available_datasets.length) {
      document.getElementById("bundle-pill").textContent = "Chưa có gói trọng số .pt — chỉ xem được Tab 4 Benchmark";
      renderBenchmarksTab();
      return;
    }

    dsSel.innerHTML = meta.available_datasets.map((d) => `<option value="${d}">${d}</option>`).join("");
    STATE.dataset = meta.available_datasets.includes("Twitter16") ? "Twitter16" : meta.available_datasets[0];
    dsSel.value = STATE.dataset;

    const bInfo = meta.bundles_info[STATE.dataset];
    document.getElementById("bundle-pill").textContent =
      `Bundle ${STATE.dataset}: Fold ${bInfo.fold} (Seed ${bInfo.seed}, ${bInfo.device_trained}) • ${bInfo.n_test_events} mẫu Test`;

    populateEventSelect();
    await loadCurrentEvent();
    renderBenchmarksTab();
  }

  window.addEventListener("DOMContentLoaded", init);
})();
