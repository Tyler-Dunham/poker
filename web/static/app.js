(() => {
  "use strict";

  const RANKS = ["A", "K", "Q", "J", "T", "9", "8", "7", "6", "5", "4", "3", "2"];
  const SUIT_SYMBOL = { s: "♠", h: "♥", d: "♦", c: "♣" };
  const RED_SUITS = new Set(["h", "d"]);

  const state = {
    samples: [],
    profiles: [],
    activeProfileId: null,
    activeProfile: null,
    rlReport: null,
    selectedSpotIdx: 0,
    gameSessionId: null,
    gameState: null,
    coachFeedback: null,
    handsPlayedInSession: 0,
    handOver: false,
  };

  const $ = (id) => document.getElementById(id);

  function showToast(message, isError = false) {
    const el = $("toast");
    el.textContent = message;
    el.classList.toggle("error", isError);
    el.classList.remove("hidden");
    clearTimeout(showToast._timer);
    showToast._timer = setTimeout(() => el.classList.add("hidden"), 3500);
  }

  async function api(path, options = {}) {
    const res = await fetch(path, {
      headers: { "Content-Type": "application/json", ...(options.headers || {}) },
      ...options,
    });
    const data = await res.json().catch(() => ({}));
    if (!res.ok) {
      throw new Error(data.error || `Request failed (${res.status})`);
    }
    return data;
  }

  function pct(val) {
    if (val === null || val === undefined || Number.isNaN(Number(val))) return "—";
    const num = Number(val);
    const normalized = num <= 1.0 && num > 0 && String(val).includes(".") ? num * 100 : num;
    return `${normalized.toFixed(1)}%`;
  }

  function numFmt(val, digits = 1) {
    if (val === null || val === undefined || Number.isNaN(Number(val))) return "—";
    return Number(val).toFixed(digits);
  }

  function escapeHtml(str) {
    return String(str ?? "")
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;");
  }

  // Navigation Tabs
  function switchView(viewName) {
    document.querySelectorAll(".nav-tab").forEach((btn) => {
      const active = btn.dataset.view === viewName;
      btn.classList.toggle("active", active);
      btn.setAttribute("aria-selected", String(active));
    });
    document.querySelectorAll(".view").forEach((sec) => {
      sec.classList.toggle("active", sec.id === `view-${viewName}`);
    });
  }

  // Initialization
  async function init() {
    bindEvents();
    try {
      const [samples, profiles] = await Promise.all([
        api("/api/samples"),
        api("/api/profiles"),
      ]);
      state.samples = Array.isArray(samples) ? samples : [];
      state.profiles = Array.isArray(profiles) ? profiles : [];
      renderSampleButtons();
      renderProfileSelector();
      if (state.profiles.length > 0) {
        await selectProfile(state.profiles[0].id);
      } else {
        renderProfileView();
      }
    } catch (err) {
      showToast(err.message, true);
    }
  }

  function bindEvents() {
    document.querySelectorAll(".nav-tab").forEach((btn) => {
      btn.addEventListener("click", () => switchView(btn.dataset.view));
    });

    $("profileSelect").addEventListener("change", (e) => {
      if (e.target.value) selectProfile(e.target.value);
    });

    // Import Modal
    const openModal = () => $("importModal").classList.remove("hidden");
    const closeModal = () => $("importModal").classList.add("hidden");
    $("openImportBtn").addEventListener("click", openModal);
    $("emptyImportBtn").addEventListener("click", openModal);
    $("closeImportBtn").addEventListener("click", closeModal);
    $("cancelImportBtn").addEventListener("click", closeModal);
    $("importModal").addEventListener("click", (e) => {
      if (e.target === $("importModal")) closeModal();
    });

    // Drag & drop + file browse
    const dropZone = $("dropZone");
    const fileInput = $("fileInput");
    $("browseFileBtn").addEventListener("click", () => fileInput.click());
    fileInput.addEventListener("change", (e) => {
      if (e.target.files[0]) readUploadedFile(e.target.files[0]);
    });
    dropZone.addEventListener("dragover", (e) => {
      e.preventDefault();
      dropZone.classList.add("dragover");
    });
    dropZone.addEventListener("dragleave", () => dropZone.classList.remove("dragover"));
    dropZone.addEventListener("drop", (e) => {
      e.preventDefault();
      dropZone.classList.remove("dragover");
      if (e.dataTransfer.files[0]) readUploadedFile(e.dataTransfer.files[0]);
    });

    $("submitImportBtn").addEventListener("click", handleImportSubmit);

    // RL Lab
    $("trainRlBtn").addEventListener("click", handleTrainRl);
    $("spotSelect").addEventListener("change", (e) => {
      state.selectedSpotIdx = Number(e.target.value) || 0;
      renderSpotComparison();
    });

    // Play Yourself
    $("startGameBtn").addEventListener("click", () => startGameSession());
    $("nextHandBtn").addEventListener("click", handleNextHand);
    $("saveToProfileBtn").addEventListener("click", handleSaveToProfile);

    $("btnFold").addEventListener("click", () => sendAction("fold"));
    $("btnCallCheck").addEventListener("click", () => {
      const kind = $("btnCallCheck").dataset.kind || "check";
      sendAction(kind);
    });
    $("btnBet33").addEventListener("click", () => sendSizedBet(0.33));
    $("btnBet66").addEventListener("click", () => sendSizedBet(0.66));
    $("btnBetPot").addEventListener("click", () => sendSizedBet(1.0));
    $("btnAllIn").addEventListener("click", () => {
      const maxTo = state.gameState?.max_raise_to;
      const kind = (state.gameState?.legal_actions || []).includes("raise") ? "raise" : "bet";
      if (maxTo != null) sendAction(kind, maxTo);
    });
    $("betSlider").addEventListener("input", (e) => {
      $("betAmountInput").value = e.target.value;
    });
    $("betAmountInput").addEventListener("input", (e) => {
      $("betSlider").value = e.target.value;
    });
    $("btnCustomBet").addEventListener("click", () => {
      const amt = Number($("betAmountInput").value);
      const legal = state.gameState?.legal_actions || [];
      const kind = legal.includes("raise") ? "raise" : "bet";
      sendAction(kind, amt);
    });
  }

  function readUploadedFile(file) {
    const reader = new FileReader();
    reader.onload = () => {
      $("importRawText").value = reader.result || "";
      if (!$("importName").value) {
        $("importName").value = file.name.replace(/\.[^.]+$/, "");
      }
      showToast(`Loaded ${file.name}`);
    };
    reader.readAsText(file);
  }

  function renderSampleButtons() {
    const targets = [$("emptySampleButtons"), $("modalSampleButtons")];
    const html = state.samples
      .map(
        (s) => `
      <button type="button" class="sample-pack-btn" data-pack-id="${escapeHtml(s.id)}">
        <div class="sample-pack-title">${escapeHtml(s.name || s.id)}</div>
        <div class="sample-pack-desc">${escapeHtml(s.description || s.archetype || "Sample hand pack")}</div>
      </button>`
      )
      .join("");
    targets.forEach((el) => {
      el.innerHTML = html;
      el.querySelectorAll(".sample-pack-btn").forEach((btn) => {
        btn.addEventListener("click", () => loadSampleProfile(btn.dataset.packId));
      });
    });
  }

  function renderProfileSelector() {
    const sel = $("profileSelect");
    const opts = ['<option value="">Select a profile…</option>'].concat(
      state.profiles.map(
        (p) =>
          `<option value="${escapeHtml(p.id)}" ${p.id === state.activeProfileId ? "selected" : ""}>${escapeHtml(
            p.name
          )} (${p.hands_count ?? p.hand_count ?? "?"} hands)</option>`
      )
    );
    sel.innerHTML = opts.join("");
  }

  async function refreshProfilesList() {
    state.profiles = await api("/api/profiles");
    renderProfileSelector();
  }

  async function selectProfile(profileId) {
    try {
      const profile = await api(`/api/profiles/${encodeURIComponent(profileId)}`);
      state.activeProfileId = profileId;
      state.activeProfile = profile;
      state.rlReport = profile.rl_report || null;
      renderProfileSelector();
      renderProfileView();
      renderRlView();
    } catch (err) {
      showToast(err.message, true);
    }
  }

  async function loadSampleProfile(packId) {
    try {
      showToast("Generating sample hands & fitting clone…");
      const profile = await api("/api/profiles/sample", {
        method: "POST",
        body: JSON.stringify({ pack_id: packId }),
      });
      $("importModal").classList.add("hidden");
      state.activeProfileId = profile.id;
      state.activeProfile = profile;
      state.rlReport = profile.rl_report || null;
      await refreshProfilesList();
      renderProfileView();
      renderRlView();
      showToast(`Loaded "${profile.name}"`);
    } catch (err) {
      showToast(err.message, true);
    }
  }

  async function handleImportSubmit() {
    const raw_text = $("importRawText").value.trim();
    const name = $("importName").value.trim() || "Imported Player";
    const hero_name = $("importHeroName").value.trim() || undefined;
    const append = $("appendToCurrent").checked && state.activeProfileId;
    if (!raw_text) {
      showToast("Please paste or upload a hand history first.", true);
      return;
    }
    try {
      $("submitImportBtn").disabled = true;
      const profile = await api("/api/profiles/import", {
        method: "POST",
        body: JSON.stringify({
          name,
          hero_name,
          raw_text,
          profile_id: append ? state.activeProfileId : undefined,
        }),
      });
      $("importModal").classList.add("hidden");
      $("importRawText").value = "";
      state.activeProfileId = profile.id;
      state.activeProfile = profile;
      state.rlReport = profile.rl_report || null;
      await refreshProfilesList();
      renderProfileView();
      renderRlView();
      showToast(`Imported "${profile.name}"`);
    } catch (err) {
      showToast(err.message, true);
    } finally {
      $("submitImportBtn").disabled = false;
    }
  }

  // VIEW 1: PROFILE & RANGES
  function renderProfileView() {
    const p = state.activeProfile;
    if (!p) {
      $("profileEmptyState").classList.remove("hidden");
      $("profileDashboard").classList.add("hidden");
      return;
    }
    $("profileEmptyState").classList.add("hidden");
    $("profileDashboard").classList.remove("hidden");

    const stats = p.stats || {};
    const arch = p.archetype || stats.archetype || {};
    const handsCount = stats.hands_count ?? stats.total_hands ?? p.hands_count ?? 0;

    $("profileTitle").textContent = p.name || "Player Profile";
    $("handsCountBadge").textContent = `${handsCount} hands`;
    $("archetypeBadge").textContent = arch.archetype || arch.label || arch.name || "Balanced";
    $("archetypeDesc").textContent = arch.tagline || arch.description || arch.summary || "";
    $("rlTargetName").textContent = p.name || "your clone";

    // HUD Strip
    const hudItems = [
      { label: "VPIP", val: pct(stats.vpip ?? stats.vpip_pct) },
      { label: "PFR", val: pct(stats.pfr ?? stats.pfr_pct) },
      { label: "3-Bet", val: pct(stats.three_bet ?? stats.three_bet_pct) },
      { label: "Fold to 3B", val: pct(stats.fold_to_3bet ?? stats.fold_to_three_bet ?? stats.fold_to_3bet_pct) },
      { label: "Flop C-Bet", val: pct(stats.cbet_flop ?? stats.flop_cbet_pct) },
      { label: "Fold to CB", val: pct(stats.fold_to_cbet ?? stats.fold_to_cbet_pct) },
      { label: "Aggression (AF)", val: numFmt(stats.af ?? stats.aggression_factor, 2) },
      { label: "BB / 100", val: numFmt(stats.bb_per_100 ?? stats.bb100, 1) },
    ];
    $("hudStrip").innerHTML = hudItems
      .map(
        (h) => `
      <div class="hud-stat">
        <div class="hud-label">${h.label}</div>
        <div class="hud-value">${h.val}</div>
      </div>`
      )
      .join("");

    renderRangeMatrix(p.matrix || stats.preflop_matrix || stats.matrix_13x13);
    renderTraits(arch.traits || stats.traits || []);
    renderPositionalTable(stats.position_stats || stats.positional_stats || stats.by_position || {});
  }

  function comboName(r, c) {
    if (r === c) return RANKS[r] + RANKS[c];
    if (r < c) return RANKS[r] + RANKS[c] + "s";
    return RANKS[c] + RANKS[r] + "o";
  }

  function normalizeFreqs(cell) {
    if (!cell) return { raise: 0, call: 0, fold: 100, count: 0 };
    if ("vpip" in cell && "pfr" in cell && !("raise" in cell)) {
      const pfr = Math.max(0, Math.min(100, Number(cell.pfr || 0)));
      const vpip = Math.max(pfr, Math.min(100, Number(cell.vpip || 0)));
      return {
        raise: pfr,
        call: Math.max(0, vpip - pfr),
        fold: Math.max(0, 100 - vpip),
        count: cell.hands ?? cell.count ?? 0,
      };
    }
    if ("check_call" in cell || "bet_small" in cell) {
      const r = Number(cell.bet_small || 0) + Number(cell.bet_med || 0) + Number(cell.bet_big || 0) + Number(cell.all_in || 0);
      const c = Number(cell.check_call || 0);
      const f = Number(cell.fold || 0);
      const tot = r + c + f || 1;
      return {
        raise: (r / tot) * 100,
        call: (c / tot) * 100,
        fold: (f / tot) * 100,
        count: cell.count ?? cell.hands ?? 0,
      };
    }
    let r = Number(cell.raise ?? cell.raise_pct ?? 0);
    let c = Number(cell.call ?? cell.call_pct ?? 0);
    let f = Number(cell.fold ?? cell.fold_pct ?? 0);
    if (r + c + f <= 1.05 && (r > 0 || c > 0 || f > 0)) {
      r *= 100;
      c *= 100;
      f *= 100;
    }
    const total = r + c + f;
    if (total <= 0) return { raise: 0, call: 0, fold: 100, count: cell.count ?? cell.hands ?? 0 };
    return {
      raise: (r / total) * 100,
      call: (c / total) * 100,
      fold: (f / total) * 100,
      count: cell.count ?? cell.hands ?? 0,
    };
  }

  function renderRangeMatrix(matrixData) {
    const container = $("rangeMatrix");
    const cells = [];
    for (let r = 0; r < 13; r++) {
      for (let c = 0; c < 13; c++) {
        const combo = comboName(r, c);
        let rawCell = null;
        if (Array.isArray(matrixData) && Array.isArray(matrixData[r])) {
          rawCell = matrixData[r][c];
        } else if (matrixData && typeof matrixData === "object") {
          rawCell = matrixData[combo];
        }
        const freq = normalizeFreqs(rawCell);
        const rEnd = freq.raise.toFixed(1);
        const cEnd = (freq.raise + freq.call).toFixed(1);
        const bg = `linear-gradient(90deg, var(--raise) 0%, var(--raise) ${rEnd}%, var(--call) ${rEnd}%, var(--call) ${cEnd}%, #27303F ${cEnd}%, #27303F 100%)`;
        cells.push(
          `<div class="matrix-cell" data-combo="${combo}" data-raise="${freq.raise.toFixed(1)}" data-call="${freq.call.toFixed(1)}" data-fold="${freq.fold.toFixed(1)}" data-count="${freq.count}" style="background:${bg}">${combo}</div>`
        );
      }
    }
    container.innerHTML = cells.join("");

    const inspectCell = (el) => {
      container.querySelectorAll(".matrix-cell.selected").forEach((n) => n.classList.remove("selected"));
      el.classList.add("selected");
      const { combo, raise, call, fold, count } = el.dataset;
      $("inspectorCombo").textContent = combo;
      $("inspectorSub").textContent = `${count} observed hand${Number(count) === 1 ? "" : "s"}`;
      $("inspectorBars").innerHTML = `
        <div class="freq-bar-row">
          <div class="freq-bar-head"><span>Raise</span><span>${raise}%</span></div>
          <div class="freq-track"><div class="freq-fill-raise" style="width:${raise}%"></div></div>
        </div>
        <div class="freq-bar-row">
          <div class="freq-bar-head"><span>Call</span><span>${call}%</span></div>
          <div class="freq-track"><div class="freq-fill-call" style="width:${call}%"></div></div>
        </div>
        <div class="freq-bar-row">
          <div class="freq-bar-head"><span>Fold</span><span>${fold}%</span></div>
          <div class="freq-track"><div class="freq-fill-fold" style="width:${fold}%"></div></div>
        </div>`;
    };

    container.querySelectorAll(".matrix-cell").forEach((el) => {
      el.addEventListener("mouseenter", () => inspectCell(el));
      el.addEventListener("click", () => inspectCell(el));
    });
    const first = container.querySelector(".matrix-cell");
    if (first) inspectCell(first);
  }

  function renderTraits(traits) {
    const el = $("traitBars");
    if (Array.isArray(traits) && traits.length && typeof traits[0] === "string") {
      el.innerHTML = traits
        .map((t) => `<div class="log-entry">${escapeHtml(t)}</div>`)
        .join("");
      return;
    }
    const entries = Array.isArray(traits)
      ? traits.map((t) => [t.name || t.label, t.value ?? t.score])
      : Object.entries(traits || {});
    if (!entries.length) {
      el.innerHTML = '<div class="card-sub">No trait breakdown available.</div>';
      return;
    }
    el.innerHTML = entries
      .map(([name, val]) => {
        const num = Number(val);
        const width = Math.max(0, Math.min(100, num <= 1 ? num * 100 : num));
        return `
        <div>
          <div class="trait-row-head">
            <span>${escapeHtml(name)}</span>
            <span>${width.toFixed(0)}%</span>
          </div>
          <div class="trait-track"><div class="trait-fill" style="width:${width}%"></div></div>
        </div>`;
      })
      .join("");
  }

  function renderPositionalTable(posData) {
    const tbody = $("positionalTable").querySelector("tbody");
    const entries = Array.isArray(posData)
      ? posData.map((row) => [row.position || row.pos, row])
      : Object.entries(posData || {});
    if (!entries.length) {
      tbody.innerHTML = '<tr><td colspan="6" class="card-sub">No positional data yet.</td></tr>';
      return;
    }
    tbody.innerHTML = entries
      .map(
        ([pos, d]) => `
      <tr>
        <td>${escapeHtml(pos)}</td>
        <td>${d.hands ?? d.hands_count ?? 0}</td>
        <td>${pct(d.vpip ?? d.vpip_pct)}</td>
        <td>${pct(d.pfr ?? d.pfr_pct)}</td>
        <td>${pct(d.three_bet ?? d.three_bet_pct)}</td>
        <td>${numFmt(d.bb_per_100 ?? d.bb100, 1)}</td>
      </tr>`
      )
      .join("");
  }

  // VIEW 2: RL EXPLOIT LAB
  async function handleTrainRl() {
    if (!state.activeProfileId) {
      showToast("Import or select a profile first.", true);
      return;
    }
    const episodes = Number($("episodesSelect").value) || 2500;
    const btn = $("trainRlBtn");
    try {
      btn.disabled = true;
      btn.textContent = "Training RL Exploiter…";
      const report = await api(`/api/profiles/${encodeURIComponent(state.activeProfileId)}/train`, {
        method: "POST",
        body: JSON.stringify({ episodes }),
      });
      state.rlReport = report;
      state.selectedSpotIdx = 0;
      renderRlView();
      showToast(`RL training complete in ${report.training_ms ?? "—"} ms`);
    } catch (err) {
      showToast(err.message, true);
    } finally {
      btn.disabled = false;
      btn.textContent = "Train RL Exploiter";
    }
  }

  function renderRlView() {
    const r = state.rlReport;
    if (!r) {
      $("rlEmptyState").classList.remove("hidden");
      $("rlResults").classList.add("hidden");
      return;
    }
    $("rlEmptyState").classList.add("hidden");
    $("rlResults").classList.remove("hidden");

    const exploits = r.exploits || [];
    const strengths = r.strengths || [];
    $("rlExploitability").textContent = `${numFmt(r.exploitability_bb100, 1)} bb/100`;
    $("rlOverallScore").textContent = `${numFmt(r.overall_score, 0)} / 100`;
    $("rlExploitCount").textContent = String(exploits.length);
    $("rlTrainingMs").textContent = r.training_ms != null ? `${r.training_ms} ms` : "—";

    renderLearningCurve(r.learning_curve || []);
    renderExploits(exploits);
    renderStrengths(strengths);
    renderSpotSelector(r.spot_comparisons || []);
  }

  function renderLearningCurve(curve) {
    const wrap = $("learningCurveWrap");
    if (!curve.length) {
      wrap.innerHTML = '<div class="card-sub">No learning curve points recorded.</div>';
      return;
    }
    const pts = curve.map((pt, idx) => {
      if (typeof pt === "number") return { x: idx + 1, y: pt };
      return {
        x: pt.episode ?? pt.step ?? idx + 1,
        y: pt.avg_bb_per_100 ?? pt.cumulative_bb_per_100 ?? pt.ev_bb100 ?? pt.exploitability ?? pt.reward ?? pt.y ?? 0,
      };
    });
    const w = 520;
    const h = 210;
    const pad = { l: 44, r: 16, t: 16, b: 28 };
    const xs = pts.map((p) => p.x);
    const ys = pts.map((p) => p.y);
    const minX = Math.min(...xs);
    const maxX = Math.max(...xs) || 1;
    const minY = Math.min(0, ...ys);
    const maxY = Math.max(1, ...ys);
    const sx = (x) => pad.l + ((x - minX) / Math.max(1, maxX - minX)) * (w - pad.l - pad.r);
    const sy = (y) => h - pad.b - ((y - minY) / Math.max(0.001, maxY - minY)) * (h - pad.t - pad.b);

    const pathD = pts.map((p, i) => `${i === 0 ? "M" : "L"}${sx(p.x).toFixed(1)},${sy(p.y).toFixed(1)}`).join(" ");
    const areaD = `${pathD} L${sx(pts[pts.length - 1].x).toFixed(1)},${(h - pad.b).toFixed(1)} L${sx(pts[0].x).toFixed(1)},${(h - pad.b).toFixed(1)} Z`;

    wrap.innerHTML = `
      <svg viewBox="0 0 ${w} ${h}" preserveAspectRatio="none">
        <defs>
          <linearGradient id="rlAreaGrad" x1="0" y1="0" x2="0" y2="1">
            <stop offset="0%" stop-color="#E5484D" stop-opacity="0.35"/>
            <stop offset="100%" stop-color="#E5484D" stop-opacity="0.02"/>
          </linearGradient>
        </defs>
        <line x1="${pad.l}" y1="${pad.t}" x2="${pad.l}" y2="${h - pad.b}" stroke="#2A313C" />
        <line x1="${pad.l}" y1="${h - pad.b}" x2="${w - pad.r}" y2="${h - pad.b}" stroke="#2A313C" />
        <path d="${areaD}" fill="url(#rlAreaGrad)" />
        <path d="${pathD}" fill="none" stroke="#E5484D" stroke-width="2.5" stroke-linejoin="round" />
        <text x="${pad.l - 6}" y="${pad.t + 8}" fill="#8B949E" font-size="10" text-anchor="end">${maxY.toFixed(1)}</text>
        <text x="${pad.l - 6}" y="${h - pad.b}" fill="#8B949E" font-size="10" text-anchor="end">${minY.toFixed(1)}</text>
        <text x="${w - pad.r}" y="${h - 8}" fill="#8B949E" font-size="10" text-anchor="end">Ep ${maxX}</text>
      </svg>`;
  }

  function renderExploits(exploits) {
    const el = $("exploitsList");
    if (!exploits.length) {
      el.innerHTML = '<div class="card-sub">No major exploits found!</div>';
      return;
    }
    el.innerHTML = exploits
      .map((ex, idx) => {
        const sev = String(ex.severity || "MED").toUpperCase();
        const badgeCls = sev === "HIGH" ? "badge-high" : sev === "LOW" ? "badge-low" : "badge-med";
        const metricHtml = ex.metric
          ? `<div class="compare-bars">
              <div class="compare-pill">Observed<strong>${escapeHtml(ex.metric)}</strong></div>
              <div class="compare-pill" style="grid-column: span 2;">RL Counter<strong>${escapeHtml(ex.counter || ex.rl_counter || "Exploit")}</strong></div>
            </div>`
          : `<div class="compare-bars">
              <div class="compare-pill">Your %<strong>${pct(ex.your_pct ?? ex.player_pct)}</strong></div>
              <div class="compare-pill">GTO %<strong>${pct(ex.gto_pct)}</strong></div>
              <div class="compare-pill">RL Counter<strong>${escapeHtml(ex.rl_counter || ex.counter_action || "Exploit")}</strong></div>
            </div>`;
        return `
        <div class="exploit-item">
          <div class="exploit-head">
            <span class="exploit-title">${escapeHtml(ex.title || ex.spot || `Leak #${idx + 1}`)}</span>
            <span class="badge ${badgeCls}">${escapeHtml(sev)}</span>
          </div>
          <div class="exploit-desc">${escapeHtml(ex.description || ex.explanation || "")}</div>
          ${metricHtml}
          <button class="btn btn-secondary btn-sm practice-spot-btn" data-idx="${idx}">Practice vs Clone</button>
        </div>`;
      })
      .join("");

    el.querySelectorAll(".practice-spot-btn").forEach((btn) => {
      btn.addEventListener("click", () => {
        const ex = exploits[Number(btn.dataset.idx)];
        switchView("play");
        startGameSession(ex.drill_spot || ex.spot || ex.title);
      });
    });
  }

  function renderStrengths(strengths) {
    const el = $("strengthsList");
    if (!strengths.length) {
      el.innerHTML = '<div class="card-sub">Run RL training to identify strengths.</div>';
      return;
    }
    el.innerHTML = strengths
      .map(
        (s) => `
      <div class="strength-item">
        <div class="exploit-head">
          <span class="exploit-title">${escapeHtml(s.title || s.spot || "Solid Play")}</span>
          <span class="badge badge-call">SOLID</span>
        </div>
        <div class="exploit-desc">${escapeHtml(s.metric || s.description || s.explanation || "")}</div>
      </div>`
      )
      .join("");
  }

  function renderSpotSelector(spots) {
    const sel = $("spotSelect");
    if (!spots.length) {
      sel.innerHTML = "<option>No spots</option>";
      $("spotComparisonBody").innerHTML = '<div class="card-sub">No spot comparisons available.</div>';
      return;
    }
    sel.innerHTML = spots
      .map((s, i) => `<option value="${i}">${escapeHtml(s.spot || s.label || `Spot ${i + 1}`)}</option>`)
      .join("");
    sel.value = String(Math.min(state.selectedSpotIdx, spots.length - 1));
    renderSpotComparison();
  }

  function renderPolicyBar(label, probs) {
    const f = normalizeFreqs(probs);
    return `
      <div class="policy-row">
        <div class="policy-row-label">${escapeHtml(label)}</div>
        <div class="stacked-bar">
          <div class="seg-raise" style="width:${f.raise}%">${f.raise >= 12 ? `R ${f.raise.toFixed(0)}%` : ""}</div>
          <div class="seg-call" style="width:${f.call}%">${f.call >= 12 ? `C ${f.call.toFixed(0)}%` : ""}</div>
          <div class="seg-fold" style="width:${f.fold}%">${f.fold >= 12 ? `F ${f.fold.toFixed(0)}%` : ""}</div>
        </div>
      </div>`;
  }

  function renderSpotComparison() {
    const spots = state.rlReport?.spot_comparisons || [];
    const spot = spots[state.selectedSpotIdx];
    const body = $("spotComparisonBody");
    if (!spot) {
      body.innerHTML = '<div class="card-sub">Select a spot to inspect policies.</div>';
      return;
    }
    body.innerHTML =
      renderPolicyBar("Your Clone", spot.clone_probs || spot.clone || spot.your_policy) +
      renderPolicyBar("GTO Baseline", spot.gto_probs || spot.gto || spot.gto_policy) +
      renderPolicyBar("RL Exploiter", spot.exploiter_probs || spot.rl || spot.exploiter_policy);
  }

  // VIEW 3: PLAY YOURSELF
  async function startGameSession(drillSpot = null) {
    try {
      const payload = {
        profile_id: state.activeProfileId || "",
        opponent_mode: $("opponentModeSelect").value,
        num_seats: Number($("seatsSelect").value) || 2,
        starting_stack: 1000,
        big_blind: 10,
      };
      if (drillSpot) payload.drill_spot = drillSpot;
      const res = await api("/api/game/start", {
        method: "POST",
        body: JSON.stringify(payload),
      });
      state.gameSessionId = res.session_id;
      state.gameState = res.state;
      state.coachFeedback = res.coach_feedback || null;
      state.handOver = Boolean(res.hand_over);
      state.handsPlayedInSession = 0;
      renderGameTable();
      if (drillSpot) showToast(`Drilling spot: ${drillSpot}`);
    } catch (err) {
      showToast(err.message, true);
    }
  }

  async function sendAction(kind, amount = null) {
    if (!state.gameSessionId) return;
    try {
      const body = amount !== null ? { kind, amount } : { kind };
      const res = await api(`/api/game/${encodeURIComponent(state.gameSessionId)}/action`, {
        method: "POST",
        body: JSON.stringify(body),
      });
      state.gameState = res.state;
      state.coachFeedback = res.coach_feedback || null;
      state.handOver = Boolean(res.hand_over);
      if (state.handOver) {
        state.handsPlayedInSession += 1;
      }
      renderGameTable(res.hand_summary);
    } catch (err) {
      showToast(err.message, true);
    }
  }

  async function handleNextHand() {
    if (!state.gameSessionId) return;
    try {
      const res = await api(`/api/game/${encodeURIComponent(state.gameSessionId)}/next_hand`, {
        method: "POST",
        body: JSON.stringify({}),
      });
      state.gameState = res.state;
      state.handOver = false;
      state.coachFeedback = null;
      renderGameTable();
    } catch (err) {
      showToast(err.message, true);
    }
  }

  async function handleSaveToProfile() {
    if (!state.gameSessionId || !state.activeProfileId) return;
    try {
      const res = await api(`/api/game/${encodeURIComponent(state.gameSessionId)}/save_to_profile`, {
        method: "POST",
        body: JSON.stringify({}),
      });
      showToast(`Saved ${res.hands_added} hand(s) to profile!`);
      await selectProfile(state.activeProfileId);
    } catch (err) {
      showToast(err.message, true);
    }
  }

  function renderCard(cardStr, small = false) {
    if (!cardStr || cardStr === "??" || cardStr === "back") {
      return `<div class="playing-card ${small ? "sm" : ""} back"></div>`;
    }
    const rank = cardStr.slice(0, -1);
    const suitChar = cardStr.slice(-1).toLowerCase();
    const isRed = RED_SUITS.has(suitChar);
    return `
      <div class="playing-card ${small ? "sm" : ""} ${isRed ? "red" : ""}">
        <span class="card-rank">${escapeHtml(rank)}</span>
        <span class="card-suit">${SUIT_SYMBOL[suitChar] || suitChar}</span>
      </div>`;
  }

  function seatCoords(numSeats) {
    if (numSeats <= 2) {
      return [
        { left: "50%", top: "88%" },
        { left: "50%", top: "12%" },
      ];
    }
    return [
      { left: "50%", top: "88%" },
      { left: "15%", top: "70%" },
      { left: "15%", top: "28%" },
      { left: "50%", top: "11%" },
      { left: "85%", top: "28%" },
      { left: "85%", top: "70%" },
    ];
  }

  function renderGameTable(handSummary = null) {
    const gs = state.gameState;
    if (!gs) return;

    $("tablePot").textContent = String(gs.pot ?? 0);
    $("streetBadge").textContent = String(gs.street || "PREFLOP").toUpperCase();
    $("coachStreet").textContent = String(gs.street || "PREFLOP").toUpperCase();
    $("handNumberBadge").textContent = `Hand #${gs.hand_number ?? state.handsPlayedInSession + 1}`;
    $("handsPlayedCounter").textContent = `${state.handsPlayedInSession} hands played`;
    $("saveToProfileBtn").disabled = !state.activeProfileId || state.handsPlayedInSession === 0;
    $("nextHandBtn").classList.toggle("hidden", !state.handOver);

    // Board
    const board = gs.board || [];
    $("boardCards").innerHTML = board.map((c) => renderCard(c, false)).join("");

    // Seats
    const seats = gs.seats || [];
    const coords = seatCoords(seats.length);
    $("seatsContainer").innerHTML = seats
      .map((s, idx) => {
        const pos = coords[idx] || coords[0];
        const cards = (s.hole || []).map((c) => renderCard(c, true)).join("");
        const activeCls = s.is_turn && !state.handOver ? "active-turn" : "";
        const foldCls = s.folded ? "folded" : "";
        return `
        <div class="seat-box ${activeCls} ${foldCls}" style="left:${pos.left};top:${pos.top};">
          <div class="seat-cards">${cards}</div>
          <div class="seat-name">
            ${escapeHtml(s.name || `Seat ${idx}`)}
            ${s.is_button ? '<span class="dealer-chip">D</span>' : ""}
          </div>
          <div class="seat-stack">${s.stack ?? 0}</div>
          ${s.bet ? `<div class="seat-bet">Bet: ${s.bet}</div>` : ""}
        </div>`;
      })
      .join("");

    // Action Bar
    const legal = new Set(gs.legal_actions || []);
    const heroTurn = !state.handOver && legal.size > 0;
    $("btnFold").disabled = !heroTurn || !legal.has("fold");
    const canCall = legal.has("call");
    const canCheck = legal.has("check");
    $("btnCallCheck").disabled = !heroTurn || (!canCall && !canCheck);
    $("btnCallCheck").dataset.kind = canCall ? "call" : "check";
    $("btnCallCheck").textContent = canCall ? `Call ${gs.to_call ?? ""}` : "Check";

    const canRaiseOrBet = heroTurn && (legal.has("bet") || legal.has("raise"));
    ["btnBet33", "btnBet66", "btnBetPot", "btnAllIn", "btnCustomBet"].forEach((id) => {
      $(id).disabled = !canRaiseOrBet;
    });
    const minTo = gs.min_raise_to ?? 20;
    const maxTo = gs.max_raise_to ?? 1000;
    $("betSlider").min = minTo;
    $("betSlider").max = maxTo;
    $("betSlider").value = minTo;
    $("betAmountInput").value = minTo;

    // Coach Panel
    const coach = state.coachFeedback;
    if (coach) {
      $("coachBody").innerHTML =
        (coach.clone_probs ? renderPolicyBar("Your Clone Would Do", coach.clone_probs) : "") +
        (coach.gto_probs ? renderPolicyBar("GTO Recommends", coach.gto_probs) : "") +
        `<div class="coach-advice">${escapeHtml(coach.advice || coach.message || "")}</div>`;
    } else if (handSummary) {
      $("coachBody").innerHTML = `<div class="coach-advice"><strong>Hand Complete:</strong> ${escapeHtml(
        handSummary.summary || `Net: ${handSummary.hero_net ?? 0}`
      )}</div>`;
    }

    // Action Log
    const history = gs.history || [];
    $("actionLog").innerHTML = history
      .map(
        (ev) =>
          `<div class="log-entry">[${escapeHtml(ev.street)}] ${escapeHtml(ev.player || `Seat ${ev.seat}`)}: ${escapeHtml(
            ev.kind
          )} ${ev.amount ? ev.amount : ""}</div>`
      )
      .join("");
  }

  function sendSizedBet(potFraction) {
    const gs = state.gameState;
    if (!gs) return;
    const legal = gs.legal_actions || [];
    const kind = legal.includes("raise") ? "raise" : "bet";
    const minTo = gs.min_raise_to ?? 20;
    const maxTo = gs.max_raise_to ?? 1000;
    const raw = Math.round((gs.pot || 20) * potFraction) + (gs.to_call || 0);
    const clamped = Math.max(minTo, Math.min(maxTo, raw));
    sendAction(kind, clamped);
  }

  document.addEventListener("DOMContentLoaded", init);
})();
