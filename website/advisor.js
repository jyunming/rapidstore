/* advisor.js — Config Advisor for TurboQuantDB */

(function () {
    'use strict';

    // ── State ────────────────────────────────────────────────────────────────
    let advisorData = null;
    let userDim   = 1536;
    let recallKey = '8';
    let wRecall   = 0.50;
    let wCompr    = 0.35;
    let wSpeed    = 0.15;

    const SCENARIOS = {
        rag:      { r: 60, c: 25, s: 15, rk: '8'  },
        search:   { r: 45, c: 35, s: 20, rk: '16' },
        laptop:   { r: 25, c: 60, s: 15, rk: '8'  },
        accuracy: { r: 75, c: 10, s: 15, rk: '4'  },
        balanced: { r: 50, c: 35, s: 15, rk: '8'  },
        ingest:   { r: 35, c: 30, s: 35, rk: '8'  },
    };

    const BIT_COLORS = { 1: '#ff8b73', 2: '#facc15', 3: '#60a5fa', 4: '#00f0ff' };

    // ── DOM refs ─────────────────────────────────────────────────────────────
    const dimInput     = document.getElementById('dim-input');
    const dimMatch     = document.getElementById('dim-match');
    const dimPresets   = document.getElementById('dim-presets');
    const topkRow      = document.getElementById('topk-row');
    const scenarioGrid = document.getElementById('scenario-grid');
    const sliderRecall = document.getElementById('slider-recall');
    const sliderCompr  = document.getElementById('slider-compr');
    const sliderSpeed  = document.getElementById('slider-speed');
    const pctRecall    = document.getElementById('pct-recall');
    const pctCompr     = document.getElementById('pct-compr');
    const pctSpeed     = document.getElementById('pct-speed');
    const resultsPanel = document.getElementById('results-panel');

    // ── Wire controls immediately so clicks always work
    initControls();
    applyScenario('rag');
    scenarioGrid.querySelector('[data-scenario="rag"]').classList.add('active');

    // ── Data is preloaded as a global by data/advisor_data.js (no fetch — works on file:// too)
    if (window.__TQDB_ADVISOR_DATA__) {
        advisorData = window.__TQDB_ADVISOR_DATA__;
        render();
    } else {
        resultsPanel.innerHTML = `<div class="adv-no-data">Benchmark data script did not load. Make sure <code>data/advisor_data.js</code> is reachable from this page.</div>`;
    }

    // ── Nearest-dataset matching ─────────────────────────────────────────────
    function nearestDataset(dim) {
        if (!advisorData) return null;
        let best = null, bestDelta = Infinity;
        for (const [name, d] of Object.entries(advisorData.dims)) {
            const delta = Math.abs(d - dim);
            if (delta < bestDelta) { bestDelta = delta; best = name; }
        }
        return best;
    }

    // ── Scoring ──────────────────────────────────────────────────────────────
    function norm(vals) {
        const lo = Math.min(...vals), hi = Math.max(...vals);
        if (hi === lo) return vals.map(() => 1.0);
        return vals.map(v => (v - lo) / (hi - lo));
    }

    function scoreConfigs(rows) {
        const recallVals = rows.map(r => r.rk[recallKey] || 0);
        const comprVals  = rows.map(r => r.compr || 0);
        const speedVals  = rows.map(r => r.p50 > 0 ? 1000 / r.p50 : 0);

        const nr = norm(recallVals);
        const nc = norm(comprVals);
        const ns = norm(speedVals);

        return rows.map((r, i) => ({
            ...r,
            _score: wRecall * nr[i] + wCompr * nc[i] + wSpeed * ns[i]
        }));
    }

    // ── Render ───────────────────────────────────────────────────────────────
    function render() {
        if (!advisorData) return;
        const ds = nearestDataset(userDim);
        if (!ds) return;

        const d = advisorData.dims[ds];
        dimMatch.innerHTML = `Matched to <span>${ds} (d=${d})</span>`;

        const rows = advisorData.configs[ds] || [];
        if (rows.length === 0) {
            resultsPanel.innerHTML = '<div class="adv-no-data">No data for this dataset.</div>';
            return;
        }

        const scored = scoreConfigs(rows).sort((a, b) => b._score - a._score);
        const top3   = scored.slice(0, 3);

        resultsPanel.innerHTML = '';

        // ─ Hero row: top recommendation (left) + scatter map (right)
        const heroRow = document.createElement('div');
        heroRow.className = 'results-hero';
        heroRow.appendChild(buildCard(top3[0], 0, d));
        heroRow.appendChild(buildScatter(scored, top3[0]));
        resultsPanel.appendChild(heroRow);

        // ─ Alternatives row: ranks 2 + 3 side-by-side
        if (top3.length > 1) {
            const altsRow = document.createElement('div');
            altsRow.className = 'results-alts';
            top3.slice(1).forEach((cfg, i) => altsRow.appendChild(buildCard(cfg, i + 1, d)));
            resultsPanel.appendChild(altsRow);
        }
    }

    // ── Card builder ─────────────────────────────────────────────────────────
    function colorClass(val, lo, hi) {
        if (val >= hi)             return 'good';
        if (val >= (lo + hi) / 2)  return 'ok';
        return 'warn';
    }

    function buildCard(cfg, rank, dim) {
        const recall = (cfg.rk[recallKey] * 100).toFixed(1);
        const compr  = cfg.compr.toFixed(1);
        const p50    = cfg.p50.toFixed(1);
        const disk1M = (cfg.disk * 100).toFixed(0);

        const card = document.createElement('div');
        card.className = `result-card${rank === 0 ? ' top' : ''}`;

        const badge = rank === 0
            ? '<span class="result-badge badge-best">★ Recommended</span>'
            : `<span class="result-badge badge-alt">#${rank + 1}</span>`;

        // Promote ri4 if rerank=true and dim ≥ 1024 (where ri4 is the recommended path)
        const showRi4 = cfg.rerank && dim >= 1024;
        const ri4Badge = showRi4 ? '<span class="result-badge badge-ri4">ri4 ready</span>' : '';

        const cfgLine = formatConfig(cfg);
        const snippet = pythonSnippet(cfg, showRi4);
        const tip = rank === 0 ? buildTip(cfg, dim) : '';

        const metricsHTML = `
            <div class="metric">
                <div class="metric-num ${colorClass(parseFloat(recall), 50, 90)}">${recall}<small>%</small></div>
                <div class="metric-label">R@${recallKey}</div>
            </div>
            <div class="metric">
                <div class="metric-num ${colorClass(parseFloat(compr), 2, 6)}">${compr}<small>×</small></div>
                <div class="metric-label">smaller</div>
            </div>
            <div class="metric">
                <div class="metric-num ${colorClass(10 - parseFloat(p50), 0, 8)}">${p50}<small>ms</small></div>
                <div class="metric-label">p50</div>
            </div>
            ${rank === 0 ? `<div class="metric">
                <div class="metric-num">${disk1M}<small>MB</small></div>
                <div class="metric-label">disk @ 1M</div>
            </div>` : ''}
        `;

        card.innerHTML = `
            <div class="result-rank">
                ${badge}
                ${ri4Badge}
            </div>
            <div class="result-config-line">${cfgLine}</div>
            <div class="result-metrics">${metricsHTML}</div>
            ${rank === 0 ? `<div class="result-snippet">${snippet}</div>` : ''}
            ${tip ? `<div class="result-tip">${tip}</div>` : ''}
        `;
        return card;
    }

    function formatConfig(cfg) {
        const parts = [`<span class="accent">b=${cfg.bits}</span>`];
        if (cfg.rerank) parts.push('rerank');
        if (cfg.ann)    parts.push('ann');
        return parts.join(' · ');
    }

    function pythonSnippet(cfg, useRi4) {
        const parts = [];
        parts.push(`<span class="py-key">bits</span>=<span class="py-num">${cfg.bits}</span>`);
        parts.push(`<span class="py-key">rerank</span>=<span class="py-bool">${cfg.rerank ? 'True' : 'False'}</span>`);
        if (cfg.rerank && useRi4) {
            parts.push(`<span class="py-key">rerank_precision</span>=<span class="py-str">"residual_int4"</span>`);
        }
        return `Database.open(<span class="py-str">"./db"</span>, dim=${cfg.d}, ${parts.join(', ')})`;
    }

    function buildTip(cfg, dim) {
        if (dim <= 384 && !cfg.rerank) {
            return `<strong>Heads-up:</strong> at low dimensions (≤384) codes alone often miss recall — your recall&nbsp;score reflects that. If you can spare disk, enable <code>rerank=True</code>.`;
        }
        if (dim >= 1024 && cfg.rerank) {
            return `<strong>Tip:</strong> at d=${dim}, codes alone (rerank=False) recover most of the recall at <strong>0&nbsp;extra disk</strong>. Worth a try if you're disk-bound.`;
        }
        if (dim >= 1024 && !cfg.rerank) {
            return `<strong>Nice fit:</strong> high-dim embeddings keep enough signal that codes alone hit good recall — no rerank file needed.`;
        }
        return `<strong>Bring your own model:</strong> any embedder works (sentence-transformers, OpenAI, Cohere, …). TQDB only stores vectors.`;
    }

    // ── Scatter ──────────────────────────────────────────────────────────────
    function buildScatterSVG(scored, topCfg) {
        const PAD = { left: 46, right: 16, top: 14, bottom: 36 };
        const W = 480, H = 200;
        const pw = W - PAD.left - PAD.right;
        const ph = H - PAD.top  - PAD.bottom;

        const xs = scored.map(r => r.rk[recallKey] || 0);
        const ys = scored.map(r => r.compr || 0);
        const xMax = Math.max(...xs) * 1.05 || 1;
        const yMax = Math.max(...ys) * 1.10 || 1;

        const px = v => PAD.left + (v / xMax) * pw;
        const py = v => PAD.top  + ph - (v / yMax) * ph;

        let inner = '';

        // Grid + axis labels
        for (let i = 0; i <= 5; i++) {
            const x = (i / 5) * xMax;
            const sx = px(x);
            inner += `<line x1="${sx}" y1="${PAD.top}" x2="${sx}" y2="${PAD.top + ph}" stroke="rgba(255,255,255,0.06)" stroke-width="1"/>`;
            inner += `<text x="${sx}" y="${PAD.top + ph + 16}" text-anchor="middle" font-size="10" fill="rgba(255,255,255,0.5)" font-family="JetBrains Mono, monospace">${Math.round(x * 100)}%</text>`;
        }
        for (let i = 0; i <= 4; i++) {
            const y = (i / 4) * yMax;
            const sy = py(y);
            inner += `<line x1="${PAD.left}" y1="${sy}" x2="${PAD.left + pw}" y2="${sy}" stroke="rgba(255,255,255,0.06)" stroke-width="1"/>`;
            inner += `<text x="${PAD.left - 6}" y="${sy + 4}" text-anchor="end" font-size="10" fill="rgba(255,255,255,0.5)" font-family="JetBrains Mono, monospace">${y.toFixed(1)}×</text>`;
        }

        // Dots (non-top first so the star sits on top)
        for (const r of scored) {
            if (r === topCfg) continue;
            const col = BIT_COLORS[r.bits] || '#888';
            inner += `<circle cx="${px(r.rk[recallKey] || 0)}" cy="${py(r.compr || 0)}" r="5" fill="${col}" fill-opacity="0.55" stroke="${col}" stroke-opacity="0.85" stroke-width="1"/>`;
        }
        if (topCfg) {
            const cx = px(topCfg.rk[recallKey] || 0);
            const cy = py(topCfg.compr || 0);
            const col = BIT_COLORS[topCfg.bits] || '#888';
            inner += `<circle cx="${cx}" cy="${cy}" r="11" fill="none" stroke="${col}" stroke-width="1.5" stroke-opacity="0.6"/>`;
            inner += `<circle cx="${cx}" cy="${cy}" r="6" fill="${col}"/>`;
            inner += `<text x="${cx}" y="${cy - 14}" text-anchor="middle" font-size="13" fill="${col}" font-weight="700">★</text>`;
        }

        // Build the entire <svg>...</svg> as a single string (parsed in SVG namespace).
        return `<svg class="scatter-svg" viewBox="0 0 ${W} ${H}" preserveAspectRatio="xMidYMid meet" width="100%" height="${H}" aria-label="All configs scatter">${inner}</svg>`;
    }

    function buildScatter(scored, topCfg) {
        const block = document.createElement('div');
        block.className = 'scatter-block';
        block.innerHTML = `
            <div class="scatter-block-title">
                <span>All combinations · ${scored.length} configs</span>
                <span class="scatter-axes">x: R@${recallKey} · y: compression × · ★ = recommended</span>
            </div>
            ${buildScatterSVG(scored, topCfg)}
            <div class="scatter-legend">
                ${[1,2,3,4].map(b =>
                    `<span><i class="legend-dot" style="background:${BIT_COLORS[b]}"></i>b=${b}</span>`
                ).join('')}
            </div>
        `;
        return block;
    }

    // ── Sliders / weights ────────────────────────────────────────────────────
    function updateWeights() {
        const r = parseFloat(sliderRecall.value);
        const c = parseFloat(sliderCompr.value);
        const s = parseFloat(sliderSpeed.value);
        const total = r + c + s;
        if (total === 0) {
            wRecall = wCompr = wSpeed = 1 / 3;
        } else {
            wRecall = r / total; wCompr = c / total; wSpeed = s / total;
        }
        pctRecall.textContent = Math.round(wRecall * 100) + '%';
        pctCompr.textContent  = Math.round(wCompr  * 100) + '%';
        pctSpeed.textContent  = Math.round(wSpeed  * 100) + '%';
    }

    function applyScenario(key) {
        const s = SCENARIOS[key];
        if (!s) return;
        sliderRecall.value = s.r;
        sliderCompr.value  = s.c;
        sliderSpeed.value  = s.s;
        updateWeights();
        recallKey = s.rk;
        topkRow.querySelectorAll('.chip').forEach(btn => {
            btn.classList.toggle('active', btn.dataset.rk === recallKey);
        });
    }

    // ── Init controls ────────────────────────────────────────────────────────
    function initControls() {
        scenarioGrid.querySelectorAll('.scenario-btn').forEach(btn => {
            btn.addEventListener('click', () => {
                scenarioGrid.querySelectorAll('.scenario-btn').forEach(b => b.classList.remove('active'));
                btn.classList.add('active');
                applyScenario(btn.dataset.scenario);
                render();
            });
        });

        dimPresets.querySelectorAll('.chip[data-dim]').forEach(btn => {
            btn.addEventListener('click', () => {
                dimPresets.querySelectorAll('.chip[data-dim]').forEach(b => b.classList.remove('active'));
                btn.classList.add('active');
                userDim = parseInt(btn.dataset.dim, 10);
                dimInput.value = userDim;
                render();
            });
            if (parseInt(btn.dataset.dim, 10) === userDim) btn.classList.add('active');
        });

        dimInput.addEventListener('input', () => {
            const v = parseInt(dimInput.value, 10);
            if (!isNaN(v) && v > 0) {
                userDim = v;
                dimPresets.querySelectorAll('.chip[data-dim]').forEach(btn => {
                    btn.classList.toggle('active', parseInt(btn.dataset.dim, 10) === userDim);
                });
                render();
            }
        });

        topkRow.querySelectorAll('.chip').forEach(btn => {
            btn.addEventListener('click', () => {
                topkRow.querySelectorAll('.chip').forEach(b => b.classList.remove('active'));
                btn.classList.add('active');
                recallKey = btn.dataset.rk;
                render();
            });
        });

        [sliderRecall, sliderCompr, sliderSpeed].forEach(sl => {
            sl.addEventListener('input', () => {
                scenarioGrid.querySelectorAll('.scenario-btn').forEach(b => b.classList.remove('active'));
                updateWeights();
                render();
            });
        });

        updateWeights();
    }
})();
