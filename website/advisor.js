/* advisor.js — Interactive Config Advisor for TurboQuantDB */

(function () {
    'use strict';

    // ── State ─────────────────────────────────────────────────────────────────
    let advisorData = null;
    let userDim  = 768;
    let recallKey = '8';
    let wRecall  = 0.50;
    let wCompr   = 0.35;
    let wSpeed   = 0.15;

    // ── Scenario presets ──────────────────────────────────────────────────────
    // Each entry: [recall, compr, speed, recallKey]
    const SCENARIOS = {
        rag:      { r: 60, c: 25, s: 15, rk: '8',  label: 'RAG / Chatbot'     },
        search:   { r: 45, c: 35, s: 20, rk: '16', label: 'Search at Scale'   },
        laptop:   { r: 25, c: 60, s: 15, rk: '8',  label: 'Laptop / Edge'     },
        accuracy: { r: 75, c: 10, s: 15, rk: '4',  label: 'Max Accuracy'      },
        balanced: { r: 50, c: 35, s: 15, rk: '8',  label: 'Balanced'          },
        ingest:   { r: 35, c: 30, s: 35, rk: '8',  label: 'Fast Ingest'       },
    };

    // ── DOM refs ──────────────────────────────────────────────────────────────
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

    // ── Data loading ──────────────────────────────────────────────────────────
    fetch('data/advisor_data.json')
        .then(r => r.ok ? r.json() : Promise.reject('Failed to load benchmark data'))
        .then(data => {
            advisorData = data;
            initControls();
            render();
        })
        .catch(err => {
            resultsPanel.innerHTML = `<div class="no-data">Could not load benchmark data.<br><small>${err}</small></div>`;
        });

    // ── Nearest-dataset matching ──────────────────────────────────────────────
    function nearestDataset(dim) {
        if (!advisorData) return null;
        let best = null, bestDelta = Infinity;
        for (const [name, d] of Object.entries(advisorData.dims)) {
            const delta = Math.abs(d - dim);
            if (delta < bestDelta) { bestDelta = delta; best = name; }
        }
        return best;
    }

    // ── Scoring ───────────────────────────────────────────────────────────────
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

    // ── Rendering ─────────────────────────────────────────────────────────────
    function render() {
        if (!advisorData) return;

        const ds = nearestDataset(userDim);
        if (!ds) return;

        const d = advisorData.dims[ds];
        dimMatch.innerHTML = `Nearest benchmark dataset: <span>${ds} (d=${d})</span>`;

        const rows = advisorData.configs[ds] || [];
        if (rows.length === 0) {
            resultsPanel.innerHTML = '<div class="no-data">No data for this dataset.</div>';
            return;
        }

        const scored = scoreConfigs(rows).sort((a, b) => b._score - a._score);
        const top3   = scored.slice(0, 3);

        resultsPanel.innerHTML = '';

        // Scatter first (compact overview)
        renderScatter(scored, top3[0]);

        // Then ranked cards
        top3.forEach((cfg, idx) => renderCard(cfg, idx));
    }

    // ── Config label ──────────────────────────────────────────────────────────
    function cfgLabel(cfg) {
        const parts = [`b${cfg.bits}`];
        if (cfg.rerank) parts.push('rerank');
        if (cfg.ann)    parts.push('ann');
        if (cfg.fast)   parts.push('fast');
        parts.push(cfg.qtype || 'dense');
        return parts.join(' · ');
    }

    // ── Result card ───────────────────────────────────────────────────────────
    function colorClass(val, lo, hi) {
        const mid = (lo + hi) / 2;
        if (val >= hi)  return 'good';
        if (val >= mid) return 'ok';
        return 'warn';
    }

    function renderCard(cfg, rank) {
        const recall  = (cfg.rk[recallKey] * 100).toFixed(1);
        const compr   = cfg.compr.toFixed(1);
        const p50     = cfg.p50.toFixed(1);
        const disk1M  = (cfg.disk * 100).toFixed(0);
        const rk      = recallKey;

        const card = document.createElement('div');
        card.className = `result-card${rank === 0 ? ' top' : ''}`;

        const rankLabel = rank === 0
            ? '<span class="tag">⭐ Recommended</span>'
            : rank === 1
                ? '<span class="tag" style="background:rgba(255,255,255,0.08);color:var(--text-secondary)">#2</span>'
                : '<span class="tag" style="background:rgba(255,255,255,0.06);color:var(--text-secondary)">#3</span>';

        card.innerHTML = `
            <div class="result-header">
                <span class="result-rank-label">Rank ${rank + 1}</span>
            </div>
            <div class="result-config">
                <span style="font-family:'JetBrains Mono',monospace">${cfgLabel(cfg)}</span>${rankLabel}
            </div>
            <div class="result-dataset">${cfg.ds} · d=${cfg.d} · source: ${cfg.src === 'bit' ? 'bit_sweep' : 'dim_sweep'}</div>
            <div class="result-metrics">
                <div class="metric-box">
                    <div class="metric-value ${colorClass(parseFloat(recall), 30, 75)}">${recall}%</div>
                    <div class="metric-label">Recall@${rk}</div>
                </div>
                <div class="metric-box">
                    <div class="metric-value ${colorClass(parseFloat(compr), 2, 6)}">${compr}×</div>
                    <div class="metric-label">Compression</div>
                </div>
                <div class="metric-box">
                    <div class="metric-value ${colorClass(10 - parseFloat(p50), 0, 8)}">${p50}ms</div>
                    <div class="metric-label">p50 latency</div>
                </div>
                <div class="metric-box">
                    <div class="metric-value">${disk1M} MB</div>
                    <div class="metric-label">Disk @ 1M vecs</div>
                </div>
            </div>
            <div class="result-pip-row">
                ${bitPip(cfg.bits)}
                ${cfg.rerank ? '<span class="pip"><span class="dot" style="background:#34d399"></span>rerank</span>' : ''}
                ${cfg.ann    ? '<span class="pip"><span class="dot" style="background:#60a5fa"></span>ann</span>' : ''}
                ${cfg.fast   ? '<span class="pip"><span class="dot" style="background:#a78bfa"></span>fast_mode</span>' : ''}
                <span class="pip"><span class="dot" style="background:rgba(255,255,255,0.3)"></span>${cfg.qtype || 'dense'}</span>
                <span class="pip" style="margin-left:auto;font-size:0.7rem;opacity:0.6">score ${(cfg._score * 100).toFixed(0)}/100</span>
            </div>`;

        resultsPanel.appendChild(card);
    }

    function bitPip(bits) {
        const colors = { 1: '#f87171', 2: '#facc15', 3: '#60a5fa', 4: '#34d399' };
        return `<span class="pip"><span class="dot" style="background:${colors[bits] || '#888'}"></span>${bits}-bit</span>`;
    }

    // ── Compact scatter plot ──────────────────────────────────────────────────
    const BIT_COLORS = { 1: '#f87171', 2: '#facc15', 3: '#60a5fa', 4: '#34d399' };

    function renderScatter(scored, topCfg) {
        const card = document.createElement('div');
        card.className = 'scatter-card-compact';
        card.innerHTML = `
            <h3>All configs — Compression × vs Recall@${recallKey}
                <span style="float:right;font-weight:400;opacity:0.5">
                    color = bits &nbsp;|&nbsp; ★ = recommended
                </span>
            </h3>
            <svg id="scatter-svg" viewBox="0 0 480 180" aria-label="Scatter plot"></svg>
            <div style="display:flex;gap:1rem;margin-top:0.4rem;flex-wrap:wrap;justify-content:center">
                ${[1,2,3,4].map(b => `<span style="display:flex;align-items:center;gap:0.3rem;font-size:0.7rem;color:var(--text-secondary)"><svg width="10" height="10"><circle cx="5" cy="5" r="4" fill="${BIT_COLORS[b]}"/></svg>${b}-bit</span>`).join('')}
            </div>`;
        resultsPanel.appendChild(card);

        const svg = card.querySelector('#scatter-svg');
        drawScatter(svg, scored, topCfg);
    }

    function drawScatter(svg, scored, topCfg) {
        const PAD = { left: 44, right: 16, top: 12, bottom: 34 };
        const W = 480, H = 180;
        const pw = W - PAD.left - PAD.right;
        const ph = H - PAD.top  - PAD.bottom;

        const xs = scored.map(r => r.rk[recallKey] || 0);
        const ys = scored.map(r => r.compr || 0);

        const xMin = 0, xMax = Math.max(...xs) * 1.05 || 1;
        const yMin = 0, yMax = Math.max(...ys) * 1.1  || 1;

        const px = v => PAD.left + ((v - xMin) / (xMax - xMin)) * pw;
        const py = v => PAD.top  + ph - ((v - yMin) / (yMax - yMin)) * ph;

        let out = '';

        // Grid lines + labels
        for (let i = 0; i <= 5; i++) {
            const x = xMin + (i / 5) * (xMax - xMin);
            const sx = px(x);
            out += `<line x1="${sx}" y1="${PAD.top}" x2="${sx}" y2="${PAD.top + ph}" stroke="rgba(255,255,255,0.05)" stroke-width="1"/>`;
            out += `<text x="${sx}" y="${PAD.top + ph + 14}" text-anchor="middle" font-size="8" fill="rgba(255,255,255,0.35)">${(x * 100).toFixed(0)}%</text>`;
        }
        for (let i = 0; i <= 4; i++) {
            const y = yMin + (i / 4) * (yMax - yMin);
            const sy = py(y);
            out += `<line x1="${PAD.left}" y1="${sy}" x2="${PAD.left + pw}" y2="${sy}" stroke="rgba(255,255,255,0.05)" stroke-width="1"/>`;
            out += `<text x="${PAD.left - 5}" y="${sy + 3}" text-anchor="end" font-size="8" fill="rgba(255,255,255,0.35)">${y.toFixed(1)}×</text>`;
        }

        // Axis labels
        out += `<text x="${PAD.left + pw / 2}" y="${H - 2}" text-anchor="middle" font-size="9" fill="rgba(255,255,255,0.4)">Recall@${recallKey}</text>`;
        out += `<text x="9" y="${PAD.top + ph / 2}" text-anchor="middle" font-size="9" fill="rgba(255,255,255,0.4)" transform="rotate(-90,9,${PAD.top + ph / 2})">Compr ×</text>`;

        // Points
        for (const r of scored.filter(r => r !== topCfg)) {
            const col = BIT_COLORS[r.bits] || '#888';
            out += `<circle cx="${px(r.rk[recallKey] || 0)}" cy="${py(r.compr || 0)}" r="3.5" fill="${col}" opacity="0.4"/>`;
        }
        if (topCfg) {
            const cx = px(topCfg.rk[recallKey] || 0);
            const cy = py(topCfg.compr || 0);
            const col = BIT_COLORS[topCfg.bits] || '#888';
            out += `<circle cx="${cx}" cy="${cy}" r="8" fill="none" stroke="${col}" stroke-width="1.5" opacity="0.4"/>`;
            out += `<circle cx="${cx}" cy="${cy}" r="4.5" fill="${col}"/>`;
            out += `<text x="${cx}" y="${cy - 10}" text-anchor="middle" font-size="9" fill="${col}" font-weight="700">★</text>`;
        }

        svg.innerHTML = out;
    }

    // ── Priority sliders ──────────────────────────────────────────────────────
    function updateWeights() {
        const r = parseFloat(sliderRecall.value);
        const c = parseFloat(sliderCompr.value);
        const s = parseFloat(sliderSpeed.value);
        const total = r + c + s;

        if (total === 0) {
            wRecall = wCompr = wSpeed = 1 / 3;
        } else {
            wRecall = r / total;
            wCompr  = c / total;
            wSpeed  = s / total;
        }

        pctRecall.textContent = Math.round(wRecall * 100) + '%';
        pctCompr.textContent  = Math.round(wCompr  * 100) + '%';
        pctSpeed.textContent  = Math.round(wSpeed  * 100) + '%';
    }

    function applyScenario(key) {
        const s = SCENARIOS[key];
        if (!s) return;

        // Update sliders
        sliderRecall.value = s.r;
        sliderCompr.value  = s.c;
        sliderSpeed.value  = s.s;
        updateWeights();

        // Update top-k if changed
        recallKey = s.rk;
        topkRow.querySelectorAll('.topk-btn').forEach(btn => {
            btn.classList.toggle('active', btn.dataset.rk === recallKey);
        });
    }

    // ── Controls initialization ───────────────────────────────────────────────
    function initControls() {
        // Scenario buttons
        scenarioGrid.querySelectorAll('.scenario-btn').forEach(btn => {
            btn.addEventListener('click', () => {
                scenarioGrid.querySelectorAll('.scenario-btn').forEach(b => b.classList.remove('active'));
                btn.classList.add('active');
                applyScenario(btn.dataset.scenario);
                render();
            });
        });

        // Dimension presets
        dimPresets.querySelectorAll('.preset-btn').forEach(btn => {
            btn.addEventListener('click', () => {
                dimPresets.querySelectorAll('.preset-btn').forEach(b => b.classList.remove('active'));
                btn.classList.add('active');
                userDim = parseInt(btn.dataset.dim, 10);
                dimInput.value = userDim;
                render();
            });
        });

        dimPresets.querySelectorAll('.preset-btn').forEach(btn => {
            if (parseInt(btn.dataset.dim, 10) === userDim) btn.classList.add('active');
        });

        // Dimension free input
        dimInput.addEventListener('input', () => {
            const v = parseInt(dimInput.value, 10);
            if (!isNaN(v) && v > 0) {
                userDim = v;
                dimPresets.querySelectorAll('.preset-btn').forEach(btn => {
                    btn.classList.toggle('active', parseInt(btn.dataset.dim, 10) === userDim);
                });
                render();
            }
        });

        // Top-k buttons
        topkRow.querySelectorAll('.topk-btn').forEach(btn => {
            btn.addEventListener('click', () => {
                topkRow.querySelectorAll('.topk-btn').forEach(b => b.classList.remove('active'));
                btn.classList.add('active');
                recallKey = btn.dataset.rk;
                // Clear active scenario since user is now tuning manually
                scenarioGrid.querySelectorAll('.scenario-btn').forEach(b => b.classList.remove('active'));
                render();
            });
        });

        // Priority sliders — clear active scenario when tuned manually
        [sliderRecall, sliderCompr, sliderSpeed].forEach(sl => {
            sl.addEventListener('input', () => {
                scenarioGrid.querySelectorAll('.scenario-btn').forEach(b => b.classList.remove('active'));
                updateWeights();
                render();
            });
        });

        updateWeights();
    }

    // Intersection observer for fade-in animations
    const observer = new IntersectionObserver(entries => {
        entries.forEach(e => {
            if (e.isIntersecting) { e.target.classList.add('visible'); observer.unobserve(e.target); }
        });
    }, { threshold: 0.1 });
    document.querySelectorAll('.fade-in-up').forEach(el => observer.observe(el));

})();
