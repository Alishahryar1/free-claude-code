(() => {
  "use strict";

  let api = null;
  let root = null;
  let active = false;
  let pollTimer = null;
  let timeRange = "24h";
  let searchQuery = "";
  let isFetching = false;

  let currentData = {
    summary: null,
    providers: [],
    models: [],
    fallbacks: null,
    timeseries: [],
  };

  function byId(id) {
    return document.getElementById(id);
  }

  function escapeHtml(str) {
    if (!str) return "";
    return String(str)
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;")
      .replace(/'/g, "&#039;");
  }

  function formatNumber(num) {
    if (num === null || num === undefined) return "0";
    return Number(num).toLocaleString();
  }

  function formatCompactNumber(num) {
    if (num === null || num === undefined) return "0";
    const n = Number(num);
    if (n >= 1000000) return (n / 1000000).toFixed(1) + "M";
    if (n >= 1000) return (n / 1000).toFixed(1) + "k";
    return n.toString();
  }

  function formatDuration(ms) {
    if (ms === null || ms === undefined || ms <= 0) return "0ms";
    if (ms < 1000) return `${Math.round(ms)}ms`;
    return `${(ms / 1000).toFixed(2)}s`;
  }

  async function initialize(apiClient) {
    api = apiClient;
    root = byId("usageRoot");
    if (!root) return;
    buildSkeleton();
  }

  function activate() {
    active = true;
    void refresh();
    startPolling();
  }

  function deactivate() {
    active = false;
    stopPolling();
  }

  function startPolling() {
    stopPolling();
    pollTimer = setInterval(() => {
      if (active && !document.hidden) {
        void refresh(true);
      }
    }, 5000);
  }

  function stopPolling() {
    if (pollTimer) {
      clearInterval(pollTimer);
      pollTimer = null;
    }
  }

  async function refresh(isBackground = false) {
    if (!api || isFetching) return;
    isFetching = true;
    const pulseDot = byId("usagePulseDot");
    if (pulseDot) pulseDot.style.opacity = "0.3";

    try {
      const [summary, providersRes, modelsRes, fallbacks, timeseriesRes] =
        await Promise.all([
          api(`/admin/api/usage/summary?time_range=${timeRange}`),
          api(`/admin/api/usage/providers?time_range=${timeRange}`),
          api(`/admin/api/usage/models?time_range=${timeRange}`),
          api(`/admin/api/usage/fallbacks?time_range=${timeRange}`),
          api(`/admin/api/usage/timeseries?time_range=${timeRange}`),
        ]);

      currentData = {
        summary,
        providers: providersRes.providers || [],
        models: modelsRes.models || [],
        fallbacks,
        timeseries: timeseriesRes.points || [],
      };

      renderDashboard();
    } catch (err) {
      if (!isBackground) {
        console.error("Failed to load usage data:", err);
      }
    } finally {
      isFetching = false;
      if (pulseDot) pulseDot.style.opacity = "1";
    }
  }

  function buildSkeleton() {
    root.innerHTML = `
      <div class="usage-toolbar">
        <div class="usage-time-filter" role="group" aria-label="Time range">
          <button type="button" class="usage-time-btn ${timeRange === "1h" ? "active" : ""}" data-range="1h">1h</button>
          <button type="button" class="usage-time-btn ${timeRange === "24h" ? "active" : ""}" data-range="24h">24h</button>
          <button type="button" class="usage-time-btn ${timeRange === "7d" ? "active" : ""}" data-range="7d">7d</button>
          <button type="button" class="usage-time-btn ${timeRange === "30d" ? "active" : ""}" data-range="30d">30d</button>
          <button type="button" class="usage-time-btn ${timeRange === "all" ? "active" : ""}" data-range="all">All</button>
        </div>

        <div class="usage-toolbar-actions">
          <div class="usage-live-indicator">
            <span id="usagePulseDot" class="usage-pulse-dot"></span>
            <span>Live (5s)</span>
          </div>
          <button id="usageRefreshBtn" type="button" class="usage-action-btn">
            <span>Refresh</span>
          </button>
          <button id="usageClearBtn" type="button" class="usage-action-btn danger">
            <span>Clear History</span>
          </button>
        </div>
      </div>

      <!-- KPI Summary Cards -->
      <div id="usageKpiGrid" class="usage-kpi-grid"></div>

      <!-- Charts Section -->
      <div class="usage-charts-grid">
        <div class="usage-chart-card">
          <div class="usage-chart-header">
            <span class="usage-chart-title">Requests & Errors</span>
            <div class="usage-chart-legend">
              <span class="usage-legend-item"><span class="usage-legend-dot" style="background:#10b981;"></span>Requests</span>
              <span class="usage-legend-item"><span class="usage-legend-dot" style="background:#ef4444;"></span>Errors</span>
            </div>
          </div>
          <div id="usageRequestsChart" class="usage-svg-container"></div>
        </div>

        <div class="usage-chart-card">
          <div class="usage-chart-header">
            <span class="usage-chart-title">Token Throughput</span>
            <div class="usage-chart-legend">
              <span class="usage-legend-item"><span class="usage-legend-dot" style="background:#6366f1;"></span>Input</span>
              <span class="usage-legend-item"><span class="usage-legend-dot" style="background:#8b5cf6;"></span>Output</span>
            </div>
          </div>
          <div id="usageTokensChart" class="usage-svg-container"></div>
        </div>
      </div>

      <!-- Provider Quotas & Rate Limits Section -->
      <div class="usage-section-header">
        <div>
          <h3 class="usage-section-title">Providers & Quotas</h3>
          <p class="usage-section-desc">Upstream rate limits, remaining capacities, and reset countdowns</p>
        </div>
      </div>
      <div id="usageProviderGrid" class="usage-provider-grid"></div>

      <!-- Model Performance Breakdown Table -->
      <div class="usage-table-card">
        <div class="usage-table-toolbar">
          <div>
            <strong style="color:var(--text-strong); font-size:15px;">Model Performance Breakdown</strong>
          </div>
          <input id="usageModelSearch" type="search" class="usage-search-input" placeholder="Search models or providers…" />
        </div>
        <div class="usage-table-wrap">
          <table class="usage-table">
            <thead>
              <tr>
                <th>Model</th>
                <th>Provider</th>
                <th>Requests</th>
                <th>Total Tokens</th>
                <th>Input</th>
                <th>Output</th>
                <th>Reasoning</th>
                <th>Cached</th>
                <th>Avg Latency</th>
                <th>Speed</th>
                <th>Error Rate</th>
              </tr>
            </thead>
            <tbody id="usageModelTableBody"></tbody>
          </table>
        </div>
      </div>

      <!-- Fallback Routing & Multi-Provider Analytics -->
      <div class="usage-fallback-grid">
        <div class="usage-fallback-card">
          <strong style="color:var(--text-strong); font-size:15px;">Routing Summary</strong>
          <div id="usageFallbackSummary" style="display:flex; flex-direction:column; gap:12px; margin-top:8px;"></div>
        </div>
        <div class="usage-fallback-card">
          <strong style="color:var(--text-strong); font-size:15px;">Fallback Destinations & Reasons</strong>
          <div id="usageFallbackDestinations" style="display:flex; flex-direction:column; gap:12px; margin-top:8px;"></div>
        </div>
      </div>

      <!-- Confirmation Dialog for Clear Data -->
      <dialog id="usageClearDialog" class="usage-confirm-dialog">
        <h3 style="margin-top:0; color:var(--text-strong);">Clear Usage Data</h3>
        <p style="color:var(--muted); font-size:14px; margin: 12px 0 20px 0;">
          Are you sure you want to purge all telemetry and usage records from local storage? This action cannot be undone.
        </p>
        <div style="display:flex; justify-content:flex-end; gap:10px;">
          <button id="usageCancelClearBtn" type="button" class="usage-action-btn">Cancel</button>
          <button id="usageConfirmClearBtn" type="button" class="usage-action-btn danger">Purge Records</button>
        </div>
      </dialog>
    `;

    // Event listeners
    root.querySelectorAll(".usage-time-btn").forEach((btn) => {
      btn.addEventListener("click", () => {
        root.querySelectorAll(".usage-time-btn").forEach((b) => b.classList.remove("active"));
        btn.classList.add("active");
        timeRange = btn.dataset.range;
        void refresh();
      });
    });

    byId("usageRefreshBtn")?.addEventListener("click", () => void refresh());

    const searchInput = byId("usageModelSearch");
    searchInput?.addEventListener("input", (e) => {
      searchQuery = e.target.value.toLowerCase().trim();
      renderModelTable();
    });

    const clearBtn = byId("usageClearBtn");
    const clearDialog = byId("usageClearDialog");
    const cancelClearBtn = byId("usageCancelClearBtn");
    const confirmClearBtn = byId("usageConfirmClearBtn");

    clearBtn?.addEventListener("click", () => clearDialog?.showModal());
    cancelClearBtn?.addEventListener("click", () => clearDialog?.close());
    confirmClearBtn?.addEventListener("click", async () => {
      try {
        confirmClearBtn.disabled = true;
        confirmClearBtn.textContent = "Clearing…";
        await api("/admin/api/usage/clear", { method: "POST" });
        clearDialog?.close();
        void refresh();
      } catch (err) {
        alert("Failed to clear usage: " + err.message);
      } finally {
        confirmClearBtn.disabled = false;
        confirmClearBtn.textContent = "Purge Records";
      }
    });
  }

  function renderDashboard() {
    renderKPIs();
    renderCharts();
    renderProviders();
    renderModelTable();
    renderFallbackAnalytics();
  }

  function renderKPIs() {
    const kpiGrid = byId("usageKpiGrid");
    if (!kpiGrid) return;
    const s = currentData.summary || {};

    const totalReq = s.total_requests || 0;
    const successReq = s.successful_requests || 0;
    const failedReq = s.failed_requests || 0;
    const errorRate = totalReq > 0 ? ((failedReq / totalReq) * 100).toFixed(1) : "0.0";

    const totalTok = s.total_tokens || 0;
    const inTok = s.input_tokens || 0;
    const outTok = s.output_tokens || 0;
    const reasonTok = s.reasoning_tokens || 0;
    const cacheTok = s.cached_tokens || 0;

    const avgDur = s.avg_duration_ms || 0;
    const avgTtft = s.avg_ttft_ms;
    const fallbackCount = s.fallback_count || 0;
    const fallbackPct = totalReq > 0 ? ((fallbackCount / totalReq) * 100).toFixed(1) : "0.0";

    kpiGrid.innerHTML = `
      <div class="usage-kpi-card">
        <span class="usage-kpi-label">Total Requests</span>
        <span class="usage-kpi-value">${formatNumber(totalReq)}</span>
        <div class="usage-kpi-subtext">
          <span class="usage-badge success">${formatNumber(successReq)} ok</span>
          ${failedReq > 0 ? `<span class="usage-badge warn">${formatNumber(failedReq)} err (${errorRate}%)</span>` : `<span class="usage-badge neutral">0 err</span>`}
        </div>
      </div>

      <div class="usage-kpi-card">
        <span class="usage-kpi-label">Total Tokens</span>
        <span class="usage-kpi-value">${formatCompactNumber(totalTok)}</span>
        <div class="usage-kpi-subtext">
          <span class="usage-badge info">In: ${formatCompactNumber(inTok)}</span>
          <span class="usage-badge neutral">Out: ${formatCompactNumber(outTok)}</span>
          ${reasonTok > 0 ? `<span class="usage-badge neutral">Reason: ${formatCompactNumber(reasonTok)}</span>` : ""}
          ${cacheTok > 0 ? `<span class="usage-badge success">Cache: ${formatCompactNumber(cacheTok)}</span>` : ""}
        </div>
      </div>

      <div class="usage-kpi-card">
        <span class="usage-kpi-label">Avg Latency & TTFT</span>
        <span class="usage-kpi-value">${formatDuration(avgDur)}</span>
        <div class="usage-kpi-subtext">
          <span>TTFT: <strong>${avgTtft ? formatDuration(avgTtft) : "N/A"}</strong></span>
        </div>
      </div>

      <div class="usage-kpi-card">
        <span class="usage-kpi-label">Fallback Routing</span>
        <span class="usage-kpi-value">${formatNumber(fallbackCount)}</span>
        <div class="usage-kpi-subtext">
          <span>${fallbackPct}% of total requests</span>
          <span class="usage-badge neutral">Auto-recovered</span>
        </div>
      </div>
    `;
  }

  function renderCharts() {
    const points = currentData.timeseries || [];
    renderSvgLineChart("usageRequestsChart", points, [
      { key: "requests", color: "#10b981", label: "Requests" },
      { key: "errors", color: "#ef4444", label: "Errors" },
    ]);

    renderSvgLineChart("usageTokensChart", points, [
      { key: "input_tokens", color: "#6366f1", label: "Input" },
      { key: "output_tokens", color: "#8b5cf6", label: "Output" },
    ]);
  }

  function renderSvgLineChart(containerId, points, series) {
    const container = byId(containerId);
    if (!container) return;

    if (!points || points.length === 0) {
      container.innerHTML = `
        <div style="height:100%; display:flex; align-items:center; justify-content:center; color:var(--muted); font-size:13px;">
          No activity in selected time horizon.
        </div>`;
      return;
    }

    const width = container.clientWidth || 500;
    const height = container.clientHeight || 220;
    const padding = { top: 20, right: 20, bottom: 30, left: 45 };
    const chartW = width - padding.left - padding.right;
    const chartH = height - padding.top - padding.bottom;

    let maxY = 1;
    points.forEach((pt) => {
      series.forEach((s) => {
        const val = pt[s.key] || 0;
        if (val > maxY) maxY = val;
      });
    });
    maxY = Math.ceil(maxY * 1.15); // Add headroom

    const stepX = chartW / Math.max(points.length - 1, 1);

    // SVG elements
    let svgHtml = `
      <svg class="usage-svg-chart" viewBox="0 0 ${width} ${height}" preserveAspectRatio="none">
        <defs>
          <linearGradient id="grad-${containerId}-0" x1="0" y1="0" x2="0" y2="1">
            <stop offset="0%" stop-color="${series[0].color}" stop-opacity="0.25"/>
            <stop offset="100%" stop-color="${series[0].color}" stop-opacity="0.0"/>
          </linearGradient>
        </defs>

        <!-- Horizontal Grid Lines -->
        <line x1="${padding.left}" y1="${padding.top}" x2="${width - padding.right}" y2="${padding.top}" stroke="var(--line)" stroke-width="1"/>
        <line x1="${padding.left}" y1="${padding.top + chartH * 0.5}" x2="${width - padding.right}" y2="${padding.top + chartH * 0.5}" stroke="var(--line)" stroke-width="1"/>
        <line x1="${padding.left}" y1="${height - padding.bottom}" x2="${width - padding.right}" y2="${height - padding.bottom}" stroke="var(--line-strong)" stroke-width="1"/>

        <!-- Y Axis Labels -->
        <text x="${padding.left - 8}" y="${padding.top + 4}" fill="var(--muted)" font-size="10" text-anchor="end">${formatCompactNumber(maxY)}</text>
        <text x="${padding.left - 8}" y="${padding.top + chartH * 0.5 + 4}" fill="var(--muted)" font-size="10" text-anchor="end">${formatCompactNumber(Math.round(maxY * 0.5))}</text>
        <text x="${padding.left - 8}" y="${height - padding.bottom + 4}" fill="var(--muted)" font-size="10" text-anchor="end">0</text>
    `;

    // Draw lines
    series.forEach((s, sIndex) => {
      let dPath = "";
      let areaPath = "";

      points.forEach((pt, i) => {
        const x = padding.left + i * stepX;
        const val = pt[s.key] || 0;
        const y = padding.top + chartH - (val / maxY) * chartH;

        if (i === 0) {
          dPath += `M ${x},${y} `;
          areaPath += `M ${x},${height - padding.bottom} L ${x},${y} `;
        } else {
          dPath += `L ${x},${y} `;
          areaPath += `L ${x},${y} `;
        }

        if (i === points.length - 1) {
          areaPath += `L ${x},${height - padding.bottom} Z`;
        }
      });

      if (sIndex === 0) {
        svgHtml += `<path d="${areaPath}" fill="url(#grad-${containerId}-0)" />`;
      }
      svgHtml += `<path d="${dPath}" fill="none" stroke="${s.color}" stroke-width="2" stroke-linejoin="round" stroke-linecap="round" />`;

      // Draw dots
      points.forEach((pt, i) => {
        const x = padding.left + i * stepX;
        const val = pt[s.key] || 0;
        const y = padding.top + chartH - (val / maxY) * chartH;
        svgHtml += `<circle cx="${x}" cy="${y}" r="3" fill="${s.color}" />`;
      });
    });

    // X Axis Labels (first, middle, last)
    if (points.length > 0) {
      const firstLabel = points[0].label || "";
      const lastLabel = points[points.length - 1].label || "";
      svgHtml += `
        <text x="${padding.left}" y="${height - 10}" fill="var(--muted)" font-size="10" text-anchor="start">${firstLabel}</text>
        <text x="${width - padding.right}" y="${height - 10}" fill="var(--muted)" font-size="10" text-anchor="end">${lastLabel}</text>
      `;
    }

    svgHtml += `</svg>`;
    container.innerHTML = svgHtml;
  }

  function renderProviders() {
    const grid = byId("usageProviderGrid");
    if (!grid) return;
    const providers = currentData.providers || [];

    if (providers.length === 0) {
      grid.innerHTML = `
        <div style="grid-column: 1 / -1; padding: 24px; text-align: center; color: var(--muted); background: var(--card); border: 1px solid var(--line); border-radius: var(--radius-lg);">
          No provider traffic recorded yet. Send requests via Claude Code CLI or Codex to see live metrics.
        </div>`;
      return;
    }

    grid.innerHTML = providers
      .map((p) => {
        const q = p.quota;
        let quotaContent = "";

        if (p.is_local) {
          quotaContent = `
            <div class="usage-no-quota">
              <span>Local / none</span>
              <div style="font-size:11px; margin-top:2px; opacity:0.8;">No cloud rate limits apply</div>
            </div>`;
        } else if (!q || !q.quota_available) {
          quotaContent = `
            <div class="usage-no-quota">
              <span>Provider quota unavailable</span>
              <div style="font-size:11px; margin-top:2px; opacity:0.8;">Upstream headers did not report limits</div>
            </div>`;
        } else {
          // Requests quota bar
          let reqBar = "";
          if (q.requests_limit && q.requests_remaining !== null && q.requests_remaining !== undefined) {
            const used = Math.max(q.requests_limit - q.requests_remaining, 0);
            const pct = Math.min(Math.round((used / q.requests_limit) * 100), 100);
            const statusClass = pct > 90 ? "danger" : pct > 70 ? "warn" : "";
            reqBar = `
              <div class="usage-quota-item">
                <div class="usage-quota-label-row">
                  <span class="usage-quota-label">Requests Quota</span>
                  <span class="usage-quota-count">${formatNumber(q.requests_remaining)} / ${formatNumber(q.requests_limit)} left</span>
                </div>
                <div class="usage-progress-track">
                  <div class="usage-progress-fill ${statusClass}" style="width: ${pct}%"></div>
                </div>
                ${q.requests_reset ? `<div class="usage-quota-reset">Reset: ${escapeHtml(q.requests_reset)}</div>` : ""}
              </div>`;
          }

          // Tokens quota bar
          let tokBar = "";
          if (q.tokens_limit && q.tokens_remaining !== null && q.tokens_remaining !== undefined) {
            const used = Math.max(q.tokens_limit - q.tokens_remaining, 0);
            const pct = Math.min(Math.round((used / q.tokens_limit) * 100), 100);
            const statusClass = pct > 90 ? "danger" : pct > 70 ? "warn" : "";
            tokBar = `
              <div class="usage-quota-item">
                <div class="usage-quota-label-row">
                  <span class="usage-quota-label">Tokens Quota</span>
                  <span class="usage-quota-count">${formatCompactNumber(q.tokens_remaining)} / ${formatCompactNumber(q.tokens_limit)} left</span>
                </div>
                <div class="usage-progress-track">
                  <div class="usage-progress-fill ${statusClass}" style="width: ${pct}%"></div>
                </div>
                ${q.tokens_reset ? `<div class="usage-quota-reset">Reset: ${escapeHtml(q.tokens_reset)}</div>` : ""}
              </div>`;
          }

          quotaContent = `
            <div class="usage-provider-quota-box">
              ${reqBar}
              ${tokBar}
              ${!reqBar && !tokBar ? `<div class="usage-no-quota">${escapeHtml(q.status_label || "Quota active")}</div>` : ""}
            </div>`;
        }

        const totalReq = p.requests || 0;
        const speedText = p.avg_speed_tok_s ? `${p.avg_speed_tok_s.toFixed(1)} tok/s` : "—";
        const errClass = p.failure_count > 0 ? "warn" : "success";

        return `
          <div class="usage-provider-card">
            <div class="usage-provider-head">
              <div>
                <div class="usage-provider-name">${escapeHtml(p.display_name || p.provider_id)}</div>
                <div style="font-size:11px; color:var(--muted);">${escapeHtml(p.provider_id)}</div>
              </div>
              <span class="usage-badge ${p.is_local ? "info" : "success"}">${p.is_local ? "Local" : "Cloud"}</span>
            </div>

            <div class="usage-provider-stats-strip">
              <div><strong>${formatNumber(totalReq)}</strong> reqs</div>
              <div><strong>${formatCompactNumber(p.total_tokens || 0)}</strong> toks</div>
              <div><strong>${speedText}</strong></div>
              <div class="usage-badge ${errClass}" style="margin-left:auto;">${(p.error_rate <= 1 ? p.error_rate * 100 : p.error_rate).toFixed(1)}% err</div>
            </div>

            ${quotaContent}
          </div>
        `;
      })
      .join("");
  }

  function renderModelTable() {
    const tbody = byId("usageModelTableBody");
    if (!tbody) return;
    let models = currentData.models || [];

    if (searchQuery) {
      models = models.filter(
        (m) =>
          m.model_ref.toLowerCase().includes(searchQuery) ||
          m.provider_id.toLowerCase().includes(searchQuery) ||
          m.provider_model.toLowerCase().includes(searchQuery),
      );
    }

    if (models.length === 0) {
      tbody.innerHTML = `
        <tr>
          <td colspan="11" style="text-align:center; padding: 24px; color:var(--muted);">
            No matching model activity found.
          </td>
        </tr>`;
      return;
    }

    tbody.innerHTML = models
      .map((m) => {
        const speedText = m.tokens_per_second ? `${m.tokens_per_second.toFixed(1)}` : "—";
        const errPct = (m.error_rate <= 1 ? m.error_rate * 100 : m.error_rate).toFixed(1);
        const errorBadge =
          m.error_count > 0
            ? `<span class="usage-badge warn">${errPct}% (${m.error_count})</span>`
            : `<span class="usage-badge success">0%</span>`;

        return `
          <tr>
            <td><span class="usage-model-tag">${m.model_ref}</span></td>
            <td><span class="usage-badge neutral">${m.provider_id}</span></td>
            <td><strong>${formatNumber(m.requests)}</strong></td>
            <td><strong>${formatNumber(m.total_tokens)}</strong></td>
            <td>${formatNumber(m.input_tokens)}</td>
            <td>${formatNumber(m.output_tokens)}</td>
            <td>${m.reasoning_tokens ? formatNumber(m.reasoning_tokens) : "—"}</td>
            <td>${m.cached_tokens ? formatNumber(m.cached_tokens) : "—"}</td>
            <td>${formatDuration(m.avg_duration_ms)}</td>
            <td>${speedText}</td>
            <td>${errorBadge}</td>
          </tr>
        `;
      })
      .join("");
  }

  function renderFallbackAnalytics() {
    const summaryContainer = byId("usageFallbackSummary");
    const destContainer = byId("usageFallbackDestinations");
    if (!summaryContainer || !destContainer) return;

    const fb = currentData.fallbacks || {
      primary_requests: 0,
      primary_successes: 0,
      primary_fallbacks: 0,
      fallback_success_rate: 1.0,
      destinations: [],
      reasons: [],
    };

    const fbSuccPct = (fb.fallback_success_rate <= 1 ? fb.fallback_success_rate * 100 : fb.fallback_success_rate).toFixed(1);

    summaryContainer.innerHTML = `
      <div class="usage-dest-item">
        <span>Primary Direct Deliveries</span>
        <strong>${formatNumber(fb.primary_successes)}</strong>
      </div>
      <div class="usage-dest-item">
        <span>Primary Failures / Fallbacks</span>
        <strong style="color:var(--warn);">${formatNumber(fb.primary_fallbacks)}</strong>
      </div>
      <div class="usage-dest-item">
        <span>Fallback Resolution Success</span>
        <strong style="color:var(--ok);">${fbSuccPct}%</strong>
      </div>
    `;

    const dests = fb.destinations || [];
    const reasons = fb.reasons || [];

    let destHtml = "";
    if (dests.length === 0) {
      destHtml = `<div style="color:var(--muted); font-size:13px;">No fallback transitions recorded yet.</div>`;
    } else {
      destHtml = `
        <div class="usage-dest-list">
          ${dests
            .map(
              (d) => `
            <div class="usage-dest-item">
              <div>
                <span class="usage-model-tag">${d.target_ref}</span>
                <div style="font-size:11px; color:var(--muted); margin-top:2px;">
                  Latency: ${formatDuration(d.avg_duration_ms)}
                </div>
              </div>
              <div>
                <span class="usage-badge success">${formatNumber(d.succeeded_count)} / ${formatNumber(d.triggered_count)} ok</span>
              </div>
            </div>
          `,
            )
            .join("")}
        </div>`;
    }

    let reasonsHtml = "";
    if (reasons.length > 0) {
      reasonsHtml = `
        <div style="margin-top:12px;">
          <div style="font-size:12px; color:var(--muted); margin-bottom:6px;">Trigger Reasons:</div>
          <div class="usage-reasons-wrap">
            ${reasons
              .map(
                ([reason, count]) => `
              <span class="usage-reason-chip">
                <span>${reason}</span>
                <span class="usage-badge neutral">${count}</span>
              </span>
            `,
              )
              .join("")}
          </div>
        </div>`;
    }

    destContainer.innerHTML = destHtml + reasonsHtml;
  }

  // Export to window
  window.UsageDashboard = {
    initialize,
    activate,
    deactivate,
    refresh,
  };
})();
