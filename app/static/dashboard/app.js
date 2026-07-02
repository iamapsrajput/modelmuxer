const state = {
  costRange: "daily",
  costs: null,
  routing: null,
};

const palette = ["#38bdf8", "#a78bfa", "#34d399", "#fbbf24", "#fb7185", "#22d3ee"];

function $(id) {
  return document.getElementById(id);
}

function formatCurrency(value) {
  return `$${Number(value || 0).toFixed(4)}`;
}

function formatPercent(value) {
  return `${Math.round(Number(value || 0))}%`;
}

function getApiKey() {
  return $("api-key").value.trim();
}

function authHeaders() {
  const apiKey = getApiKey();
  if (!apiKey) {
    throw new Error("Enter an API key to load dashboard data.");
  }
  return {
    Authorization: `Bearer ${apiKey}`,
  };
}

async function fetchJson(path) {
  const response = await fetch(path, { headers: authHeaders() });
  if (!response.ok) {
    const text = await response.text();
    throw new Error(`Request failed (${response.status}): ${text}`);
  }
  return response.json();
}

function showError(message) {
  let banner = document.querySelector(".error-banner");
  if (!banner) {
    banner = document.createElement("div");
    banner.className = "error-banner";
    document.querySelector("main").prepend(banner);
  }
  banner.textContent = message;
}

function clearError() {
  const banner = document.querySelector(".error-banner");
  if (banner) {
    banner.remove();
  }
}

function drawBarChart(canvas, labels, values, valueFormatter = (v) => v) {
  const ctx = canvas.getContext("2d");
  const width = canvas.clientWidth || 320;
  const height = canvas.height;
  canvas.width = width;
  ctx.clearRect(0, 0, width, height);

  if (!labels.length) {
    ctx.fillStyle = "#94a3b8";
    ctx.font = "14px sans-serif";
    ctx.fillText("No data for selected period", 16, 32);
    return;
  }

  const maxValue = Math.max(...values, 0.0001);
  const padding = 24;
  const barGap = 8;
  const barWidth = Math.max(18, (width - padding * 2 - barGap * (labels.length - 1)) / labels.length);

  labels.forEach((label, index) => {
    const value = values[index];
    const barHeight = ((height - 48) * value) / maxValue;
    const x = padding + index * (barWidth + barGap);
    const y = height - 24 - barHeight;
    ctx.fillStyle = palette[index % palette.length];
    ctx.fillRect(x, y, barWidth, barHeight);
    ctx.fillStyle = "#cbd5e1";
    ctx.font = "11px sans-serif";
    ctx.fillText(valueFormatter(value), x, y - 6);
    ctx.fillStyle = "#94a3b8";
    ctx.save();
    ctx.translate(x + barWidth / 2, height - 8);
    ctx.rotate(-0.5);
    ctx.textAlign = "right";
    ctx.fillText(label, 0, 0);
    ctx.restore();
  });
}

function drawPieChart(canvas, labels, values) {
  const ctx = canvas.getContext("2d");
  const width = canvas.clientWidth || 320;
  const height = canvas.height;
  canvas.width = width;
  ctx.clearRect(0, 0, width, height);

  const total = values.reduce((sum, value) => sum + value, 0);
  if (!total) {
    ctx.fillStyle = "#94a3b8";
    ctx.font = "14px sans-serif";
    ctx.fillText("No data for selected period", 16, 32);
    return;
  }

  const centerX = width / 2;
  const centerY = height / 2;
  const radius = Math.min(width, height) / 2 - 24;
  let startAngle = -Math.PI / 2;

  values.forEach((value, index) => {
    const slice = (value / total) * Math.PI * 2;
    ctx.beginPath();
    ctx.moveTo(centerX, centerY);
    ctx.arc(centerX, centerY, radius, startAngle, startAngle + slice);
    ctx.closePath();
    ctx.fillStyle = palette[index % palette.length];
    ctx.fill();
    startAngle += slice;
  });

  labels.forEach((label, index) => {
    ctx.fillStyle = palette[index % palette.length];
    ctx.fillRect(16, 16 + index * 18, 10, 10);
    ctx.fillStyle = "#cbd5e1";
    ctx.font = "12px sans-serif";
    ctx.fillText(`${label}: ${values[index]}`, 32, 25 + index * 18);
  });
}

function renderSummary() {
  const costs = state.costs || {};
  const routing = state.routing || {};
  $("total-cost").textContent = formatCurrency(costs.total_cost);
  $("total-requests").textContent = String(costs.total_requests || routing.total_requests || 0);

  const local = routing.local_vs_cloud?.local || 0;
  const cloud = routing.local_vs_cloud?.cloud || 0;
  const total = local + cloud;
  $("local-share").textContent = total ? formatPercent((local / total) * 100) : "0%";

  const intents = Object.entries(routing.requests_by_intent || {});
  intents.sort((a, b) => b[1] - a[1]);
  $("top-intent").textContent = intents.length ? intents[0][0] : "—";
}

function renderCostChart() {
  const costs = state.costs || {};
  const breakdown =
    state.costRange === "weekly" ? costs.weekly_breakdown || [] : costs.daily_breakdown || [];
  const labels = breakdown.map((entry) => entry.date || entry.week || "");
  const values = breakdown.map((entry) => Number(entry.cost || 0));
  drawBarChart($("cost-chart"), labels, values, formatCurrency);
}

function renderIntentChart() {
  const routing = state.routing || {};
  const entries = Object.entries(routing.requests_by_intent || {}).slice(0, 6);
  drawPieChart(
    $("intent-chart"),
    entries.map(([label]) => label),
    entries.map(([, value]) => value)
  );
}

function renderLocalCloudChart() {
  const routing = state.routing || {};
  const local = routing.local_vs_cloud?.local || 0;
  const cloud = routing.local_vs_cloud?.cloud || 0;
  drawPieChart($("local-cloud-chart"), ["Local (Ollama)", "Cloud"], [local, cloud]);
}

function renderList(containerId, items, formatter) {
  const container = $(containerId);
  container.innerHTML = "";
  if (!items.length) {
    const empty = document.createElement("li");
    empty.textContent = "No data for selected period";
    container.appendChild(empty);
    return;
  }
  items.forEach((item) => {
    const li = document.createElement("li");
    li.innerHTML = formatter(item);
    container.appendChild(li);
  });
}

function renderTopLists() {
  const routing = state.routing || {};
  renderList("top-providers", routing.top_providers || [], (item) =>
    `<span>${item.provider}</span><span>${item.count} req · ${formatCurrency(item.cost)}</span>`
  );
  renderList("top-models", routing.top_models || [], (item) =>
    `<span>${item.model}</span><span>${item.count} req · ${formatCurrency(item.cost)}</span>`
  );
}

function renderRecentRequests() {
  const tbody = $("recent-requests");
  tbody.innerHTML = "";
  const rows = state.costs?.recent_requests || [];
  if (!rows.length) {
    const tr = document.createElement("tr");
    tr.innerHTML = `<td colspan="7">No recent requests</td>`;
    tbody.appendChild(tr);
    return;
  }
  rows.forEach((row) => {
    const tr = document.createElement("tr");
    tr.innerHTML = `
      <td>${row.created_at || "—"}</td>
      <td>${row.provider || "—"}</td>
      <td>${row.model || "—"}</td>
      <td>${row.intent_label || "—"}</td>
      <td>${row.intent_method || "—"}</td>
      <td>${row.routing_rule || "—"}</td>
      <td>${formatCurrency(row.cost)}</td>
    `;
    tbody.appendChild(tr);
  });
}

function renderAll() {
  renderSummary();
  renderCostChart();
  renderIntentChart();
  renderLocalCloudChart();
  renderTopLists();
  renderRecentRequests();
}

async function loadDashboard() {
  clearError();
  const days = Number($("period-select").value || 30);
  try {
    const [costs, routing] = await Promise.all([
      fetchJson(`/v1/analytics/costs?days=${days}`),
      fetchJson(`/v1/analytics/routing?days=${days}`),
    ]);
    state.costs = costs;
    state.routing = routing;
    renderAll();
  } catch (error) {
    showError(error.message);
  }
}

function bindEvents() {
  $("refresh-btn").addEventListener("click", loadDashboard);
  $("period-select").addEventListener("change", loadDashboard);
  $("api-key").addEventListener("change", () => {
    sessionStorage.setItem("modelmuxer_api_key", getApiKey());
  });

  document.querySelectorAll(".toggle").forEach((button) => {
    button.addEventListener("click", () => {
      document.querySelectorAll(".toggle").forEach((node) => node.classList.remove("active"));
      button.classList.add("active");
      state.costRange = button.dataset.range;
      renderCostChart();
    });
  });

  window.addEventListener("resize", renderAll);
}

function init() {
  const savedKey = sessionStorage.getItem("modelmuxer_api_key");
  if (savedKey) {
    $("api-key").value = savedKey;
  }
  bindEvents();
  if (savedKey) {
    loadDashboard();
  }
}

init();
