let priceChart = null;

async function fetchJSON(url) {
  const res = await fetch(url);
  if (!res.ok) {
    const text = await res.text();
    let msg = text;
    try {
      const j = JSON.parse(text);
      msg = j.error || text;
    } catch {}
    throw new Error(msg || `HTTP ${res.status}`);
  }
  return res.json();
}

async function loadSymbol(symbol) {
  const status = document.getElementById("statusMessage");
  status.textContent = `Loading ${symbol}...`;

  try {
    const payload = await fetchJSON(`/api/symbol/${encodeURIComponent(symbol)}`);

    document.getElementById("title").textContent =
      `Stock Dashboard – ${payload.symbol}`;

    renderChart(payload.data);
    renderLast20Table(payload.last20);
    renderToday(payload.today);
    renderPrediction(payload.prediction);

    status.textContent = `Loaded ${payload.symbol}`;
  } catch (err) {
    console.error(err);
    status.textContent = `Error: ${err.message}`;
    clearDisplay();
  }
}

function clearDisplay() {
  if (priceChart) {
    priceChart.destroy();
    priceChart = null;
  }
  document.querySelector("#last20Table tbody").innerHTML = "";
  document.getElementById("todayBox").innerHTML = "";
  document.getElementById("predictionBox").innerHTML = "";
}

function renderChart(data) {
  if (!data || data.length === 0) {
    clearDisplay();
    return;
  }

  const ctx = document.getElementById("priceChart").getContext("2d");
  const labels = data.map(row => row.date);
  const closes = data.map(row => Number(row.close));

  if (priceChart) priceChart.destroy();

  priceChart = new Chart(ctx, {
    type: "line",
    data: {
      labels,
      datasets: [{
        label: "Close",
        data: closes,
        tension: 0.1
      }]
    },
    options: {
      responsive: true,
      interaction: {
        mode: "index",
        intersect: false
      },
      scales: {
        x: {
          ticks: { maxTicksLimit: 10 }
        }
      }
    }
  });
}

function renderLast20Table(rows) {
  const tbody = document.querySelector("#last20Table tbody");
  tbody.innerHTML = "";

  rows.forEach(row => {
    const tr = document.createElement("tr");

    const cells = [
      row.date,
      Number(row.open).toFixed(2),
      Number(row.high).toFixed(2),
      Number(row.low).toFixed(2),
      Number(row.close).toFixed(2),
      row.volume
    ];

    cells.forEach(val => {
      const td = document.createElement("td");
      td.textContent = val;
      tr.appendChild(td);
    });

    tbody.appendChild(tr);
  });
}

function renderToday(row) {
  if (!row) {
    document.getElementById("todayBox").textContent = "No data.";
    return;
  }
  const box = document.getElementById("todayBox");

  box.innerHTML = `
    <strong>Date:</strong> ${row.date}<br>
    <strong>Open:</strong> ${Number(row.open).toFixed(2)}<br>
    <strong>High:</strong> ${Number(row.high).toFixed(2)}<br>
    <strong>Low:</strong> ${Number(row.low).toFixed(2)}<br>
    <strong>Close:</strong> ${Number(row.close).toFixed(2)}<br>
    <strong>Volume:</strong> ${row.volume}
  `;
}

function renderPrediction(pred) {
  const box = document.getElementById("predictionBox");
  if (!pred || pred.next_close == null) {
    box.innerHTML = `<em>Prediction unavailable.</em>`;
    return;
  }

  box.innerHTML = `
    <strong>Next Date:</strong> ${pred.next_date}<br>
    <strong>Predicted Close:</strong> ${Number(pred.next_close).toFixed(2)}
  `;
}

function setupSearch() {
  const input = document.getElementById("symbolInput");
  const button = document.getElementById("searchButton");

  button.addEventListener("click", () => {
    const sym = input.value.trim().toUpperCase();
    if (!sym) return;
    loadSymbol(sym);
  });

  input.addEventListener("keydown", (e) => {
    if (e.key === "Enter") {
      button.click();
    }
  });

  // Optional: load a default symbol on first load
  input.value = "CIPLA";
  loadSymbol("CIPLA");
}

document.addEventListener("DOMContentLoaded", () => {
  setupSearch();
});
