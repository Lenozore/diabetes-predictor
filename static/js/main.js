const form = document.getElementById('predict-form');
const gaugeFill = document.getElementById('gauge-fill');
const gaugeNeedle = document.getElementById('gauge-needle');
const gaugePercent = document.getElementById('gauge-percent');
const gaugeLabel = document.getElementById('gauge-label');
const contribBox = document.getElementById('contributions');
const errorBox = document.getElementById('form-error');

const GAUGE_CIRCUMFERENCE = 283; // matches the semicircle path length approximation

function paintGauge(prob) {
  const offset = GAUGE_CIRCUMFERENCE * (1 - prob);
  gaugeFill.style.strokeDashoffset = offset;
  gaugeFill.style.stroke = prob >= 0.5 ? 'var(--risk)' : 'var(--teal)';
  const angle = -90 + prob * 180; // -90deg (left) to +90deg (right)
  gaugeNeedle.style.transform = `rotate(${angle}deg)`;
  gaugePercent.textContent = `${Math.round(prob * 100)}%`;
}

function renderContributions(contributions) {
  contribBox.innerHTML = '';
  const maxAbs = Math.max(...contributions.map(([, v]) => Math.abs(v)), 0.0001);
  contributions.forEach(([name, value]) => {
    const row = document.createElement('div');
    row.className = 'contrib-row';
    const pct = Math.abs(value) / maxAbs * 100;
    const color = value >= 0 ? 'var(--risk)' : 'var(--teal)';
    row.innerHTML = `
      <span>${name}</span>
      <span class="contrib-bar-track"><span class="contrib-bar-fill" style="width:${pct}%;background:${color}"></span></span>
      <span class="mono-line">${value.toFixed(3)}</span>
    `;
    contribBox.appendChild(row);
  });
}

form.addEventListener('submit', async (e) => {
  e.preventDefault();
  errorBox.hidden = true;
  const payload = Object.fromEntries(new FormData(form).entries());

  try {
    const res = await fetch('/api/predict', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.error || 'Prediction failed');

    paintGauge(data.probability);
    gaugeLabel.textContent = `${data.label} · ${data.model_used} · ${data.latency_ms}ms`;

    const explainRes = await fetch('/api/explain', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    });
    const explainData = await explainRes.json();
    if (explainRes.ok) renderContributions(explainData.contributions);
  } catch (err) {
    errorBox.textContent = err.message;
    errorBox.hidden = false;
  }
});
