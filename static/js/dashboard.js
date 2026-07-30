const modelNames = Object.keys(allMetrics);
const metricKeys = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc'];
const colors = ['#1F6F64', '#D98E2B', '#C1440E', '#3C5158', '#12463F'];

new Chart(document.getElementById('metrics-chart'), {
  type: 'bar',
  data: {
    labels: modelNames,
    datasets: metricKeys.map((key, i) => ({
      label: key,
      data: modelNames.map((m) => allMetrics[m][key]),
      backgroundColor: colors[i],
    })),
  },
  options: {
    responsive: true,
    scales: { y: { beginAtZero: true, max: 1 } },
    plugins: { legend: { position: 'bottom' } },
  },
});

const shapEntries = Object.entries(shapRanking).sort((a, b) => b[1] - a[1]);
new Chart(document.getElementById('shap-chart'), {
  type: 'bar',
  data: {
    labels: shapEntries.map(([k]) => k),
    datasets: [{
      label: 'Mean |SHAP value|',
      data: shapEntries.map(([, v]) => v),
      backgroundColor: '#1F6F64',
    }],
  },
  options: {
    indexAxis: 'y',
    responsive: true,
    plugins: { legend: { display: false } },
  },
});
