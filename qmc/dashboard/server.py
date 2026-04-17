"""
QMC Live Training Dashboard
============================
Real-time training dashboard with Chart.js for quantum vs classical
ML comparison. Runs an HTTP server in a daemon thread and provides
helpers to feed live training metrics from PyTorch and sklearn models.
"""

import json
import time
import threading
import warnings
import webbrowser

import numpy as np

from http.server import HTTPServer, SimpleHTTPRequestHandler

warnings.filterwarnings('ignore')


# Global state for dashboard
TRAINING_STATE = {
    "models": {},
    "current_model": "",
    "status": "initializing",
    "data_info": {},
    "learning_curves": {},
}


def _add_log(msg):
    """Append a timestamped message to the training log."""
    if 'log' not in TRAINING_STATE:
        TRAINING_STATE['log'] = []
    TRAINING_STATE['log'].append(f"[{time.strftime('%H:%M:%S')}] {msg}")
    # Keep last 100 lines
    TRAINING_STATE['log'] = TRAINING_STATE['log'][-100:]


# ============================================================
# HTML DASHBOARD
# ============================================================

DASHBOARD_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>QMC Live Training Dashboard</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { font-family: 'Segoe UI', system-ui, -apple-system, sans-serif; background: #0d1117; color: #c9d1d9; }
  .header { background: linear-gradient(135deg, #1a1b2e 0%, #16213e 100%); padding: 20px 30px; border-bottom: 1px solid #30363d; }
  .header h1 { font-size: 22px; color: #58a6ff; }
  .header .subtitle { color: #8b949e; font-size: 13px; margin-top: 4px; }
  .status-bar { display: flex; gap: 20px; padding: 12px 30px; background: #161b22; border-bottom: 1px solid #21262d; align-items: center; }
  .status-pill { padding: 4px 12px; border-radius: 12px; font-size: 12px; font-weight: 600; }
  .status-running { background: #1f6feb33; color: #58a6ff; border: 1px solid #1f6feb; }
  .status-done { background: #2ea04333; color: #3fb950; border: 1px solid #2ea043; }
  .status-init { background: #d29922; color: #0d1117; }
  .metric-cards { display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 12px; padding: 20px 30px; }
  .metric-card { background: #161b22; border: 1px solid #30363d; border-radius: 8px; padding: 16px; }
  .metric-card .label { font-size: 11px; color: #8b949e; text-transform: uppercase; letter-spacing: 0.5px; }
  .metric-card .value { font-size: 28px; font-weight: 700; margin-top: 4px; }
  .metric-card .value.highlight { color: #58a6ff; }
  .metric-card .value.green { color: #3fb950; }
  .metric-card .value.orange { color: #d29922; }
  .metric-card .value.red { color: #f85149; }
  .charts { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; padding: 0 30px 20px; }
  .chart-container { background: #161b22; border: 1px solid #30363d; border-radius: 8px; padding: 16px; }
  .chart-container h3 { font-size: 14px; color: #c9d1d9; margin-bottom: 12px; }
  .full-width { grid-column: 1 / -1; }
  .results-table { padding: 0 30px 30px; }
  .results-table table { width: 100%; border-collapse: collapse; background: #161b22; border-radius: 8px; overflow: hidden; }
  .results-table th { background: #1c2128; color: #8b949e; padding: 10px 16px; text-align: left; font-size: 12px; text-transform: uppercase; }
  .results-table td { padding: 10px 16px; border-top: 1px solid #21262d; font-size: 14px; }
  .results-table tr:hover { background: #1c2128; }
  .winner { color: #3fb950; font-weight: 700; }
  .log { padding: 0 30px 30px; }
  .log pre { background: #0d1117; border: 1px solid #30363d; border-radius: 8px; padding: 16px; font-size: 12px; max-height: 300px; overflow-y: auto; color: #8b949e; font-family: 'SF Mono', monospace; }
</style>
</head>
<body>
<div class="header">
  <h1>QMC Live Training Dashboard</h1>
  <div class="subtitle">Quantum vs Classical ML Comparison</div>
</div>

<div class="status-bar">
  <span>Status:</span>
  <span id="status-pill" class="status-pill status-init">Initializing</span>
  <span id="current-model" style="color:#58a6ff; font-weight:600;"></span>
  <span style="margin-left:auto; color:#8b949e; font-size:12px;" id="timestamp"></span>
</div>

<div class="metric-cards">
  <div class="metric-card"><div class="label">Best F1 (QML)</div><div class="value highlight" id="best-qml">--</div></div>
  <div class="metric-card"><div class="label">Best F1 (Classical)</div><div class="value green" id="best-classical">--</div></div>
  <div class="metric-card"><div class="label">Models Trained</div><div class="value orange" id="models-done">0</div></div>
  <div class="metric-card"><div class="label">Total Epochs</div><div class="value" id="total-epochs">0</div></div>
  <div class="metric-card"><div class="label">Training Samples</div><div class="value" id="train-samples">--</div></div>
</div>

<div class="charts">
  <div class="chart-container">
    <h3>Training Loss</h3>
    <canvas id="lossChart"></canvas>
  </div>
  <div class="chart-container">
    <h3>Validation F1 Score</h3>
    <canvas id="f1Chart"></canvas>
  </div>
  <div class="chart-container full-width">
    <h3>Learning Curves: F1 vs Training Data Size</h3>
    <canvas id="learningCurveChart"></canvas>
  </div>
</div>

<div class="results-table">
  <table>
    <thead><tr><th>Model</th><th>Type</th><th>Params</th><th>F1 (macro)</th><th>AUC-ROC</th><th>Status</th></tr></thead>
    <tbody id="results-body"></tbody>
  </table>
</div>

<div class="log">
  <h3 style="margin-bottom:8px; font-size:14px;">Training Log</h3>
  <pre id="log-output">Waiting for training to start...</pre>
</div>

<script>
const COLORS = {
  'VQC': '#f85149', 'Tiny MLP': '#58a6ff', 'Small MLP': '#3fb950',
  'SVM': '#d29922', 'RF': '#bc8cff', 'Hybrid': '#f778ba'
};

let lossChart, f1Chart, lcChart;

function initCharts() {
  const lossCtx = document.getElementById('lossChart').getContext('2d');
  lossChart = new Chart(lossCtx, {
    type: 'line',
    data: { labels: [], datasets: [] },
    options: {
      responsive: true, animation: { duration: 300 },
      scales: { x: { title: { display: true, text: 'Epoch', color: '#8b949e' }, ticks: { color: '#8b949e' }, grid: { color: '#21262d' } },
                y: { title: { display: true, text: 'Loss', color: '#8b949e' }, ticks: { color: '#8b949e' }, grid: { color: '#21262d' } } },
      plugins: { legend: { labels: { color: '#c9d1d9' } } }
    }
  });

  const f1Ctx = document.getElementById('f1Chart').getContext('2d');
  f1Chart = new Chart(f1Ctx, {
    type: 'line',
    data: { labels: [], datasets: [] },
    options: {
      responsive: true, animation: { duration: 300 },
      scales: { x: { title: { display: true, text: 'Epoch', color: '#8b949e' }, ticks: { color: '#8b949e' }, grid: { color: '#21262d' } },
                y: { title: { display: true, text: 'F1 (macro)', color: '#8b949e' }, ticks: { color: '#8b949e' }, grid: { color: '#21262d' }, min: 0.3, max: 0.9 } },
      plugins: { legend: { labels: { color: '#c9d1d9' } } }
    }
  });

  const lcCtx = document.getElementById('learningCurveChart').getContext('2d');
  lcChart = new Chart(lcCtx, {
    type: 'line',
    data: { labels: [], datasets: [] },
    options: {
      responsive: true, animation: { duration: 300 },
      scales: { x: { type: 'logarithmic', title: { display: true, text: 'Training Samples', color: '#8b949e' }, ticks: { color: '#8b949e' }, grid: { color: '#21262d' } },
                y: { title: { display: true, text: 'F1 (macro)', color: '#8b949e' }, ticks: { color: '#8b949e' }, grid: { color: '#21262d' }, min: 0.4, max: 0.75 } },
      plugins: { legend: { labels: { color: '#c9d1d9' } } }
    }
  });
}

function getColor(name) {
  for (const [key, color] of Object.entries(COLORS)) {
    if (name.includes(key)) return color;
  }
  return '#8b949e';
}

function updateDashboard(state) {
  // Status
  const pill = document.getElementById('status-pill');
  pill.textContent = state.status;
  pill.className = 'status-pill ' + (state.status === 'done' ? 'status-done' : state.status === 'initializing' ? 'status-init' : 'status-running');
  document.getElementById('current-model').textContent = state.current_model || '';
  document.getElementById('timestamp').textContent = new Date().toLocaleTimeString();

  // Metrics
  let bestQml = 0, bestClassical = 0, modelsDone = 0, totalEpochs = 0;
  const models = state.models || {};
  for (const [name, info] of Object.entries(models)) {
    if (info.final_f1 !== undefined) {
      modelsDone++;
      if (name.includes('VQC') || name.includes('Hybrid') || name.includes('Kernel')) {
        bestQml = Math.max(bestQml, info.final_f1);
      } else {
        bestClassical = Math.max(bestClassical, info.final_f1);
      }
    }
    totalEpochs += (info.epochs || []).length;
  }
  document.getElementById('best-qml').textContent = bestQml > 0 ? bestQml.toFixed(4) : '--';
  document.getElementById('best-classical').textContent = bestClassical > 0 ? bestClassical.toFixed(4) : '--';
  document.getElementById('models-done').textContent = modelsDone;
  document.getElementById('total-epochs').textContent = totalEpochs;
  if (state.data_info) document.getElementById('train-samples').textContent = state.data_info.train_size || '--';

  // Loss chart
  const lossDatasets = [];
  const f1Datasets = [];
  let maxEpoch = 0;
  for (const [name, info] of Object.entries(models)) {
    if (info.losses && info.losses.length > 0) {
      maxEpoch = Math.max(maxEpoch, info.losses.length);
      lossDatasets.push({ label: name, data: info.losses, borderColor: getColor(name), borderWidth: 2, pointRadius: 0, tension: 0.3 });
    }
    if (info.f1s && info.f1s.length > 0) {
      f1Datasets.push({ label: name, data: info.f1s, borderColor: getColor(name), borderWidth: 2, pointRadius: 0, tension: 0.3 });
    }
  }
  if (maxEpoch > 0) {
    lossChart.data.labels = Array.from({length: maxEpoch}, (_, i) => i + 1);
    lossChart.data.datasets = lossDatasets;
    lossChart.update('none');
    f1Chart.data.labels = Array.from({length: maxEpoch}, (_, i) => i + 1);
    f1Chart.data.datasets = f1Datasets;
    f1Chart.update('none');
  }

  // Learning curves
  const lc = state.learning_curves || {};
  if (Object.keys(lc).length > 0) {
    const lcDatasets = [];
    let sizes = [];
    for (const [name, data] of Object.entries(lc)) {
      if (data.sizes && data.means) {
        sizes = data.sizes;
        lcDatasets.push({ label: name, data: data.means, borderColor: getColor(name), borderWidth: 2, pointRadius: 5, tension: 0.3 });
      }
    }
    lcChart.data.labels = sizes;
    lcChart.data.datasets = lcDatasets;
    lcChart.update('none');
  }

  // Results table
  const tbody = document.getElementById('results-body');
  let rows = '';
  const sortedModels = Object.entries(models).sort((a, b) => (b[1].final_f1 || 0) - (a[1].final_f1 || 0));
  const bestF1 = sortedModels.length > 0 ? (sortedModels[0][1].final_f1 || 0) : 0;
  for (const [name, info] of sortedModels) {
    const isWinner = info.final_f1 === bestF1 && bestF1 > 0;
    const type = (name.includes('VQC') || name.includes('Hybrid') || name.includes('Kernel')) ? 'Quantum' : 'Classical';
    rows += '<tr>' +
      '<td' + (isWinner ? ' class="winner"' : '') + '>' + name + '</td>' +
      '<td>' + type + '</td>' +
      '<td>' + (info.params || '--') + '</td>' +
      '<td' + (isWinner ? ' class="winner"' : '') + '>' + (info.final_f1 !== undefined ? info.final_f1.toFixed(4) : '--') + '</td>' +
      '<td>' + (info.final_auc !== undefined ? info.final_auc.toFixed(4) : '--') + '</td>' +
      '<td>' + (info.status || 'pending') + '</td>' +
      '</tr>';
  }
  tbody.innerHTML = rows;

  // Log
  if (state.log) {
    const logEl = document.getElementById('log-output');
    logEl.textContent = state.log.join('\\n');
    logEl.scrollTop = logEl.scrollHeight;
  }
}

initCharts();

// Poll for updates
setInterval(async () => {
  try {
    const resp = await fetch('/state?' + Date.now());
    const state = await resp.json();
    updateDashboard(state);
  } catch(e) {}
}, 1000);
</script>
</body>
</html>"""


class DashboardHandler(SimpleHTTPRequestHandler):
    """HTTP handler that serves the dashboard HTML and JSON state."""

    def do_GET(self):
        if self.path == '/' or self.path == '/index.html':
            self.send_response(200)
            self.send_header('Content-Type', 'text/html')
            self.end_headers()
            self.wfile.write(DASHBOARD_HTML.encode())
        elif self.path.startswith('/state'):
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Cache-Control', 'no-cache')
            self.end_headers()
            self.wfile.write(json.dumps(TRAINING_STATE, default=str).encode())
        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, format, *args):
        pass  # suppress HTTP logs


def start_dashboard_server(port=8501):
    """
    Start the live training dashboard in a daemon thread.

    Parameters
    ----------
    port : int
        HTTP port to serve on (default: 8501).

    Returns
    -------
    HTTPServer
        The running server instance.
    """
    server = HTTPServer(('0.0.0.0', port), DashboardHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    print(f"\n  Dashboard: http://localhost:{port}")
    webbrowser.open(f'http://localhost:{port}')
    return server


# ============================================================
# TRAINING WITH LIVE UPDATES
# ============================================================

def train_model_live(name, model, X_train, y_train, X_val, y_val,
                     X_test, y_test, epochs=100, loss_fn=None,
                     is_binary=True):
    """
    Train a PyTorch model with live dashboard updates.

    Parameters
    ----------
    name : str
        Display name for the model.
    model : torch.nn.Module
        PyTorch model.
    X_train, y_train : array-like
        Training data.
    X_val, y_val : array-like
        Validation data.
    X_test, y_test : array-like
        Test data for final evaluation.
    epochs : int
        Number of training epochs.
    loss_fn : torch.nn.Module, optional
        Loss function (auto-detected if None).
    is_binary : bool
        Whether this is a binary classification task.

    Returns
    -------
    tuple of (model, test_f1)
    """
    import torch
    from sklearn.metrics import f1_score, roc_auc_score

    TRAINING_STATE['current_model'] = name
    TRAINING_STATE['status'] = f'training {name}'
    TRAINING_STATE['models'][name] = {
        'losses': [], 'val_losses': [], 'f1s': [],
        'params': sum(p.numel() for p in model.parameters()),
        'status': 'training',
    }
    _add_log(f"Started training: {name} "
             f"({TRAINING_STATE['models'][name]['params']} params)")

    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    if loss_fn is None:
        loss_fn = (torch.nn.BCEWithLogitsLoss() if is_binary
                   else torch.nn.CrossEntropyLoss())

    X_train_t = torch.FloatTensor(X_train)
    y_train_t = (torch.FloatTensor(y_train) if is_binary
                 else torch.LongTensor(y_train))
    X_val_t = torch.FloatTensor(X_val)
    y_val_t = (torch.FloatTensor(y_val) if is_binary
               else torch.LongTensor(y_val))
    batch_size = min(32, len(X_train))

    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(len(X_train_t))
        epoch_loss = 0
        n_batches = 0

        for i in range(0, len(X_train_t), batch_size):
            X_batch = X_train_t[perm[i:i + batch_size]]
            y_batch = y_train_t[perm[i:i + batch_size]]
            optimizer.zero_grad()
            out = model(X_batch).squeeze()
            if out.dim() == 0:
                out = out.unsqueeze(0)
            loss = loss_fn(out, y_batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1

        avg_loss = epoch_loss / max(n_batches, 1)

        # Validation
        model.eval()
        with torch.no_grad():
            val_out = model(X_val_t).squeeze()
            if val_out.dim() == 0:
                val_out = val_out.unsqueeze(0)
            val_loss = loss_fn(val_out, y_val_t).item()

            if is_binary:
                val_preds = (val_out.numpy() > 0).astype(int)
            else:
                val_preds = val_out.numpy().argmax(axis=1)
            val_f1 = f1_score(
                y_val, val_preds, average='macro', zero_division=0,
            )

        # Update dashboard state
        TRAINING_STATE['models'][name]['losses'].append(round(avg_loss, 5))
        TRAINING_STATE['models'][name]['val_losses'].append(
            round(val_loss, 5))
        TRAINING_STATE['models'][name]['f1s'].append(round(val_f1, 5))

        if (epoch + 1) % 10 == 0 or epoch == 0:
            _add_log(f"  {name} epoch {epoch + 1}/{epochs}: "
                     f"loss={avg_loss:.4f} val_f1={val_f1:.4f}")

    # Final evaluation on test set
    model.eval()
    with torch.no_grad():
        test_out = model(torch.FloatTensor(X_test)).squeeze()
        if is_binary:
            test_preds = (test_out.numpy() > 0).astype(int)
            test_probs = 1 / (1 + np.exp(-test_out.numpy()))
        else:
            test_preds = test_out.numpy().argmax(axis=1)
            test_probs = torch.softmax(test_out, dim=-1).numpy()

    test_f1 = f1_score(y_test, test_preds, average='macro', zero_division=0)
    try:
        if is_binary:
            test_auc = roc_auc_score(y_test, test_probs)
        else:
            test_auc = roc_auc_score(
                y_test, test_probs, multi_class='ovr', average='macro',
            )
    except Exception:
        test_auc = 0.0

    TRAINING_STATE['models'][name]['final_f1'] = round(test_f1, 5)
    TRAINING_STATE['models'][name]['final_auc'] = round(test_auc, 5)
    TRAINING_STATE['models'][name]['status'] = 'done'
    _add_log(f"  {name} DONE: test_F1={test_f1:.4f} AUC={test_auc:.4f}")

    return model, test_f1


def train_sklearn_live(name, clf, X_train, y_train, X_test, y_test):
    """
    Train an sklearn model with dashboard update.

    Parameters
    ----------
    name : str
        Display name for the model.
    clf : sklearn estimator
        Fitted classifier.
    X_train, y_train : array-like
        Training data.
    X_test, y_test : array-like
        Test data.

    Returns
    -------
    tuple of (clf, test_f1)
    """
    from sklearn.metrics import f1_score, roc_auc_score

    TRAINING_STATE['current_model'] = name
    TRAINING_STATE['status'] = f'training {name}'
    _add_log(f"Training: {name}")

    clf.fit(X_train, y_train)
    preds = clf.predict(X_test)
    test_f1 = f1_score(y_test, preds, average='macro', zero_division=0)

    try:
        if hasattr(clf, 'predict_proba'):
            probs = clf.predict_proba(X_test)
            if probs.shape[1] == 2:
                test_auc = roc_auc_score(y_test, probs[:, 1])
            else:
                test_auc = roc_auc_score(
                    y_test, probs, multi_class='ovr', average='macro',
                )
        else:
            test_auc = 0.0
    except Exception:
        test_auc = 0.0

    TRAINING_STATE['models'][name] = {
        'final_f1': round(test_f1, 5),
        'final_auc': round(test_auc, 5),
        'params': 'N/A',
        'status': 'done',
        'losses': [], 'f1s': [],
    }
    _add_log(f"  {name} DONE: test_F1={test_f1:.4f} AUC={test_auc:.4f}")
    return clf, test_f1
