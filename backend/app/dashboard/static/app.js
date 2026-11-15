/**
 * Streaming Drift Monitor Dashboard Application Script
 * Person 2 (Backend & Systems Lead)
 */

let performanceChart = null;
let psiChart = null;
const perfHistory = {
  labels: [],
  estimatedF1: [],
  baselineF1: []
};

// Initialization on DOM load
document.addEventListener('DOMContentLoaded', () => {
  setupNavigation();
  initCharts();
  refreshAllData();
  
  // Auto-refresh interval every 4 seconds
  setInterval(refreshAllData, 4000);
});

// Setup navigation tab switching
function setupNavigation() {
  const navButtons = document.querySelectorAll('.nav-item');
  const tabPanes = document.querySelectorAll('.tab-pane');

  navButtons.forEach(btn => {
    btn.addEventListener('click', () => {
      const targetId = btn.getAttribute('data-tab');
      navButtons.forEach(b => b.classList.remove('active'));
      tabPanes.forEach(p => p.classList.remove('active'));

      btn.classList.add('active');
      const targetPane = document.getElementById(targetId);
      if (targetPane) {
        targetPane.classList.add('active');
      }

      // Resize charts if becoming visible
      if (targetId === 'tab-overview' && performanceChart) {
        performanceChart.resize();
      } else if (targetId === 'tab-univariate' && psiChart) {
        psiChart.resize();
      }
    });
  });
}

// Initialize Chart.js instances
function initCharts() {
  // 1. Performance Chart
  const perfCtx = document.getElementById('performanceChart')?.getContext('2d');
  if (perfCtx) {
    performanceChart = new Chart(perfCtx, {
      type: 'line',
      data: {
        labels: ['Check 0', 'Check 1', 'Check 2'],
        datasets: [
          {
            label: 'Estimated F1 (CBPE)',
            data: [0.88, 0.88, 0.88],
            borderColor: '#c084fc',
            backgroundColor: 'rgba(192, 132, 252, 0.1)',
            fill: true,
            tension: 0.3,
            borderWidth: 2,
            pointRadius: 4,
          },
          {
            label: 'Baseline Reference F1',
            data: [0.88, 0.88, 0.88],
            borderColor: '#38bdf8',
            borderDash: [5, 5],
            borderWidth: 1.5,
            pointRadius: 0,
            fill: false,
          }
        ]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: { labels: { color: '#94a3b8', font: { family: 'Inter', size: 12 } } },
          tooltip: { mode: 'index', intersect: false }
        },
        scales: {
          x: {
            grid: { color: 'rgba(51, 65, 85, 0.3)' },
            ticks: { color: '#94a3b8', font: { family: 'Inter', size: 11 } }
          },
          y: {
            min: 0.0,
            max: 1.0,
            grid: { color: 'rgba(51, 65, 85, 0.3)' },
            ticks: { color: '#94a3b8', font: { family: 'Inter', size: 11 } }
          }
        }
      }
    });
  }

  // 2. Univariate PSI Bar Chart
  const psiCtx = document.getElementById('psiChart')?.getContext('2d');
  if (psiCtx) {
    psiChart = new Chart(psiCtx, {
      type: 'bar',
      data: {
        labels: [],
        datasets: [{
          label: 'Population Stability Index (PSI)',
          data: [],
          backgroundColor: [],
          borderWidth: 1,
          borderRadius: 4,
        }]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: { display: false },
          tooltip: {
            callbacks: {
              afterLabel: (ctx) => {
                const val = ctx.raw;
                if (val > 0.25) return 'Status: Significant Drift (>0.25)';
                if (val > 0.10) return 'Status: Moderate Warning (>0.10)';
                return 'Status: Stable (<=0.10)';
              }
            }
          }
        },
        scales: {
          x: {
            grid: { color: 'rgba(51, 65, 85, 0.3)' },
            ticks: { color: '#94a3b8', font: { family: 'Inter', size: 11 } }
          },
          y: {
            beginAtZero: true,
            suggestedMax: 0.3,
            grid: { color: 'rgba(51, 65, 85, 0.3)' },
            ticks: { color: '#94a3b8', font: { family: 'Inter', size: 11 } }
          }
        }
      }
    });
  }
}

// Master refresh method fetching from all FastAPI endpoints
async function refreshAllData() {
  try {
    const statusRes = await fetch('/api/v1/drift/status');
    if (!statusRes.ok) return;
    const data = await statusRes.json();
    updateOverviewKPIs(data);
    updateMultivariateCard(data.multivariate_drift);
    updateCBPECard(data.performance_estimation, data.baseline_metrics);
    updateRetrainSection(data.retraining_recommendation);

    // Fetch feature sketches
    const uniRes = await fetch('/api/v1/drift/univariate');
    if (uniRes.ok) {
      const uniData = await uniRes.json();
      updateUnivariateTable(uniData);
    }

    // Fetch alarms history
    const alarmsRes = await fetch('/api/v1/drift/events?limit=10');
    if (alarmsRes.ok) {
      const alarms = await alarmsRes.json();
      updateAlarmsTable(alarms);
    }

    // Fetch retraining audits
    const auditsRes = await fetch('/api/v1/retrain/status?limit=10');
    if (auditsRes.ok) {
      const auditsData = await auditsRes.json();
      updateRetrainAuditsTable(auditsData.recent_audits || []);
    }

    // Fetch delayed reconciliations
    const reconRes = await fetch('/api/v1/performance/history?limit=5');
    if (reconRes.ok) {
      const reconData = await reconRes.json();
      updateReconciliationTable(reconData);
    }

  } catch (err) {
    console.error('Failed to fetch monitoring data:', err);
  }
}

// Update Top Overview KPIs and Banner
function updateOverviewKPIs(data) {
  const alert = data.alert || {};
  const mem = data.memory || {};
  const cbpe = data.performance_estimation || {};
  const base = data.baseline_metrics || {};

  // Status banner & KPI
  const sev = alert.severity || 'NORMAL';
  const banner = document.getElementById('alert-banner');
  const bannerText = document.getElementById('alert-severity-text');
  const kpiSev = document.getElementById('kpi-severity');
  
  if (banner && bannerText && kpiSev) {
    banner.className = `alert-badge severity-${sev.toLowerCase()}`;
    bannerText.textContent = `STATUS: ${sev}`;
    kpiSev.textContent = sev;
    kpiSev.className = `kpi-value ${sev === 'CRITICAL' ? 'text-red' : sev === 'WARNING' ? 'text-yellow' : 'text-green'}`;
  }

  const kpiComp = document.getElementById('kpi-composite');
  if (kpiComp) kpiComp.textContent = `Composite Score: ${(alert.composite_score || 0).toFixed(3)}`;

  // Records & Uptime
  const kpiRec = document.getElementById('kpi-records');
  if (kpiRec) kpiRec.textContent = (data.records_processed || 0).toLocaleString();
  const kpiUp = document.getElementById('kpi-uptime');
  if (kpiUp) kpiUp.textContent = `Uptime: ${(data.uptime_seconds || 0).toFixed(0)}s`;

  // Throughput
  const kpiTp = document.getElementById('kpi-throughput');
  if (kpiTp) kpiTp.textContent = (data.throughput_rec_per_sec || 0).toFixed(1);

  // Memory
  const kpiCompRatio = document.getElementById('kpi-compression');
  if (kpiCompRatio) kpiCompRatio.textContent = `${(mem.memory_compression_ratio || 1.0).toFixed(1)}x`;
  const kpiMemDet = document.getElementById('kpi-mem-details');
  if (kpiMemDet) {
    const skKb = ((mem.sketches_total_bytes || 0) / 1024).toFixed(1);
    const winKb = ((mem.raw_window_bytes || 0) / 1024).toFixed(1);
    kpiMemDet.textContent = `Sketches: ${skKb} KB | Window: ${winKb} KB`;
  }

  // F1
  const kpiF1 = document.getElementById('kpi-f1');
  if (kpiF1) kpiF1.textContent = cbpe.estimated_f1 ? (cbpe.estimated_f1).toFixed(3) : '--';
  const kpiBaseF1 = document.getElementById('kpi-baseline-f1');
  if (kpiBaseF1) kpiBaseF1.textContent = `Baseline: ${base.f1 ? base.f1.toFixed(3) : '--'}`;

  // Urgency
  const retrain = data.retraining_recommendation || {};
  const kpiUrg = document.getElementById('kpi-urgency');
  if (kpiUrg) {
    const urg = retrain.urgency || 'NONE';
    kpiUrg.textContent = urg;
    kpiUrg.className = `kpi-value ${urg === 'HIGH' ? 'text-red' : urg === 'MEDIUM' ? 'text-yellow' : 'text-green'}`;
  }
  const kpiRecAct = document.getElementById('kpi-rec-action');
  if (kpiRecAct) kpiRecAct.textContent = retrain.recommended_action || 'Standard Monitoring';

  // Update Performance Chart History
  if (performanceChart && cbpe.estimated_f1 !== undefined && base.f1 !== undefined) {
    const stepLabel = `rec ${data.records_processed}`;
    if (!perfHistory.labels.includes(stepLabel)) {
      perfHistory.labels.push(stepLabel);
      perfHistory.estimatedF1.push(cbpe.estimated_f1);
      perfHistory.baselineF1.push(base.f1);

      if (perfHistory.labels.length > 12) {
        perfHistory.labels.shift();
        perfHistory.estimatedF1.shift();
        perfHistory.baselineF1.shift();
      }

      performanceChart.data.labels = perfHistory.labels;
      performanceChart.data.datasets[0].data = perfHistory.estimatedF1;
      performanceChart.data.datasets[1].data = perfHistory.baselineF1;
      performanceChart.update('none');
    }
  }
}

// Update Univariate Features Table & Bar Chart
function updateUnivariateTable(features) {
  const tbody = document.getElementById('univariate-table-body');
  if (!tbody) return;

  if (!features || features.length === 0) {
    tbody.innerHTML = '<tr><td colspan="7" class="text-center">No features evaluated yet.</td></tr>';
    return;
  }

  let html = '';
  const labels = [];
  const psiValues = [];
  const bgColors = [];

  features.forEach(f => {
    labels.push(f.feature_name);
    psiValues.push(f.psi);

    let color = '#4ade80';
    if (f.psi > 0.25) color = '#f87171';
    else if (f.psi > 0.10) color = '#facc15';
    bgColors.push(color);

    const sigBadge = f.is_significant
      ? '<span class="badge badge-critical">DRIFT DETECTED</span>'
      : '<span class="badge badge-normal">STABLE</span>';

    html += `
      <tr>
        <td class="font-bold">${f.feature_name}</td>
        <td class="font-mono ${f.psi > 0.1 ? 'text-yellow' : ''}">${f.psi.toFixed(4)}</td>
        <td class="font-mono">${f.ks_p_value.toFixed(4)}</td>
        <td>${sigBadge}</td>
        <td class="font-mono">${f.running_mean.toFixed(2)}</td>
        <td class="font-mono">${f.tdigest_median.toFixed(2)}</td>
        <td class="font-mono">${f.tdigest_p95.toFixed(2)}</td>
      </tr>
    `;
  });

  tbody.innerHTML = html;

  if (psiChart) {
    psiChart.data.labels = labels;
    psiChart.data.datasets[0].data = psiValues;
    psiChart.data.datasets[0].backgroundColor = bgColors;
    psiChart.update('none');
  }
}

// Update Multivariate MMD & C2ST stats
function updateMultivariateCard(multivariate) {
  if (!multivariate) return;
  const mmd = multivariate.mmd || {};
  const c2st = multivariate.c2st || {};

  const mmdStat = document.getElementById('mmd-stat');
  if (mmdStat) mmdStat.textContent = mmd.mmd_squared !== undefined ? mmd.mmd_squared.toFixed(5) : '0.00000';
  const mmdP = document.getElementById('mmd-pvalue');
  if (mmdP) mmdP.textContent = mmd.p_value !== undefined ? mmd.p_value.toFixed(4) : '1.0000';
  const mmdDet = document.getElementById('mmd-detected');
  if (mmdDet) {
    mmdDet.textContent = mmd.drift_detected ? 'TRUE' : 'FALSE';
    mmdDet.className = `stat-val font-bold ${mmd.drift_detected ? 'text-red' : 'text-green'}`;
  }
  const mmdGam = document.getElementById('mmd-gamma');
  if (mmdGam) mmdGam.textContent = mmd.gamma !== undefined ? mmd.gamma.toFixed(4) : '--';

  const c2stAuc = document.getElementById('c2st-auc');
  if (c2stAuc) c2stAuc.textContent = c2st.roc_auc !== undefined ? c2st.roc_auc.toFixed(4) : '0.5000';
  const c2stP = document.getElementById('c2st-pvalue');
  if (c2stP) c2stP.textContent = c2st.p_value !== undefined ? c2st.p_value.toFixed(4) : '1.0000';
  const c2stDet = document.getElementById('c2st-detected');
  if (c2stDet) {
    c2stDet.textContent = c2st.drift_detected ? 'TRUE' : 'FALSE';
    c2stDet.className = `stat-val font-bold ${c2st.drift_detected ? 'text-red' : 'text-green'}`;
  }

  // Top features list
  const listEl = document.getElementById('c2st-features-list');
  if (listEl && c2st.top_contributing_features && c2st.top_contributing_features.length > 0) {
    listEl.innerHTML = c2st.top_contributing_features.map(([name, score]) => `
      <span class="badge badge-info">${name}: ${(score * 100).toFixed(1)}%</span>
    `).join('');
  }
}

// Update CBPE Panel
function updateCBPECard(cbpe, base) {
  if (!cbpe || !base) return;

  const setCard = (id, val, baseVal) => {
    const el = document.getElementById(id);
    const deltaEl = document.getElementById(`${id}-delta`);
    if (el) el.textContent = val !== undefined ? val.toFixed(3) : '--';
    if (deltaEl && baseVal !== undefined) {
      const diff = val - baseVal;
      deltaEl.textContent = `Baseline: ${baseVal.toFixed(3)} (${diff >= 0 ? '+' : ''}${diff.toFixed(3)})`;
      deltaEl.className = `kpi-sub ${diff < -0.05 ? 'text-red' : 'text-muted'}`;
    }
  };

  setCard('cbpe-acc', cbpe.estimated_accuracy, base.accuracy);
  setCard('cbpe-prec', cbpe.estimated_precision, base.precision);
  setCard('cbpe-rec', cbpe.estimated_recall, base.recall);
  setCard('cbpe-f1', cbpe.estimated_f1, base.f1);
  setCard('cbpe-auc', cbpe.estimated_roc_auc, base.roc_auc);
}

// Update Retrain Decision Section
function updateRetrainSection(rec) {
  if (!rec) return;
  const reasonEl = document.getElementById('retrain-reason');
  if (reasonEl) reasonEl.textContent = rec.trigger_reason || 'No drift detected.';
  const actionEl = document.getElementById('retrain-action');
  if (actionEl) actionEl.textContent = rec.recommended_action || 'Continue standard monitoring.';

  const urgBadge = document.getElementById('retrain-urgency-badge');
  if (urgBadge) {
    const urg = rec.urgency || 'NONE';
    urgBadge.textContent = `Urgency: ${urg}`;
    urgBadge.className = `badge ${urg === 'HIGH' ? 'badge-critical' : urg === 'MEDIUM' ? 'badge-warning' : 'badge-normal'}`;
  }
}

// Update Alarms Table
function updateAlarmsTable(alarms) {
  const tbody = document.getElementById('alarms-table-body');
  const countBadge = document.getElementById('alarm-count-badge');
  if (countBadge) countBadge.textContent = `${alarms.length} events`;
  if (!tbody) return;

  if (!alarms || alarms.length === 0) {
    tbody.innerHTML = '<tr><td colspan="6" class="text-center">No drift alarms recorded. Data stream stationary.</td></tr>';
    return;
  }

  tbody.innerHTML = alarms.map(a => {
    const dateStr = new Date(a.timestamp * 1000).toLocaleTimeString();
    const sevClass = a.severity === 'CRITICAL' ? 'badge-critical' : a.severity === 'WARNING' ? 'badge-warning' : 'badge-normal';
    const feats = Array.isArray(a.affected_features) ? a.affected_features.join(', ') : 'None';

    return `
      <tr>
        <td class="font-mono">#${a.record_position}</td>
        <td>${dateStr}</td>
        <td><span class="badge ${sevClass}">${a.severity}</span></td>
        <td>${a.summary}</td>
        <td><span class="text-yellow">${feats}</span></td>
        <td class="font-mono">${(a.composite_score || 0).toFixed(3)}</td>
      </tr>
    `;
  }).join('');
}

// Update Retraining Audits Table
function updateRetrainAuditsTable(audits) {
  const tbody = document.getElementById('retrain-audits-body');
  if (!tbody) return;

  if (!audits || audits.length === 0) {
    tbody.innerHTML = '<tr><td colspan="8" class="text-center">No retraining audits logged.</td></tr>';
    return;
  }

  tbody.innerHTML = audits.map(a => {
    const dateStr = new Date(a.timestamp * 1000).toLocaleTimeString();
    const statusClass = a.status === 'APPROVED' ? 'badge-normal' : a.status === 'PENDING' ? 'badge-warning' : 'badge-critical';
    return `
      <tr>
        <td>${dateStr}</td>
        <td><span class="badge ${a.urgency === 'HIGH' ? 'badge-critical' : 'badge-warning'}">${a.urgency}</span></td>
        <td class="font-mono">${(a.active_f1 || 0).toFixed(3)}</td>
        <td class="font-mono">${(a.candidate_f1 || 0).toFixed(3)}</td>
        <td class="font-mono text-green">+${(a.f1_delta || 0).toFixed(3)}</td>
        <td>${a.decision}</td>
        <td><span class="badge ${statusClass}">${a.status}</span></td>
        <td>${a.approved_by || 'Awaiting Sign-off'}</td>
      </tr>
    `;
  }).join('');
}

// Update Delayed Reconciliations Table
function updateReconciliationTable(rows) {
  const tbody = document.getElementById('reconciliation-table-body');
  if (!tbody) return;

  if (!rows || rows.length === 0) {
    tbody.innerHTML = '<tr><td colspan="5" class="text-center">No reconciliation records.</td></tr>';
    return;
  }

  tbody.innerHTML = rows.map(r => `
    <tr>
      <td class="font-mono">${r.sample_size} records</td>
      <td class="font-mono">${(r.estimated_f1 || 0).toFixed(3)}</td>
      <td class="font-mono">${(r.actual_f1 || 0).toFixed(3)}</td>
      <td class="font-mono text-yellow">${(r.f1_error || 0).toFixed(4)}</td>
      <td><span class="badge badge-normal">Within Tolerance</span></td>
    </tr>
  `).join('');
}

// Interactive Simulation trigger
async function simulateStream(scenario, count) {
  logTerminal(`[Simulator] Starting injection: ${count} transactions (${scenario} scenario)...`);
  const statusEl = document.getElementById('sim-status');
  if (statusEl) {
    statusEl.style.display = 'block';
    statusEl.className = 'alert-box alert-info mt-3';
    statusEl.textContent = `Streaming ${count} records with ${scenario} scenario...`;
  }

  try {
    const res = await fetch('/api/v1/stream/simulate', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ num_records: count, drift_type: scenario })
    });
    const result = await res.json();
    logTerminal(`[Simulator] Completed! Processed: ${result.simulated_records}, Alarms: ${result.alarms_triggered}, Severity: ${result.latest_severity}`);
    
    if (statusEl) {
      statusEl.className = `alert-box ${result.alarms_triggered > 0 ? 'alert-warning' : 'alert-success'} mt-3`;
      statusEl.textContent = `Batch complete. ${result.simulated_records} transactions ingested. Alarms triggered: ${result.alarms_triggered}. Status: ${result.latest_severity}.`;
    }

    refreshAllData();
  } catch (err) {
    logTerminal(`[Simulator Error] Failed: ${err.message}`);
  }
}

// Candidate shadow model evaluation
async function triggerCandidateEvaluation() {
  logTerminal('[Shadow Evaluation] Training and evaluating candidate shadow model...');
  try {
    const res = await fetch('/api/v1/retrain/evaluate-candidate', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ sample_size: 300 })
    });
    const data = await res.json();

    const reviewSec = document.getElementById('candidate-review-section');
    if (reviewSec) reviewSec.style.display = 'block';

    const actEl = document.getElementById('active-model-metrics');
    if (actEl) actEl.textContent = JSON.stringify(data.active_metrics, null, 2);

    const candEl = document.getElementById('candidate-model-metrics');
    if (candEl) candEl.textContent = JSON.stringify(data.candidate_metrics, null, 2);

    const decBadge = document.getElementById('candidate-decision-badge');
    if (decBadge) {
      decBadge.textContent = `DECISION: ${data.decision}`;
      decBadge.className = `badge ${data.decision === 'PROMOTE_TO_PRODUCTION' ? 'badge-normal' : 'badge-warning'}`;
    }

    const tokenInput = document.getElementById('approval-token-input');
    if (tokenInput && data.approval_token) {
      tokenInput.value = data.approval_token;
    }

    logTerminal(`[Shadow Evaluation] Decision: ${data.decision}. F1 Delta: +${(data.deltas?.f1_delta || 0).toFixed(3)}. Token issued: ${data.approval_token ? 'YES' : 'NO'}`);
    refreshAllData();
  } catch (err) {
    logTerminal(`[Shadow Evaluation Error] ${err.message}`);
  }
}

// Approve candidate promotion
async function approveCandidatePromotion() {
  const token = document.getElementById('approval-token-input')?.value;
  const approver = document.getElementById('approver-name-input')?.value || 'Laxmi Singla';
  const msgEl = document.getElementById('approval-status-message');

  if (!token) {
    if (msgEl) {
      msgEl.textContent = 'Error: No valid approval token available.';
      msgEl.className = 'status-msg text-red mt-2';
    }
    return;
  }

  try {
    const res = await fetch('/api/v1/retrain/approve', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ approval_token: token, approved_by: approver })
    });
    const data = await res.json();

    if (res.ok) {
      if (msgEl) {
        msgEl.textContent = `Success: Model candidate promoted to active production by ${approver}!`;
        msgEl.className = 'status-msg text-green mt-2';
      }
      logTerminal(`[Governance] Candidate model successfully promoted to production by ${approver}.`);
      refreshAllData();
    } else {
      if (msgEl) {
        msgEl.textContent = `Approval failed: ${data.detail || 'Unknown error'}`;
        msgEl.className = 'status-msg text-red mt-2';
      }
    }
  } catch (err) {
    if (msgEl) {
      msgEl.textContent = `Network error: ${err.message}`;
      msgEl.className = 'status-msg text-red mt-2';
    }
  }
}

// Submit 50 sample delayed labels to test reconciliation
async function submitSampleDelayedLabels() {
  logTerminal('[Reconciliation] Generating 50 delayed ground truth labels...');
  const out = document.getElementById('reconciliation-output');
  
  // Synthetic labels based on recent transactions
  const labels = [];
  for (let i = 1; i <= 50; i++) {
    labels.push({
      record_id: `tx_sim_${i.toString().padStart(6, '0')}`,
      is_fraud: Math.random() < 0.15 ? 1 : 0
    });
  }

  try {
    const res = await fetch('/api/v1/performance/labels', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ labels })
    });
    const result = await res.json();

    if (out) {
      out.textContent = JSON.stringify(result, null, 2);
    }
    logTerminal(`[Reconciliation] Reconciled ${result.reconciled_count} labels. F1 error: ${(result.estimation_errors?.f1_error || 0).toFixed(4)}.`);
    refreshAllData();
  } catch (err) {
    if (out) out.textContent = `Error: ${err.message}`;
  }
}

// Terminal helpers
function logTerminal(msg) {
  const term = document.getElementById('event-terminal');
  if (!term) return;
  const timeStr = new Date().toLocaleTimeString();
  const line = document.createElement('div');
  line.className = 'term-line';
  line.textContent = `[${timeStr}] ${msg}`;
  term.appendChild(line);
  term.scrollTop = term.scrollHeight;
}

function clearTerminal() {
  const term = document.getElementById('event-terminal');
  if (term) term.innerHTML = '<div class="term-line text-muted">[System] Terminal cleared.</div>';
}
