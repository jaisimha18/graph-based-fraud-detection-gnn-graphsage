import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Cell } from 'recharts';
import { TrendingUp, Target, AlertTriangle, Repeat, ShieldCheck, ArrowUpRight } from 'lucide-react';
import './ResultsDashboard.css';

const patternData = [
  { name: 'CYCLE', recall: 100.0, color: '#c084fc' },
  { name: 'GATHER-SCATTER', recall: 100.0, color: '#fb7185' },
  { name: 'SCATTER-GATHER', recall: 100.0, color: '#fbbf24' },
  { name: 'FAN-IN', recall: 100.0, color: '#34d399' },
  { name: 'RANDOM', recall: 100.0, color: '#64748b' },
  { name: 'FAN-OUT', recall: 96.4, color: '#a78bfa' },
  { name: 'STACK', recall: 96.4, color: '#f97316' },
  { name: 'BIPARTITE', recall: 95.7, color: '#22d3ee' },
];

const metrics = [
  {
    label: 'PR-AUC (Key Metric)',
    value: '0.5759',
    subtext: '+78.5% over vanilla GraphSAGE',
    icon: AlertTriangle,
    color: '#38bdf8',
  },
  {
    label: 'Full Ring Recall',
    value: '85.2%',
    subtext: 'Complete closed loop detection',
    icon: Repeat,
    color: '#c084fc',
  },
  {
    label: 'Alert Recall',
    value: '98.7%',
    subtext: '226 / 229 schemes flagged',
    icon: Target,
    color: '#34d399',
  },
  {
    label: 'Global ROC-AUC',
    value: '0.9889',
    subtext: 'High class separability',
    icon: TrendingUp,
    color: '#f59e0b',
  },
];

const ablationRows = [
  {
    name: '1. Vanilla GraphSAGE',
    type: 'Baseline',
    rocAuc: '0.9853',
    prAuc: '0.3227',
    fraudRec: '87.5%',
    alertRec: '99.1%',
    fullRing: '79.0%',
    badge: 'Baseline',
    badgeClass: 'badge-blue',
  },
  {
    name: '2. + RW Cycle Features (Naive)',
    type: 'Dilution Failure',
    rocAuc: '0.9704',
    prAuc: '0.1113',
    fraudRec: '75.3%',
    alertRec: '96.1%',
    fullRing: '63.8%',
    badge: '-65.6% Drop',
    badgeClass: 'badge-amber',
  },
  {
    name: '3. + Cycle Attention (Proposed)',
    type: 'Full Architecture',
    rocAuc: '0.9889',
    prAuc: '0.5759',
    fraudRec: '87.7%',
    alertRec: '98.7%',
    fullRing: '85.2%',
    badge: '+78.5% PR-AUC 🏆',
    badgeClass: 'badge-emerald',
    isWinner: true,
  },
];

const CustomTooltip = ({ active, payload }) => {
  if (active && payload && payload.length) {
    return (
      <div className="chart-tooltip">
        <p className="chart-tooltip-label">{payload[0].payload.name}</p>
        <p className="chart-tooltip-value">{payload[0].value}% recall</p>
      </div>
    );
  }
  return null;
};

export default function ResultsDashboard() {
  return (
    <section className="section results-section" id="results">
      <div className="container">
        <h2 className="section-title">Evaluation & Ablation Results</h2>
        <p className="section-subtitle">
          Rigorous benchmarking across global metrics, 3-stage ablation studies, and per-pattern typology detection on the test set (Days 10–18).
        </p>

        {/* Metric Cards */}
        <div className="metrics-row">
          {metrics.map(({ label, value, subtext, icon: Icon, color }) => (
            <div key={label} className="glass-card metric-card">
              <div className="metric-icon" style={{ color, borderColor: `${color}33` }}>
                <Icon size={22} />
              </div>
              <div className="metric-value" style={{ color }}>{value}</div>
              <div className="metric-label">{label}</div>
              <div className="metric-subtext">{subtext}</div>
            </div>
          ))}
        </div>

        {/* Ablation Comparison Table */}
        <div className="glass-card ablation-card">
          <div className="chart-header">
            <div>
              <h3 className="chart-title">3-Stage Scientific Ablation Benchmark</h3>
              <p className="ablation-subtitle">
                Demonstrating why naive cycle concatenation fails and how learned cycle attention unlocks an 85.2% full ring recall.
              </p>
            </div>
            <span className="badge badge-purple">Cycle-Aware Benchmark</span>
          </div>

          <div className="table-responsive">
            <table className="ablation-table">
              <thead>
                <tr>
                  <th>Model Configuration</th>
                  <th>ROC-AUC</th>
                  <th>PR-AUC</th>
                  <th>Fraud Recall</th>
                  <th>Alert Recall</th>
                  <th>Full Ring Recall</th>
                  <th>Effect</th>
                </tr>
              </thead>
              <tbody>
                {ablationRows.map((row) => (
                  <tr key={row.name} className={row.isWinner ? 'winner-row' : ''}>
                    <td>
                      <div className="model-name-cell">
                        <strong>{row.name}</strong>
                        <span className="model-type-sub">{row.type}</span>
                      </div>
                    </td>
                    <td>{row.rocAuc}</td>
                    <td className="highlight-cell">{row.prAuc}</td>
                    <td>{row.fraudRec}</td>
                    <td>{row.alertRec}</td>
                    <td className="highlight-cell text-purple">{row.fullRing}</td>
                    <td>
                      <span className={`badge ${row.badgeClass}`}>{row.badge}</span>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          <div className="ablation-insight-box">
            <div className="insight-bullet">
              <span className="bullet-title">⚠️ The Naive Addition Dilution Trap (-65.6% PR-AUC):</span>
              <p>
                Simply concatenating cycle return probabilities with standard uniform mean aggregation dilutes the sparse cycle signal across 99% legitimate neighbors, resulting in severe false positives.
              </p>
            </div>
            <div className="insight-bullet">
              <span className="bullet-title">🏆 The Power of Structural Attention (+78.5% PR-AUC, 85.2% Full Ring):</span>
              <p>
                Our <code>CycleAwareSAGEConv</code> dynamically weights neighbor messages via an edge-conditioned attention mechanism. When a transfer belongs to a circular laundering ring, attention sharply focuses on the anomalous path, shielding innocent accounts and cutting compliance noise by 32%.
              </p>
            </div>
          </div>
        </div>

        {/* Pattern Recall Chart */}
        <div className="glass-card chart-card">
          <div className="chart-header">
            <h3 className="chart-title">Pattern-Level Recall by Laundering Typology</h3>
            <span className="badge badge-emerald">370 Total Patterns</span>
          </div>
          <p className="chart-description">
            Detection recall across canonical AML typologies defined in <code>HI-Small_Patterns.txt</code>. A laundering scheme is considered detected if the model raises alerts on its constituent transfers.
          </p>

          <div className="chart-container">
            <ResponsiveContainer width="100%" height={360}>
              <BarChart data={patternData} margin={{ top: 20, right: 30, left: 0, bottom: 5 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" />
                <XAxis
                  dataKey="name"
                  tick={{ fill: '#94a3b8', fontSize: 11, fontFamily: 'JetBrains Mono' }}
                  axisLine={{ stroke: 'rgba(255,255,255,0.08)' }}
                  tickLine={false}
                />
                <YAxis
                  domain={[0, 100]}
                  tick={{ fill: '#94a3b8', fontSize: 11 }}
                  axisLine={{ stroke: 'rgba(255,255,255,0.08)' }}
                  tickLine={false}
                  unit="%"
                />
                <Tooltip content={<CustomTooltip />} cursor={{ fill: 'rgba(255,255,255,0.03)' }} />
                <Bar dataKey="recall" radius={[6, 6, 0, 0]} maxBarSize={50}>
                  {patternData.map((entry, index) => (
                    <Cell key={`cell-${index}`} fill={entry.color} fillOpacity={0.85} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>
      </div>
    </section>
  );
}