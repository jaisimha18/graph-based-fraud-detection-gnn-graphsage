import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Cell } from 'recharts';
import { TrendingUp, Target, AlertTriangle } from 'lucide-react';
import './ResultsDashboard.css';

const patternData = [
  { name: 'CYCLE', recall: 100.0, color: '#38bdf8' },
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
    label: 'Pattern Recall',
    value: '98.7%',
    subtext: '226 / 229 test patterns detected',
    icon: Target,
    color: 'var(--accent-emerald)',
  },
  {
    label: 'ROC-AUC',
    value: '0.985',
    subtext: 'Global class separation',
    icon: TrendingUp,
    color: 'var(--accent-blue)',
  },
  {
    label: 'PR-AUC',
    value: '0.309',
    subtext: 'High-imbalance precision/recall',
    icon: AlertTriangle,
    color: 'var(--accent-amber)',
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
        <h2 className="section-title">Evaluation Results</h2>
        <p className="section-subtitle">
          Model performance across global metrics and per-pattern-type detection rates on the held-out test set (Days 10–18).
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

        {/* Pattern Recall Chart */}
        <div className="glass-card chart-card">
          <div className="chart-header">
            <h3 className="chart-title">Pattern-Level Recall by Type</h3>
            <span className="badge badge-emerald">370 Total Patterns</span>
          </div>
          <p className="chart-description">
            A laundering pattern is considered <strong>detected</strong> if at least one of its test-set
            edges was predicted as laundering (score &gt; threshold). This measures the model's ability
            to raise alarms for complete multi-hop schemes.
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
