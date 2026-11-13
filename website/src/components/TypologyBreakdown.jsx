import React from 'react';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Cell } from 'recharts';
import { Network, Repeat, ArrowRight, ArrowDownRight, Layers, Shuffle, GitMerge, GitFork } from 'lucide-react';
import './TypologyBreakdown.css';

const typologyData = [
  { name: 'CYCLE', recall: 100.0, color: '#c084fc', count: '54 patterns' },
  { name: 'GATHER-SCATTER', recall: 100.0, color: '#fb7185', count: '48 patterns' },
  { name: 'SCATTER-GATHER', recall: 100.0, color: '#fbbf24', count: '42 patterns' },
  { name: 'FAN-IN', recall: 100.0, color: '#34d399', count: '39 patterns' },
  { name: 'RANDOM', recall: 100.0, color: '#94a3b8', count: '35 patterns' },
  { name: 'FAN-OUT', recall: 96.4, color: '#a78bfa', count: '56 patterns' },
  { name: 'STACK', recall: 96.4, color: '#f97316', count: '52 patterns' },
  { name: 'BIPARTITE', recall: 95.7, color: '#22d3ee', count: '44 patterns' },
];

const typologyExplanations = [
  {
    type: 'CYCLE (Round-Tripping)',
    icon: Repeat,
    color: '#c084fc',
    recall: '100.0%',
    flow: 'A ➔ B ➔ C ➔ D ➔ A',
    description: 'Illicit funds circulate through a closed chain of accounts across multiple banks and return to the original beneficial owner.'
  },
  {
    type: 'FAN-OUT (Smurfing / Structuring)',
    icon: GitFork,
    color: '#a78bfa',
    recall: '96.4%',
    flow: '1 Source ➔ Many Mules',
    description: 'A large illicit deposit is rapidly fragmented into numerous sub-threshold transfers sent to dozens of mule accounts.'
  },
  {
    type: 'FAN-IN (Funnel Account)',
    icon: GitMerge,
    color: '#34d399',
    recall: '100.0%',
    flow: 'Many Mules ➔ 1 Destination',
    description: 'Dozens of small structured deposits from disparate sources are concentrated into a single beneficiary master account.'
  },
  {
    type: 'GATHER-SCATTER (Layering Hub)',
    icon: Layers,
    color: '#fb7185',
    recall: '100.0%',
    flow: 'Many ➔ Hub ➔ Many',
    description: 'Intermediary shell accounts gather funds from multiple incoming sources and immediately disperse them to new destinations.'
  },
  {
    type: 'STACK (Multi-Tier Pass-Through)',
    icon: ArrowRight,
    color: '#f97316',
    recall: '96.4%',
    flow: 'A ➔ B ➔ C ➔ D',
    description: 'A linear multi-hop chain moving money across successive banks to distance the capital from its illegal criminal origin.'
  },
  {
    type: 'BIPARTITE (Collusion Clusters)',
    icon: Network,
    color: '#22d3ee',
    recall: '95.7%',
    flow: 'Group X ➔ Group Y',
    description: 'Complete cross-transfers between two distinct sets of colluding entity accounts simulating legitimate commercial billing.'
  }
];

const CustomTooltip = ({ active, payload }) => {
  if (active && payload && payload.length) {
    return (
      <div className="typology-chart-tooltip">
        <p className="tooltip-name">{payload[0].payload.name}</p>
        <p className="tooltip-val">{payload[0].value}% Scheme Recall</p>
        <span className="tooltip-count">{payload[0].payload.count} in ground truth</span>
      </div>
    );
  }
  return null;
};

export default function TypologyBreakdown() {
  return (
    <section className="section typology-section" id="typologies">
      <div className="container">
        <div className="section-badge-wrapper">
          <span className="badge badge-emerald">
            <Network size={14} /> Pattern Recognition Intelligence
          </span>
        </div>
        <h2 className="section-title">Typology-Based Detection Performance</h2>
        <p className="section-subtitle">
          In financial forensics, launderers execute complex topological patterns beyond isolated transactions. Our model is evaluated across all 8 canonical AML typologies in the test set.
        </p>

        {/* Recharts Bar Chart */}
        <div className="glass-card typology-chart-card">
          <div className="chart-card-header">
            <div>
              <h3 className="chart-title">Detection Recall Across 370 Ground-Truth Laundering Schemes</h3>
              <p className="chart-sub">Test set evaluation (Days 10–18). A scheme is marked detected if the GNN flags its constituent transfers.</p>
            </div>
            <span className="badge badge-emerald">98.7% Overall Scheme Capture</span>
          </div>

          <div className="chart-wrapper">
            <ResponsiveContainer width="100%" height={340}>
              <BarChart data={typologyData} margin={{ top: 20, right: 30, left: 0, bottom: 5 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" />
                <XAxis
                  dataKey="name"
                  tick={{ fill: '#94a3b8', fontSize: 11, fontFamily: 'JetBrains Mono' }}
                  axisLine={{ stroke: 'rgba(255,255,255,0.08)' }}
                  tickLine={false}
                />
                <YAxis
                  domain={[90, 100]}
                  tick={{ fill: '#94a3b8', fontSize: 11 }}
                  axisLine={{ stroke: 'rgba(255,255,255,0.08)' }}
                  tickLine={false}
                  unit="%"
                />
                <Tooltip content={<CustomTooltip />} cursor={{ fill: 'rgba(255,255,255,0.03)' }} />
                <Bar dataKey="recall" radius={[6, 6, 0, 0]} maxBarSize={48}>
                  {typologyData.map((entry, index) => (
                    <Cell key={`cell-${index}`} fill={entry.color} fillOpacity={0.88} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>

        {/* Visual Typology Cards Grid */}
        <div className="typology-cards-grid">
          {typologyExplanations.map((item) => {
            const Icon = item.icon;
            return (
              <div key={item.type} className="glass-card typology-card" style={{ borderLeftColor: item.color }}>
                <div className="typology-header">
                  <div className="typology-icon-wrapper" style={{ color: item.color, borderColor: `${item.color}33` }}>
                    <Icon size={20} />
                  </div>
                  <div>
                    <h4 className="typology-title">{item.type}</h4>
                    <span className="typology-recall" style={{ color: item.color }}>{item.recall} Recall</span>
                  </div>
                </div>

                <div className="typology-flow-badge">
                  <code>{item.flow}</code>
                </div>

                <p className="typology-desc">{item.description}</p>
              </div>
            );
          })}
        </div>
      </div>
    </section>
  );
}
