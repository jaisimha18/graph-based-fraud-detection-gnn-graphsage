import {
  ArrowRightLeft,
  Clock,
  Landmark,
  Repeat,
  Users,
  DollarSign,
  Globe,
  BarChart3,
  Layers,
} from 'lucide-react';
import './Insights.css';

const insights = [
  {
    icon: Repeat,
    title: 'All Cycles Closed',
    value: '100%',
    description: 'All 54 laundering cycles are fully closed loops — money returns to the originator.',
    color: 'var(--accent-blue)',
  },
  {
    icon: Landmark,
    title: 'Cross-Bank Hops',
    value: '97.2%',
    description: 'Nearly all laundering hops cross bank boundaries, making the is_cross_bank edge feature a strong signal.',
    color: 'var(--accent-emerald)',
  },
  {
    icon: Clock,
    title: 'Temporal Span',
    value: '2–5 days',
    description: '77.8% of cycles span 2–5 days. Multi-day patterns validate the need for temporal-aware splitting.',
    color: 'var(--accent-purple)',
  },
  {
    icon: ArrowRightLeft,
    title: 'Variable Cycle Length',
    value: '2–12 hops',
    description: 'Cycles range from 2 to 12 hops (mean 5.3), requiring multi-hop neighborhood aggregation — GraphSAGE\'s core strength.',
    color: 'var(--accent-amber)',
  },
  {
    icon: Users,
    title: 'Unique Accounts',
    value: '271',
    description: '271 unique accounts participate across all cycles. Top accounts appear up to 8 times in different schemes.',
    color: 'var(--accent-rose)',
  },
  {
    icon: DollarSign,
    title: 'Total Money Moved',
    value: '~$1B',
    description: 'Approximately $1 billion moved through laundering cycles. Per-hop median is ~$12K.',
    color: 'var(--accent-cyan)',
  },
  {
    icon: Globe,
    title: 'Currency Distribution',
    value: '3 dominant',
    description: 'USD (37.6%), EUR (25.8%), and SAR (14.6%) dominate laundering transactions.',
    color: 'var(--accent-blue)',
  },
  {
    icon: BarChart3,
    title: 'Pattern Types',
    value: '8 types',
    description: '370 total patterns across 8 types: Cycle, Fan-Out, Fan-In, Scatter-Gather, Gather-Scatter, Stack, Bipartite, Random.',
    color: 'var(--accent-emerald)',
  },
  {
    icon: Layers,
    title: 'Described vs Actual',
    value: '100% match',
    description: 'All 54 cycles exactly match their stated "Max N hops" from the dataset metadata.',
    color: 'var(--accent-purple)',
  },
];

export default function Insights() {
  return (
    <section className="section" id="insights">
      <div className="container">
        <h2 className="section-title">Key EDA Insights</h2>
        <p className="section-subtitle">
          Exploratory analysis of the 54 cycle laundering patterns reveals structural properties
          that directly inform model design decisions.
        </p>

        <div className="insights-grid">
          {insights.map(({ icon: Icon, title, value, description, color }, i) => (
            <div
              key={title}
              className={`glass-card insight-card animate-in animate-in-delay-${(i % 4) + 1}`}
            >
              <div className="insight-header">
                <div className="insight-icon" style={{ color, borderColor: `${color}33` }}>
                  <Icon size={20} />
                </div>
                <span className="insight-value" style={{ color }}>{value}</span>
              </div>
              <h3 className="insight-title">{title}</h3>
              <p className="insight-desc">{description}</p>
            </div>
          ))}
        </div>
      </div>
    </section>
  );
}
