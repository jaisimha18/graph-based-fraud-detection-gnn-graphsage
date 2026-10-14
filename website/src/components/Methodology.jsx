import { Database, GitBranch, Brain, Target } from 'lucide-react';
import './Methodology.css';

const steps = [
  {
    icon: Database,
    number: '01',
    title: 'Data Preprocessing',
    badge: 'Phase 1',
    badgeClass: 'badge-blue',
    description: 'Cleaned the IBM HI-Small AML dataset. Implemented strict chronological splitting (Days 1–8 Train, Day 9 Val, Days 10–18 Test) to simulate real-world streaming.',
    details: [
      'Training-only account aggregates to prevent data leakage',
      'StandardScaler fit exclusively on training accounts',
      'Removed laundering_rate target-leaking feature',
      'Cyclical sine/cosine encoding for timestamps',
    ],
    color: 'var(--accent-blue)',
  },
  {
    icon: GitBranch,
    number: '02',
    title: 'Graph Construction',
    badge: 'Phase 1',
    badgeClass: 'badge-emerald',
    description: 'Converted tabular CSVs into a PyTorch Geometric (PyG) tensor-based graph. Nodes represent accounts, edges represent transactions.',
    details: [
      'Node features: [518,573 × 8] matrix (bank ID, entity type, aggregates)',
      'Edge features: [E × 14] tensor (amount, currency, time)',
      'Single graph with boolean train/val/test masks',
      'Continuous integer ID mapping for all accounts',
    ],
    color: 'var(--accent-emerald)',
  },
  {
    icon: Brain,
    number: '03',
    title: 'GraphSAGE Model',
    badge: 'Phase 2',
    badgeClass: 'badge-purple',
    description: 'Designed an inductive 2-layer GraphSAGE encoder with a dedicated Edge Classifier MLP for transaction-level fraud prediction.',
    details: [
      'Mean aggregation over sampled neighborhoods',
      'Inductive: handles unseen accounts at inference',
      'Edge classifier concatenates source + destination embeddings + edge features',
      'Outputs probability score (0.0 – 1.0) per transaction',
    ],
    color: 'var(--accent-purple)',
  },
  {
    icon: Target,
    number: '04',
    title: 'Training & Evaluation',
    badge: 'Phase 2',
    badgeClass: 'badge-amber',
    description: 'Trained with mini-batching via LinkNeighborLoader. Handled extreme class imbalance with weighted BCE loss (pos_weight ≈ 979).',
    details: [
      'Mini-batch sampling: 15 neighbors (hop 1), 10 (hop 2)',
      'Metrics: ROC-AUC, PR-AUC, Pattern-Level Recall',
      'Custom pattern parser for 370 laundering schemes',
      'Per-type detection rates (Cycle, Fan-Out, Bipartite, etc.)',
    ],
    color: 'var(--accent-amber)',
  },
];

export default function Methodology() {
  return (
    <section className="section" id="methodology">
      <div className="container">
        <h2 className="section-title">Methodology Pipeline</h2>
        <p className="section-subtitle">
          A rigorous, four-phase approach from raw data to production-ready graph-based fraud detection.
        </p>

        <div className="methodology-grid">
          {steps.map((step, i) => {
            const Icon = step.icon;
            return (
              <div
                key={step.number}
                className={`glass-card methodology-card animate-in animate-in-delay-${i + 1}`}
              >
                <div className="methodology-header">
                  <div className="methodology-icon" style={{ color: step.color, borderColor: `${step.color}33` }}>
                    <Icon size={24} />
                  </div>
                  <span className="methodology-number">{step.number}</span>
                </div>

                <span className={`badge ${step.badgeClass}`}>{step.badge}</span>
                <h3 className="methodology-title">{step.title}</h3>
                <p className="methodology-desc">{step.description}</p>

                <ul className="methodology-details">
                  {step.details.map((detail, j) => (
                    <li key={j} style={{ '--dot-color': step.color }}>
                      {detail}
                    </li>
                  ))}
                </ul>
              </div>
            );
          })}
        </div>
      </div>
    </section>
  );
}
