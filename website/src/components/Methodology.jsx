import { Database, GitBranch, Brain, Target, Repeat } from 'lucide-react';
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
    title: 'Inductive GraphSAGE',
    badge: 'Phase 2',
    badgeClass: 'badge-purple',
    description: 'Designed an inductive 2-layer GraphSAGE encoder with a dedicated Edge Classifier MLP for transaction-level fraud prediction.',
    details: [
      'Mean aggregation over sampled neighborhoods',
      'Inductive: scores unseen accounts without retraining',
      'Edge classifier concatenates source + destination embeddings + edge features',
      'Outputs calibrated probability score per transaction',
    ],
    color: 'var(--accent-purple)',
  },
  {
    icon: Repeat,
    number: '04',
    title: 'Cycle Attention Layer',
    badge: 'Phase 3 (New)',
    badgeClass: 'badge-emerald',
    description: 'Engineered random-walk cycle return probabilities and an edge-conditioned structural attention mechanism to detect closed laundering loops.',
    details: [
      'Matrix power random walk cycle return probs: R_k(u,v) = [P^k]_{v,u}',
      'Extracts 2, 3, 4, 5-hop return probabilities in O(k·|E|·d_avg)',
      'CycleAwareSAGEConv overcomes standard 1-WL expressiveness limits',
      'Attention shields innocent accounts and preserves sparse cycle motifs',
    ],
    color: '#c084fc',
  },
  {
    icon: Target,
    number: '05',
    title: 'Training & Imbalance Strategy',
    badge: 'Phase 2 & 3',
    badgeClass: 'badge-amber',
    description: 'Trained with mini-batching via LinkNeighborLoader. Handled extreme class imbalance with weighted BCE loss (pos_weight ≈ 979).',
    details: [
      'Mini-batch sampling: 15 neighbors (hop 1), 10 (hop 2)',
      'Metrics: ROC-AUC, PR-AUC, Typology Pattern Recall',
      'Parses all 370 canonical laundering schemes from metadata',
      'Achieves 85.2% complete closed loop laundering detection',
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
          A rigorous, end-to-end scientific pipeline transitioning from raw financial logs to a cycle-aware inductive GNN.
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