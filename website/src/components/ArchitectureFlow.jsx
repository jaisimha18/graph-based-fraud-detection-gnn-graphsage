import React, { useState } from 'react';
import { Database, Network, Repeat, Brain, ShieldAlert, ArrowRight, CheckCircle2, ChevronRight, Layers } from 'lucide-react';
import './ArchitectureFlow.css';

const stages = [
  {
    id: 1,
    title: 'Transaction Data Input',
    short: 'Raw Ingestion',
    icon: Database,
    badge: 'Stage 01',
    color: '#38bdf8',
    summary: 'Streaming transactions from the IBM HI-Small AML dataset with multi-bank metadata.',
    whatHappens: 'Financial transaction logs are ingested with raw attributes: source and destination account IDs, bank IDs, log-scaled amounts, receiving/payment currencies, payment formats, and timestamps.',
    whyNeeded: 'Raw tabular banking data is fragmented across banks and time. Real money laundering operations intentionally disperse transfers across multiple dates and institutions to avoid triggering single-account volume alerts.',
    input: 'Tabular CSV records (5,078,345 transactions, 518,573 unique accounts).',
    output: 'Normalized tabular feature matrices split chronologically (Days 1–8 Train, Day 9 Val, Days 10–18 Test) to prevent temporal data leakage.'
  },
  {
    id: 2,
    title: 'Graph Construction',
    short: 'Topology Graph',
    icon: Network,
    badge: 'Stage 02',
    color: '#34d399',
    summary: 'Transforming tabular rows into a PyTorch Geometric (PyG) directed multigraph.',
    whatHappens: 'Every unique bank account becomes a node (518,573 nodes with 8-dimensional historical behavior features). Every financial transfer becomes a directed edge containing 14 transaction features (amounts, cross-bank indicators, cyclical time embeddings).',
    whyNeeded: 'Standard machine learning treats each row independently. Constructing a directed multigraph allows message passing to traverse financial corridors and see multi-hop paths.',
    input: 'Preprocessed account feature matrix [518,573 × 8] and transaction edge lists.',
    output: 'Unified PyG Data object containing edge_index [2, E], edge_attr [E, 14], node features x [518573, 8], and edge train/val/test masks.'
  },
  {
    id: 3,
    title: 'Random Walk Cycle Extraction',
    short: 'Cycle Matrix (Rk)',
    icon: Repeat,
    badge: 'Stage 03 (Novel)',
    color: '#c084fc',
    summary: 'Mathematically extracting multi-step return probabilities via sparse matrix powers in O(k·|E|·d_avg).',
    whatHappens: 'We construct a row-stochastic transition matrix P = D_out^-1 * A and compute matrix powers P^2, P^3, P^4, P^5. For each edge (u ➔ v), the cycle return probability after k steps is extracted as R_k(u,v) = [P^k]_{v,u}.',
    whyNeeded: 'Standard GNNs (like GCN and GAT) are bound by the 1-Weisfeiler-Lehman (1-WL) graph isomorphism test, which mathematically proves they cannot count cycles or detect whether an edge is in a closed ring. Precomputing R_k explicitly injects cycle awareness into the edge feature tensor.',
    input: 'Directed edge topology strictly constrained to historical windows to guarantee zero data-leakage.',
    output: 'An 8-dimensional structural cycle feature vector appended to each edge (2, 3, 4, 5-hop return probabilities and discounted loop indicators).'
  },
  {
    id: 4,
    title: 'Cycle Attention GraphSAGE',
    short: 'Attention GNN',
    icon: Brain,
    badge: 'Stage 04 (Novel)',
    color: '#f59e0b',
    summary: 'Inductive neighborhood aggregation with edge-conditioned structural attention.',
    whatHappens: 'Our proposed CycleAwareSAGEConv layer aggregates neighbor messages. Instead of uniform arithmetic mean averaging, it computes a learned attention coefficient α_ji conditioned on edge features and cycle return probability R_k.',
    whyNeeded: 'Naive addition of cycle features collapses PR-AUC by -65.6% because averaging 1 fraud cycle edge with 99 normal transactions dilutes the signal. Cycle attention selectively spotlights anomalous circular flows while shielding benign accounts from false alerts.',
    input: 'Combined node features [N, 8] and 22-dimensional edge features (14 base + 8 cycle return features).',
    output: 'Enriched 64-dimensional inductive account embeddings and a dedicated Edge Classifier MLP.'
  },
  {
    id: 5,
    title: 'Fraud Prediction & Forensic Rings',
    short: 'Forensic Output',
    icon: ShieldAlert,
    badge: 'Stage 05',
    color: '#f43f5e',
    summary: 'Scoring edge laundering probabilities and extracting complete closed ring paths.',
    whatHappens: 'The classifier head outputs a calibrated fraud probability (0.0 to 1.0) for every transfer. When multiple edges in a closed loop (A ➔ B ➔ C ➔ D ➔ A) score high, our forensic engine reconstructs and isolates the full laundering ring.',
    whyNeeded: 'Compliance analysts cannot review millions of raw numbers. They need actionable forensic intelligence showing the exact accounts, total funds circulated, and the multi-hop sequence of the laundering scheme.',
    input: 'Forward pass logits evaluated against the optimal threshold (0.95 full / 0.60 mini).',
    output: 'Transaction risk labels, PR-AUC of 0.5759 (+78.5%), and an 85.2% complete closed-loop ring recall rate.'
  }
];

export default function ArchitectureFlow() {
  const [selectedStage, setSelectedStage] = useState(stages[3]); // Default to Cycle Attention (Novelty)

  return (
    <section className="section architecture-section" id="architecture">
      <div className="container">
        <div className="section-badge-wrapper">
          <span className="badge badge-purple">
            <Layers size={14} /> End-to-End System Design
          </span>
        </div>
        <h2 className="section-title">Visual Research Architecture</h2>
        <p className="section-subtitle">
          Our proposed 5-stage pipeline bridges raw banking transaction streams and inductive graph deep learning, introducing edge-conditioned Cycle Attention to detect evasive financial loops.
        </p>

        {/* 5-Stage Interactive Pipeline Flow */}
        <div className="pipeline-flow-wrapper">
          <div className="pipeline-steps-bar">
            {stages.map((stage, idx) => {
              const Icon = stage.icon;
              const isSelected = selectedStage.id === stage.id;
              return (
                <React.Fragment key={stage.id}>
                  <button
                    className={`pipeline-step-node ${isSelected ? 'active' : ''}`}
                    onClick={() => setSelectedStage(stage)}
                    style={{ '--step-color': stage.color }}
                  >
                    <div className="step-icon-circle">
                      <Icon size={20} />
                    </div>
                    <span className="step-badge">{stage.badge}</span>
                    <span className="step-title">{stage.short}</span>
                  </button>
                  {idx < stages.length - 1 && (
                    <div className="pipeline-connector-line">
                      <span className="connector-arrow">➔</span>
                    </div>
                  )}
                </React.Fragment>
              );
            })}
          </div>

          {/* Expanded Stage Inspection Card */}
          <div className="stage-detail-panel glass-card" style={{ borderLeftColor: selectedStage.color }}>
            <div className="detail-panel-header">
              <div className="detail-title-group">
                <span className="badge" style={{ background: `${selectedStage.color}1a`, color: selectedStage.color, borderColor: `${selectedStage.color}4d` }}>
                  {selectedStage.badge}
                </span>
                <h3 className="detail-stage-name">{selectedStage.title}</h3>
              </div>
              <p className="detail-summary-text">{selectedStage.summary}</p>
            </div>

            <div className="detail-grid-four">
              <div className="detail-box">
                <span className="box-label">⚙️ 1. What Happens?</span>
                <p>{selectedStage.whatHappens}</p>
              </div>

              <div className="detail-box">
                <span className="box-label">🎯 2. Why Is It Needed?</span>
                <p>{selectedStage.whyNeeded}</p>
              </div>

              <div className="detail-box">
                <span className="box-label">📥 3. Input Received</span>
                <p><code>{selectedStage.input}</code></p>
              </div>

              <div className="detail-box">
                <span className="box-label">📤 4. Output Generated</span>
                <p><code>{selectedStage.output}</code></p>
              </div>
            </div>
          </div>
        </div>
      </div>
    </section>
  );
}
