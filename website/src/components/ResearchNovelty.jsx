import React, { useState } from 'react';
import { Sparkles, TrendingDown, TrendingUp, AlertOctagon, CheckCircle2, Split, ArrowDown, HelpCircle } from 'lucide-react';
import './ResearchNovelty.css';

const ablationData = [
  {
    name: '1. Vanilla GraphSAGE',
    mechanism: 'Uniform Mean Aggregation',
    rocAuc: '0.9853',
    prAuc: '0.3227',
    fraudRec: '87.5%',
    alertRec: '99.1%',
    fullRing: '79.0%',
    status: 'Baseline Benchmark',
    badgeClass: 'badge-blue'
  },
  {
    name: '2. + Naive RW Cycle Features',
    mechanism: 'Direct Feature Concatenation + Mean',
    rocAuc: '0.9704',
    prAuc: '0.1113',
    fraudRec: '75.3%',
    alertRec: '96.1%',
    fullRing: '63.8%',
    status: '-65.6% Collapse (Dilution Trap)',
    badgeClass: 'badge-amber',
    isFailure: true
  },
  {
    name: '3. Proposed Cycle Attention',
    mechanism: 'Learned Structural Attention (α_ji)',
    rocAuc: '0.9889',
    prAuc: '0.5759',
    fraudRec: '87.7%',
    alertRec: '98.7%',
    fullRing: '85.2%',
    status: '+78.5% PR-AUC Gain 🏆',
    badgeClass: 'badge-emerald',
    isWinner: true
  }
];

export default function ResearchNovelty() {
  const [activeTab, setActiveTab] = useState('dilution');

  return (
    <section className="section novelty-section" id="research-novelty">
      <div className="container">
        <div className="section-badge-wrapper">
          <span className="badge badge-purple">
            <Sparkles size={14} /> Core Scientific Contribution
          </span>
        </div>
        <h2 className="section-title">Our Research Novelty: Why Cycle Attention?</h2>
        <p className="section-subtitle">
          Answering the critical research question: <em>"Why can't standard Graph Neural Networks detect laundering cycles, and why does naive feature concatenation catastrophically fail?"</em>
        </p>

        {/* 3-Stage Ablation Benchmark Table */}
        <div className="glass-card ablation-benchmark-card">
          <div className="benchmark-card-header">
            <div>
              <h3 className="benchmark-title">Verified 3-Stage Experimental Ablation Study</h3>
              <p className="benchmark-desc">
                Evaluated under identical conditions on the held-out test split (Days 10–18) of the IBM HI-Small AML dataset.
              </p>
            </div>
            <span className="badge badge-blue">Chronological Evaluation</span>
          </div>

          <div className="table-responsive">
            <table className="ablation-data-table">
              <thead>
                <tr>
                  <th>Configuration</th>
                  <th>Aggregation Mechanism</th>
                  <th>ROC-AUC</th>
                  <th>PR-AUC</th>
                  <th>Fraud Recall</th>
                  <th>Alert Recall</th>
                  <th>Full Ring Recall</th>
                  <th>Experimental Outcome</th>
                </tr>
              </thead>
              <tbody>
                {ablationData.map((row) => (
                  <tr key={row.name} className={row.isWinner ? 'winner-row' : (row.isFailure ? 'failure-row' : '')}>
                    <td>
                      <strong>{row.name}</strong>
                    </td>
                    <td><code>{row.mechanism}</code></td>
                    <td>{row.rocAuc}</td>
                    <td className="metric-pr-auc">{row.prAuc}</td>
                    <td>{row.fraudRec}</td>
                    <td>{row.alertRec}</td>
                    <td className="metric-ring">{row.fullRing}</td>
                    <td>
                      <span className={`badge ${row.badgeClass}`}>{row.status}</span>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>

        {/* Visual Mechanism Comparison: Dilution Trap vs Cycle Attention */}
        <div className="mechanism-comparison-wrapper">
          <div className="mechanism-nav-tabs">
            <button
              className={`mech-tab ${activeTab === 'dilution' ? 'active' : ''}`}
              onClick={() => setActiveTab('dilution')}
            >
              <TrendingDown size={16} className="text-amber" />
              Experiment 2: The Signal Dilution Trap (-65.6%)
            </button>
            <button
              className={`mech-tab ${activeTab === 'attention' ? 'active' : ''}`}
              onClick={() => setActiveTab('attention')}
            >
              <TrendingUp size={16} className="text-emerald" />
              Experiment 3: Our Proposed Cycle Attention (+78.5%)
            </button>
          </div>

          <div className="mechanism-display-card glass-card">
            {activeTab === 'dilution' ? (
              <div className="mechanism-content">
                <div className="mech-header">
                  <h3 className="mech-title">Why Naive Feature Concatenation Fails (-65.6% PR-AUC)</h3>
                  <span className="badge badge-amber">The Arithmetic Mean Trap</span>
                </div>

                <div className="mech-visual-diagram">
                  <div className="mech-diagram-box">
                    <span className="diagram-box-title">Incoming Transfers to Account B</span>
                    <div className="diagram-stream">
                      <div className="transfer-item fraud">
                        <span>1 Laundering Cycle Edge</span>
                        <code>R_k &gt; 0</code>
                      </div>
                      <span className="plus-sign">+</span>
                      <div className="transfer-item legit">
                        <span>99 Legitimate Transactions</span>
                        <code>R_k = 0 (Coffee, Salaries, Bills)</code>
                      </div>
                    </div>
                  </div>

                  <div className="arrow-down-divider">
                    <span>Uniform Arithmetic Mean Aggregation (1 / 100)</span>
                    <ArrowDown size={20} />
                  </div>

                  <div className="mech-diagram-result failure">
                    <AlertOctagon size={24} className="text-amber" />
                    <div>
                      <strong>The Cycle Signal Is 99% Diluted!</strong>
                      <p>
                        Averaging forces the non-zero cycle signal into a tiny fractional decimal. Standard linear layers fail to separate laundering accounts, mistaking high-degree merchant accounts for criminals and causing PR-AUC to plummet from 0.3227 to 0.1113.
                      </p>
                    </div>
                  </div>
                </div>
              </div>
            ) : (
              <div className="mechanism-content">
                <div className="mech-header">
                  <h3 className="mech-title">How Learned Cycle Attention Solves Signal Dilution (+78.5% PR-AUC)</h3>
                  <span className="badge badge-emerald">Selective Structural Attention</span>
                </div>

                <div className="mech-visual-diagram">
                  <div className="mech-diagram-box">
                    <span className="diagram-box-title">Learned Attention Weighting ($\alpha_{ji}$)</span>
                    <div className="diagram-stream">
                      <div className="transfer-item attention-high">
                        <span>Laundering Cycle Edge ($R_k &gt; 0$)</span>
                        <code className="text-emerald">High Attention: α_ji = 0.88</code>
                      </div>
                      <span className="plus-sign">vs</span>
                      <div className="transfer-item attention-low">
                        <span>99 Benign Transfers ($R_k = 0$)</span>
                        <code className="text-muted">Low Attention: α_ji ≈ 0.001</code>
                      </div>
                    </div>
                  </div>

                  <div className="arrow-down-divider">
                    <span>Attention-Weighted Neighborhood Update: $\sum \alpha_{ji} \cdot W h_j$</span>
                    <ArrowDown size={20} />
                  </div>

                  <div className="mech-diagram-result success">
                    <CheckCircle2 size={24} className="text-emerald" />
                    <div>
                      <strong>Cycle Signal Preserved &amp; Innocent Accounts Shielded!</strong>
                      <p>
                        Our <code>CycleAwareSAGEConv</code> dynamically learns attention coefficients conditioned on cycle return probabilities. Suspicious loops are sharply emphasized while innocent merchant accounts are shielded, driving PR-AUC to <strong>0.5759</strong> and full ring recall to <strong>85.2%</strong>.
                      </p>
                    </div>
                  </div>
                </div>
              </div>
            )}
          </div>
        </div>

        {/* Theoretical 1-WL Limitation Box */}
        <div className="glass-card wl-theory-box">
          <div className="wl-header">
            <HelpCircle size={20} className="text-purple" />
            <h4>Theoretical Foundation: Breaking the 1-WL Expressiveness Bound</h4>
          </div>
          <p className="wl-body">
            Theoretical research in Graph Deep Learning (the <strong>1-Weisfeiler-Lehman / 1-WL isomorphism test</strong>) proves that standard message-passing GNNs (Vanilla GraphSAGE, GCN, standard GAT) <em>cannot count cycles or distinguish whether an edge belongs to a closed loop or a tree</em>. By precomputing random-walk transition matrix powers R_k(u, v) = [P^k]_(v, u) and coupling them with structural attention, our architecture explicitly overcomes the 1-WL cycle blindness.
          </p>
        </div>
      </div>
    </section>
  );
}