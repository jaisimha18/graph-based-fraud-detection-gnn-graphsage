import React, { useState } from 'react';
import { ShieldAlert, ArrowRight, RefreshCw, AlertTriangle, CheckCircle2, Network, Eye } from 'lucide-react';
import './ProblemStatement.css';

export default function ProblemStatement() {
  const [activeTab, setActiveTab] = useState('circular');

  return (
    <section className="section problem-section" id="problem">
      <div className="container">
        <div className="section-badge-wrapper">
          <span className="badge badge-amber">
            <AlertTriangle size={14} /> The Financial Crime Challenge
          </span>
        </div>
        <h2 className="section-title">Why Traditional Fraud Detection Fails</h2>
        <p className="section-subtitle">
          Modern money laundering syndicates do not make simple, high-value transfers. They coordinate networks of accounts across multiple banks to systematically bypass rule-based transaction monitoring.
        </p>

        {/* Visual Flow Comparison */}
        <div className="flow-comparison-container glass-card">
          <div className="flow-tabs">
            <button
              className={`flow-tab-btn ${activeTab === 'traditional' ? 'active' : ''}`}
              onClick={() => setActiveTab('traditional')}
            >
              <span className="tab-indicator red"></span>
              Scenario 1: Conventional Linear Flow
            </button>
            <button
              className={`flow-tab-btn ${activeTab === 'circular' ? 'active' : ''}`}
              onClick={() => setActiveTab('circular')}
            >
              <span className="tab-indicator purple"></span>
              Scenario 2: Coordinated Circular Laundering Loop
            </button>
          </div>

          <div className="flow-display-area">
            {activeTab === 'traditional' ? (
              <div className="flow-scenario">
                <div className="scenario-header">
                  <div>
                    <h3 className="scenario-title">Linear Transaction Chain ($A \to B \to C \to D$)</h3>
                    <p className="scenario-desc">
                      Transactions evaluated individually in isolated database rows. Conventional Transaction Monitoring Systems (TMS) look for individual anomalies (e.g., transfers &gt; $10,000).
                    </p>
                  </div>
                  <span className="badge badge-red">Conventional Rule Monitoring</span>
                </div>

                <div className="flow-diagram-visual">
                  <div className="flow-node">
                    <span className="node-id">Acc A</span>
                    <span className="node-bank">Bank 10</span>
                  </div>
                  <div className="flow-edge">
                    <span className="edge-amount">$8,500</span>
                    <span className="edge-arrow">➔</span>
                    <span className="edge-status safe">Below $10K (Ignored)</span>
                  </div>

                  <div className="flow-node">
                    <span className="node-id">Acc B</span>
                    <span className="node-bank">Bank 20</span>
                  </div>
                  <div className="flow-edge">
                    <span className="edge-amount">$8,450</span>
                    <span className="edge-arrow">➔</span>
                    <span className="edge-status safe">Below $10K (Ignored)</span>
                  </div>

                  <div className="flow-node">
                    <span className="node-id">Acc C</span>
                    <span className="node-bank">Bank 30</span>
                  </div>
                  <div className="flow-edge">
                    <span className="edge-amount">$8,400</span>
                    <span className="edge-arrow">➔</span>
                    <span className="edge-status safe">Below $10K (Ignored)</span>
                  </div>

                  <div className="flow-node">
                    <span className="node-id">Acc D</span>
                    <span className="node-bank">Bank 40</span>
                  </div>
                </div>

                <div className="scenario-verdict red">
                  <AlertTriangle size={18} />
                  <span>
                    <strong>Vulnerability:</strong> Because each transfer is kept below reporting thresholds and spreads across independent banks, traditional systems evaluate them as 4 separate legitimate payments.
                  </span>
                </div>
              </div>
            ) : (
              <div className="flow-scenario">
                <div className="scenario-header">
                  <div>
                    <h3 className="scenario-title">Circular Laundering Loop ($A \to B \to C \to D \to A$)</h3>
                    <p className="scenario-desc">
                      Funds move through intermediary shell accounts and systematically return to the beneficial owner, disguising the true origin and purpose of the illicit capital.
                    </p>
                  </div>
                  <span className="badge badge-purple">Graph-Level Laundering Topology</span>
                </div>

                <div className="flow-diagram-visual circular-visual">
                  <div className="flow-node highlight-loop">
                    <span className="node-id">Acc A (Origin)</span>
                    <span className="node-bank">Bank 10</span>
                  </div>
                  <div className="flow-edge loop-edge">
                    <span className="edge-amount">$10,388</span>
                    <span className="edge-arrow">➔</span>
                    <span className="edge-status purple">Layer 1</span>
                  </div>

                  <div className="flow-node highlight-loop">
                    <span className="node-id">Acc B</span>
                    <span className="node-bank">Bank 20</span>
                  </div>
                  <div className="flow-edge loop-edge">
                    <span className="edge-amount">$10,250</span>
                    <span className="edge-arrow">➔</span>
                    <span className="edge-status purple">Layer 2</span>
                  </div>

                  <div className="flow-node highlight-loop">
                    <span className="node-id">Acc C</span>
                    <span className="node-bank">Bank 30</span>
                  </div>
                  <div className="flow-edge loop-edge">
                    <span className="edge-amount">$10,100</span>
                    <span className="edge-arrow">➔</span>
                    <span className="edge-status purple">Layer 3</span>
                  </div>

                  <div className="flow-node highlight-loop">
                    <span className="node-id">Acc D</span>
                    <span className="node-bank">Bank 40</span>
                  </div>
                  <div className="flow-edge loop-edge return-edge">
                    <span className="edge-amount">$9,950 ↺</span>
                    <span className="edge-arrow">➔ Returns to A</span>
                    <span className="edge-status rose">Round-Trip Closed!</span>
                  </div>
                </div>

                <div className="scenario-verdict purple">
                  <CheckCircle2 size={18} />
                  <span>
                    <strong>Graph Solution:</strong> Our Cycle-Aware GraphSAGE extracts the random-walk cycle return probability and applies structural attention to detect the complete closed ring ($A \to B \to C \to D \to A$).
                  </span>
                </div>
              </div>
            )}
          </div>
        </div>

        {/* Side-by-Side Methodology Comparison Cards */}
        <div className="comparison-cards-grid">
          <div className="glass-card comp-card traditional-card">
            <div className="comp-card-header">
              <ShieldAlert size={22} className="text-rose" />
              <h3>Traditional Transaction-Level Monitoring</h3>
            </div>
            <p className="comp-card-sub">Rules &amp; Tabular Machine Learning (XGBoost, Random Forest)</p>

            <ul className="comp-list">
              <li>
                <strong>Row-by-Row Isolation:</strong> Treats each transaction as an independent, identically distributed (i.i.d.) event.
              </li>
              <li>
                <strong>Static Thresholds:</strong> Flags transfers based solely on single-edge attributes (e.g. amount &gt; $10,000, off-hours).
              </li>
              <li>
                <strong>Blind to Multi-Hop Topology:</strong> Cannot follow funds as they traverse 3, 4, or 5 accounts across different banks.
              </li>
              <li>
                <strong>High Compliance Alert Fatigue:</strong> Over 95% of rule-based alerts are false positives, overwhelming compliance investigators.
              </li>
            </ul>
          </div>

          <div className="glass-card comp-card proposed-card">
            <div className="comp-card-header">
              <Network size={22} className="text-emerald" />
              <h3>Our Proposed Cycle-Aware Graph Learning</h3>
            </div>
            <p className="comp-card-sub">Inductive GraphSAGE + Random Walk Cycles + Structural Attention</p>

            <ul className="comp-list">
              <li>
                <strong>Holistic Network Context:</strong> Models bank accounts as nodes and transfers as directed edges in a unified financial multigraph.
              </li>
              <li>
                <strong>Inductive Multi-Hop Aggregation:</strong> Learns aggregator functions capable of scoring newly opened accounts without retraining.
              </li>
              <li>
                <strong>Random Walk Cycle Return Probabilities ($R_k$):</strong> Computes the exact mathematical probability that money returns to the source account.
              </li>
              <li>
                <strong>Selective Structural Attention:</strong> Prioritizes suspicious cyclic relationships, slashing false-alarm compliance noise by 32%.
              </li>
            </ul>
          </div>
        </div>
      </div>
    </section>
  );
}