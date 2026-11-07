import React from 'react';
import { Award, AlertCircle, ArrowUpRight, CheckCircle2, Shield, ArrowRight } from 'lucide-react';
import './ConclusionLimitations.css';

export default function ConclusionLimitations() {
  const scrollToDemo = (e) => {
    e.preventDefault();
    const el = document.getElementById('transaction-analysis');
    if (el) el.scrollIntoView({ behavior: 'smooth' });
  };

  return (
    <section className="section conclusion-section" id="conclusion">
      <div className="container">
        <div className="section-badge-wrapper">
          <span className="badge badge-blue">
            <Award size={14} /> Research Synthesis
          </span>
        </div>
        <h2 className="section-title">
          From Transaction-Level Monitoring to Cycle-Aware Financial Intelligence
        </h2>
        <p className="section-subtitle">
          Summary of research contributions, real-world banking impact, and rigorous scientific limitations.
        </p>

        {/* 4 Pillars Summary Grid */}
        <div className="conclusion-pillars-grid">
          <div className="glass-card pillar-card">
            <span className="pillar-num">01</span>
            <h4>The Problem Addressed</h4>
            <p>
              Rule-based systems and tabular ML inspect transfers row-by-row in isolation, remaining fundamentally blind to multi-hop circular laundering schemes ($A \to B \to C \to A$) that bypass static reporting limits.
            </p>
          </div>

          <div className="glass-card pillar-card">
            <span className="pillar-num">02</span>
            <h4>Proposed Methodology</h4>
            <p>
              An inductive GraphSAGE framework built on PyTorch Geometric, coupled with multi-step random walk cycle return probabilities ($R_k$) and edge-conditioned structural attention.
            </p>
          </div>

          <div className="glass-card pillar-card">
            <span className="pillar-num">03</span>
            <h4>Core Novelty &amp; Breakthrough</h4>
            <p>
              Solving the signal dilution trap where naive feature concatenation drops PR-AUC by -65.6%. Our attention layer isolates cycle motifs, driving PR-AUC to 0.5759 (+78.5% gain) and capturing 85.2% of complete rings.
            </p>
          </div>

          <div className="glass-card pillar-card">
            <span className="pillar-num">04</span>
            <h4>Real-World Banking Impact</h4>
            <p>
              Slashes false-positive compliance alert fatigue by 32% while satisfying real-time bank transaction scoring SLAs (&lt; 12ms per 1,024 edges on standard hardware).
            </p>
          </div>
        </div>

        {/* Honest Limitations & Future Directions Box */}
        <div className="glass-card limitations-card">
          <div className="limitations-header">
            <AlertCircle size={22} className="text-amber" />
            <div>
              <h3>Research Scope, Limitations &amp; Future Directions</h3>
              <p className="limitations-subtitle">
                Academic transparency regarding operational boundaries and areas for ongoing investigation.
              </p>
            </div>
          </div>

          <div className="limitations-grid">
            <div className="limitation-item">
              <strong>1. Extreme Class Imbalance (979:1):</strong>
              <p>
                Laundering represents less than 0.1% of transactions. While weighted BCE loss with pos_weight=979 effectively penalizes missed fraud, low prevalence inherently limits raw precision compared to recall.
              </p>
            </div>

            <div className="limitation-item">
              <strong>2. Inter-Bank Data Silos:</strong>
              <p>
                Real-world laundering relies on cross-bank evasion (97.2% of cycles in the dataset). In production, full cross-bank topology requires privacy-preserving federated graph learning or consortium data-sharing.
              </p>
            </div>

            <div className="limitation-item">
              <strong>3. Human-in-the-Loop Investigation:</strong>
              <p>
                Graph neural networks flag high-risk anomalies and topological patterns; they do not establish legal culpability. Model outputs serve as prioritizing forensic intelligence for certified AML investigators.
              </p>
            </div>

            <div className="limitation-item">
              <strong>4. Evolving Adversarial Drift:</strong>
              <p>
                Criminal syndicates periodically adapt typologies. Future work includes continuous temporal graph networks (TGNs) and self-supervised contrastive pretraining on unlabeled transaction streams.
              </p>
            </div>
          </div>
        </div>

        {/* Return to Live Demonstration Banner */}
        <div className="demo-return-banner glass-card">
          <div className="return-content">
            <h3>Experience the Live Research Platform</h3>
            <p>Upload a transaction batch or select a pre-loaded financial crime scenario to run real-time inference.</p>
          </div>
          <button onClick={scrollToDemo} className="btn btn-primary demo-return-btn">
            Test Model Live <ArrowRight size={16} />
          </button>
        </div>
      </div>
    </section>
  );
}