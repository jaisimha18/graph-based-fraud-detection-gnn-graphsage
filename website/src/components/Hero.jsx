import React from 'react';
import { Shield, Activity, Repeat, TrendingUp, Target, Play, BookOpen, Compass, ArrowRight } from 'lucide-react';
import './Hero.css';

const stats = [
  { value: '518K+', label: 'Bank Accounts', icon: Shield, color: '#38bdf8' },
  { value: '5M+', label: 'Transactions Analyzed', icon: Activity, color: '#38bdf8' },
  { value: '85.2%', label: 'Full Ring Recall', icon: Repeat, color: '#c084fc' },
  { value: '0.5759', label: 'PR-AUC (+78.5% Gain)', icon: TrendingUp, color: '#10b981' },
  { value: '98.7%', label: 'Alert Scheme Recall', icon: Target, color: '#f59e0b' },
];

export default function Hero({ onStartTour, onOpenGlossary }) {
  const scrollTo = (id) => {
    const el = document.getElementById(id);
    if (el) {
      el.scrollIntoView({ behavior: 'smooth' });
    }
  };

  return (
    <section className="hero" id="hero">
      {/* Background Ambience */}
      <div className="hero-bg">
        <div className="hero-grid" />
        <div className="hero-glow hero-glow-1" />
        <div className="hero-glow hero-glow-2" />
        <div className="hero-glow hero-glow-3" />
      </div>

      <div className="container hero-content">
        {/* Research Badge */}
        <div className="hero-badge animate-in">
          <span className="badge badge-purple hero-pill">
            <Repeat size={14} className="hero-spin-icon" />
            Cycle-Aware Graph Neural Network Research
          </span>
        </div>

        {/* Main Title & Subtitle */}
        <h1 className="hero-title animate-in animate-in-delay-1">
          Cycle-Aware{' '}
          <span className="hero-title-accent">Anti-Money Laundering</span>
          <br />Detection with GraphSAGE
        </h1>

        <p className="hero-subtitle animate-in animate-in-delay-2">
          An inductive Graph Neural Network framework for uncovering coordinated financial fraud and circular money-laundering patterns.
        </p>

        <p className="hero-description animate-in animate-in-delay-2">
          Conventional tabular models analyze transactions in isolation, blind to multi-hop collusion. Our research pairs inductive GraphSAGE with Random Walk Cycle Features and learned structural attention—isolating complex, multi-bank laundering rings across 5+ million transactions with zero retraining.
        </p>

        {/* Technology Badges */}
        <div className="hero-tech-stack animate-in animate-in-delay-3">
          {['PyTorch Geometric', 'GraphSAGE', 'Cycle Attention', 'Random Walk Cycles', 'IBM AML Dataset'].map((tech) => (
            <span key={tech} className="tech-pill">{tech}</span>
          ))}
        </div>

        {/* Action CTAs */}
        <div className="hero-actions animate-in animate-in-delay-3">
          <button
            className="btn btn-primary hero-btn-main"
            onClick={() => scrollTo('transaction-analysis')}
          >
            <Play size={16} fill="currentColor" />
            Explore Live Demonstration
          </button>

          <button
            className="btn btn-secondary hero-btn-sub"
            onClick={() => scrollTo('research-novelty')}
          >
            Understand Our Innovation
            <ArrowRight size={16} />
          </button>

          <button
            className="btn btn-tour"
            onClick={onStartTour}
            title="Launch an interactive 6-step guided walkthrough"
          >
            <Compass size={16} />
            Guided Demo Tour
          </button>

          <button
            className="btn btn-glossary"
            onClick={onOpenGlossary}
            title="Open searchable research definitions and formulas"
          >
            <BookOpen size={16} />
            Research Glossary
          </button>
        </div>

        {/* Animated Circular Network Illustration */}
        <div className="hero-diagram-preview animate-in animate-in-delay-3">
          <div className="hero-diagram-card glass-card">
            <div className="diagram-card-label">
              <span className="status-dot warning" />
              <span>Coordinated Laundering Ring Identified: A ➔ B ➔ C ➔ D ➔ A (4 Hops, Cross-Bank)</span>
            </div>
            <div className="mini-cycle-flow">
              <div className="cycle-node origin">
                <span className="c-id">Acc A</span>
                <span className="c-bank">Bank 012</span>
              </div>
              <div className="cycle-edge animated">
                <span className="edge-amt">$48,500</span>
                <span className="edge-arrow">➔</span>
              </div>
              <div className="cycle-node intermediate">
                <span className="c-id">Acc B</span>
                <span className="c-bank">Bank 821</span>
              </div>
              <div className="cycle-edge animated">
                <span className="edge-amt">$47,900</span>
                <span className="edge-arrow">➔</span>
              </div>
              <div className="cycle-node intermediate">
                <span className="c-id">Acc C</span>
                <span className="c-bank">Bank 404</span>
              </div>
              <div className="cycle-edge animated">
                <span className="edge-amt">$47,200</span>
                <span className="edge-arrow">➔</span>
              </div>
              <div className="cycle-node intermediate">
                <span className="c-id">Acc D</span>
                <span className="c-bank">Bank 019</span>
              </div>
              <div className="cycle-edge animated cycle-back">
                <span className="edge-amt">$46,800 (Return Loop)</span>
                <span className="edge-arrow">↺ A</span>
              </div>
            </div>
          </div>
        </div>

        {/* Verified Stats Row */}
        <div className="hero-stats animate-in animate-in-delay-4">
          {stats.map(({ value, label, icon: Icon, color }) => (
            <div key={label} className="hero-stat glass-card">
              <div className="hero-stat-icon" style={{ color, borderColor: `${color}33`, background: `${color}12` }}>
                <Icon size={20} />
              </div>
              <div className="stat-number" style={{ color: color === '#10b981' ? '#34d399' : (color === '#c084fc' ? '#c084fc' : '#f8fafc') }}>
                {value}
              </div>
              <div className="stat-label">{label}</div>
            </div>
          ))}
        </div>
      </div>
    </section>
  );
}
