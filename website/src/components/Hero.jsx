import { Shield, GitBranch, Activity } from 'lucide-react';
import './Hero.css';

const stats = [
  { value: '518K+', label: 'Accounts Analyzed', icon: Shield },
  { value: '5M+', label: 'Transactions Processed', icon: Activity },
  { value: '979:1', label: 'Class Imbalance Handled', icon: GitBranch },
];

export default function Hero() {
  return (
    <section className="hero">
      {/* Animated background grid */}
      <div className="hero-bg">
        <div className="hero-grid" />
        <div className="hero-glow hero-glow-1" />
        <div className="hero-glow hero-glow-2" />
      </div>

      <div className="container hero-content">
        <div className="hero-badge animate-in">
          <span className="badge badge-blue">
            <Shield size={14} />
            Graph Neural Network Research
          </span>
        </div>

        <h1 className="hero-title animate-in animate-in-delay-1">
          Graph-Based{' '}
          <span className="hero-title-accent">Anti-Money Laundering</span>
          <br />Detection with GraphSAGE
        </h1>

        <p className="hero-description animate-in animate-in-delay-2">
          Leveraging inductive Graph Neural Networks and PyTorch Geometric to detect
          complex, multi-hop money laundering patterns in financial transaction networks
          — going beyond traditional rule-based systems.
        </p>

        <div className="hero-tech-stack animate-in animate-in-delay-3">
          {['PyTorch Geometric', 'GraphSAGE', 'Python', 'IBM AML Dataset'].map((tech) => (
            <span key={tech} className="tech-pill">{tech}</span>
          ))}
        </div>

        <div className="hero-stats animate-in animate-in-delay-4">
          {stats.map(({ value, label, icon: Icon }) => (
            <div key={label} className="hero-stat">
              <div className="hero-stat-icon">
                <Icon size={20} />
              </div>
              <div className="stat-number">{value}</div>
              <div className="stat-label">{label}</div>
            </div>
          ))}
        </div>
      </div>
    </section>
  );
}
