import Hero from './components/Hero';
import Methodology from './components/Methodology';
import ResultsDashboard from './components/ResultsDashboard';
import Insights from './components/Insights';
import './App.css';

export default function App() {
  return (
    <div className="app">
      {/* Navigation */}
      <nav className="navbar">
        <div className="container nav-container">
          <span className="nav-logo">
            <span className="nav-logo-icon">◈</span> AML GraphSAGE
          </span>
          <div className="nav-links">
            <a href="#methodology">Methodology</a>
            <a href="#results">Results</a>
            <a href="#insights">Insights</a>
          </div>
        </div>
      </nav>

      {/* Main Sections */}
      <Hero />
      <hr className="section-divider" />
      <Methodology />
      <hr className="section-divider" />
      <ResultsDashboard />
      <hr className="section-divider" />
      <Insights />

      {/* Footer */}
      <footer className="footer">
        <div className="container">
          <p className="footer-text">
            Graph-Based Fraud Detection using GNN (GraphSAGE) — Built with PyTorch Geometric
          </p>
          <p className="footer-sub">
            IBM HI-Small Anti-Money Laundering Synthetic Dataset · 2026
          </p>
        </div>
      </footer>
    </div>
  );
}
