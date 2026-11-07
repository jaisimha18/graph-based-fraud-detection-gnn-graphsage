import React, { useState } from 'react';
import Hero from './components/Hero';
import ProblemStatement from './components/ProblemStatement';
import ArchitectureFlow from './components/ArchitectureFlow';
import TransactionAnalysis from './components/TransactionAnalysis';
import ResearchNovelty from './components/ResearchNovelty';
import TypologyBreakdown from './components/TypologyBreakdown';
import Methodology from './components/Methodology';
import Insights from './components/Insights';
import ConclusionLimitations from './components/ConclusionLimitations';
import GuidedTour from './components/GuidedTour';
import GlossaryModal from './components/GlossaryModal';
import { BookOpen, Compass, Shield } from 'lucide-react';
import './App.css';

export default function App() {
  const [isTourOpen, setIsTourOpen] = useState(false);
  const [isGlossaryOpen, setIsGlossaryOpen] = useState(false);

  return (
    <div className="app">
      {/* Navigation */}
      <nav className="navbar">
        <div className="container nav-container">
          <a href="#hero" className="nav-logo">
            <Shield className="nav-logo-icon" size={20} />
            <span className="nav-logo-text">AML GraphSAGE</span>
            <span className="badge badge-purple nav-tag">Research Demo</span>
          </a>

          <div className="nav-links">
            <a href="#problem">Problem</a>
            <a href="#architecture">Architecture</a>
            <a href="#transaction-analysis" className="nav-link-highlight">Live Demo</a>
            <a href="#research-novelty">Novelty</a>
            <a href="#typologies">Typologies</a>
            <a href="#methodology">Pipeline</a>
            <a href="#insights">EDA Insights</a>
            <a href="#conclusion">Conclusion</a>
          </div>

          <div className="nav-actions">
            <button
              className="nav-btn nav-btn-glossary"
              onClick={() => setIsGlossaryOpen(true)}
              title="Searchable Research Glossary & Mathematics"
            >
              <BookOpen size={15} />
              <span>Glossary</span>
            </button>
            <button
              className="nav-btn nav-btn-tour"
              onClick={() => setIsTourOpen(true)}
              title="Start Interactive Guided Tour"
            >
              <Compass size={15} />
              <span>Guided Tour</span>
            </button>
          </div>
        </div>
      </nav>

      {/* Main Research Platform Sections */}
      <main>
        {/* Module 1: Hero */}
        <Hero
          onStartTour={() => setIsTourOpen(true)}
          onOpenGlossary={() => setIsGlossaryOpen(true)}
        />
        <hr className="section-divider" />

        {/* Module 2: The Problem (Single Tx vs Circular Loops) */}
        <ProblemStatement />
        <hr className="section-divider" />

        {/* Module 3: Visual Research Architecture (5-Stage Interactive Pipeline) */}
        <ArchitectureFlow />
        <hr className="section-divider" />

        {/* Module 4 & 5: Live Demonstration & Ring Extraction */}
        <TransactionAnalysis />
        <hr className="section-divider" />

        {/* Module 6: Research Novelty (Why Cycle Attention? 3-Stage Ablation) */}
        <ResearchNovelty />
        <hr className="section-divider" />

        {/* Module 7: Typology-Based Detection Performance */}
        <TypologyBreakdown />
        <hr className="section-divider" />

        {/* Module 8: Methodology Pipeline */}
        <Methodology />
        <hr className="section-divider" />

        {/* Module 9: Verified EDA Data Discoveries */}
        <Insights />
        <hr className="section-divider" />

        {/* Module 10: Research Synthesis, Scientific Limitations & Future Scope */}
        <ConclusionLimitations />
      </main>

      {/* Modals */}
      <GuidedTour
        isOpen={isTourOpen}
        onClose={() => setIsTourOpen(false)}
      />
      <GlossaryModal
        isOpen={isGlossaryOpen}
        onClose={() => setIsGlossaryOpen(false)}
      />

      {/* Footer */}
      <footer className="footer">
        <div className="container footer-content">
          <div className="footer-brand">
            <span className="nav-logo-text">◈ Cycle-Aware Anti-Money Laundering Detection with GraphSAGE</span>
            <p className="footer-sub">
              University Final-Year Research Project · Supervised by Computer Science Faculty · 2026
            </p>
          </div>
          <div className="footer-meta">
            <span className="tech-pill">PyTorch Geometric 2.6.1</span>
            <span className="tech-pill">FastAPI 0.115</span>
            <span className="tech-pill">IBM AML HI-Small Benchmark</span>
          </div>
        </div>
      </footer>
    </div>
  );
}
