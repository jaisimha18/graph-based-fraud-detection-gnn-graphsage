import React, { useState } from 'react';
import { X, ChevronRight, ChevronLeft, Compass, ArrowRight, Check } from 'lucide-react';
import './GuidedTour.css';

const tourSteps = [
  {
    step: 1,
    title: 'Welcome to Cycle-Aware AML GraphSAGE',
    targetId: 'hero',
    badge: 'Step 1 of 6',
    description: 'This platform demonstrates an inductive Graph Neural Network designed to detect multi-hop circular money laundering in banking networks where traditional transaction-monitoring rules fail.',
    actionText: 'Scroll down to see the core problem.'
  },
  {
    step: 2,
    title: 'The Problem: Single Transactions vs Closed Loops',
    targetId: 'problem',
    badge: 'Step 2 of 6',
    description: 'Notice the visual comparison: conventional systems inspect transfers individually (A ➔ B ➔ C ➔ D). Money launderers exploit this by closing the loop back to the originator (A ➔ B ➔ C ➔ D ➔ A).',
    actionText: 'Inspect the 5-stage architecture.'
  },
  {
    step: 3,
    title: 'Visual Research Architecture',
    targetId: 'architecture',
    badge: 'Step 3 of 6',
    description: 'Explore the 5 interactive pipeline stages: from raw transaction ingestion to PyG graph construction, sparse random-walk cycle feature extraction (R_k), and edge-conditioned structural attention.',
    actionText: 'Try the live transaction demonstration.'
  },
  {
    step: 4,
    title: 'Live Demonstration & Ring Extraction',
    targetId: 'transaction-analysis',
    badge: 'Step 4 of 6',
    description: 'Select a pre-loaded sample scenario (or upload your own CSV) and click "Analyze Transactions". The graph will render in 2D with circular laundering rings highlighted in vibrant purple!',
    actionText: 'Understand our core research novelty.'
  },
  {
    step: 5,
    title: 'Research Novelty: Why Cycle Attention?',
    targetId: 'research-novelty',
    badge: 'Step 5 of 6',
    description: 'Review our 3-stage ablation study. Notice how naive addition drops PR-AUC by -65.6% due to signal dilution, while our proposed Cycle Attention mechanism leaps by +78.5% to 0.5759 PR-AUC with 85.2% full ring recall.',
    actionText: 'Examine typology detection rates.'
  },
  {
    step: 6,
    title: 'Typologies, Methodology & Findings',
    targetId: 'typologies',
    badge: 'Step 6 of 6',
    description: 'Review detection recall across all 8 canonical AML typologies, our 5-phase training methodology, and 9 verified forensic findings from the IBM AML dataset.',
    actionText: 'Tour completed!'
  }
];

export default function GuidedTour({ isOpen, onClose }) {
  const [currentStepIndex, setCurrentStepIndex] = useState(0);

  if (!isOpen) return null;

  const currentStep = tourSteps[currentStepIndex];

  const handleNext = () => {
    if (currentStepIndex < tourSteps.length - 1) {
      const nextIndex = currentStepIndex + 1;
      setCurrentStepIndex(nextIndex);
      const targetEl = document.getElementById(tourSteps[nextIndex].targetId);
      if (targetEl) {
        targetEl.scrollIntoView({ behavior: 'smooth' });
      }
    } else {
      onClose();
    }
  };

  const handlePrev = () => {
    if (currentStepIndex > 0) {
      const prevIndex = currentStepIndex - 1;
      setCurrentStepIndex(prevIndex);
      const targetEl = document.getElementById(tourSteps[prevIndex].targetId);
      if (targetEl) {
        targetEl.scrollIntoView({ behavior: 'smooth' });
      }
    }
  };

  const jumpTo = (index) => {
    setCurrentStepIndex(index);
    const targetEl = document.getElementById(tourSteps[index].targetId);
    if (targetEl) {
      targetEl.scrollIntoView({ behavior: 'smooth' });
    }
  };

  return (
    <div className="tour-overlay">
      <div className="tour-modal glass-card">
        <div className="tour-header">
          <div className="tour-header-left">
            <Compass size={18} className="text-purple" />
            <span className="tour-badge">{currentStep.badge}</span>
          </div>
          <button onClick={onClose} className="tour-close-btn">
            <X size={18} />
          </button>
        </div>

        <h3 className="tour-title">{currentStep.title}</h3>
        <p className="tour-description">{currentStep.description}</p>

        <div className="tour-dots-indicator">
          {tourSteps.map((s, idx) => (
            <button
              key={s.step}
              className={`tour-dot ${idx === currentStepIndex ? 'active' : ''}`}
              onClick={() => jumpTo(idx)}
              title={s.title}
            />
          ))}
        </div>

        <div className="tour-footer">
          <button
            onClick={handlePrev}
            disabled={currentStepIndex === 0}
            className="btn btn-secondary tour-btn"
          >
            <ChevronLeft size={16} /> Back
          </button>
          
          <button onClick={handleNext} className="btn btn-primary tour-btn">
            {currentStepIndex === tourSteps.length - 1 ? (
              <>Finish Tour <Check size={16} /></>
            ) : (
              <>Next: {tourSteps[currentStepIndex + 1]?.targetId} <ChevronRight size={16} /></>
            )}
          </button>
        </div>
      </div>
    </div>
  );
}
