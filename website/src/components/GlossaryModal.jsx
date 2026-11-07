import React, { useState } from 'react';
import { X, Search, BookOpen, Sparkles } from 'lucide-react';
import './GlossaryModal.css';

const terms = [
  {
    term: 'Graph Neural Network (GNN)',
    category: 'Architecture',
    definition: 'A class of deep learning models designed to perform inference on data structured as graphs (nodes and edges) via message passing between neighboring entities.'
  },
  {
    term: 'GraphSAGE',
    category: 'Architecture',
    definition: 'Graph Sample and Aggregate. An inductive GNN framework that generates node embeddings by sampling a fixed number of neighbors and aggregating their features, allowing it to generalize to unseen nodes.'
  },
  {
    term: 'Node (Account)',
    category: 'Graph Topology',
    definition: 'Represents an entity in the network. In our financial graph, each node represents an individual, corporate, or crypto bank account (518,573 nodes).'
  },
  {
    term: 'Directed Edge (Transaction)',
    category: 'Graph Topology',
    definition: 'A transfer of capital from a source account to a target account containing attributes such as amount, currency, timestamp, and payment format.'
  },
  {
    term: 'Random Walk Cycle Return Probability (R_k)',
    category: 'Cycle Theory',
    definition: 'The probability that a random walk starting at an edge (u ➔ v) returns to the origin u after k steps, computed as R_k(u,v) = [P^k]_{v,u} where P is the row-stochastic transition matrix.'
  },
  {
    term: 'Structural Attention Mechanism (α_ji)',
    category: 'Innovation',
    definition: 'A learned weighting factor that dynamically assigns higher importance to neighbor messages originating from circular laundering patterns, preventing signal dilution across benign neighbors.'
  },
  {
    term: 'PR-AUC (Precision-Recall Area Under Curve)',
    category: 'Evaluation',
    definition: 'The gold-standard evaluation metric for highly imbalanced datasets (such as fraud detection). Unlike ROC-AUC, PR-AUC evaluates true fraud capture without being inflated by massive negative class sizes.'
  },
  {
    term: 'Full Ring Recall',
    category: 'Evaluation',
    definition: 'The percentage of complete closed circular laundering schemes (100% of constituent transaction hops) successfully identified by the model (85.2% in our proposed architecture).'
  },
  {
    term: '1-Weisfeiler-Lehman (1-WL) Bound',
    category: 'Theory',
    definition: 'A theoretical upper bound on standard message-passing GNN expressiveness, proving that standard GNNs cannot count cycles or distinguish loops from trees without explicit structural augmentation.'
  },
  {
    term: 'Class Imbalance (979:1)',
    category: 'Data',
    definition: 'The severe skew in financial data where legitimate transactions outnumber laundering cases by nearly 1,000 to 1. Handled using pos_weight in Binary Cross Entropy loss.'
  }
];

export default function GlossaryModal({ isOpen, onClose }) {
  const [searchTerm, setSearchTerm] = useState('');

  if (!isOpen) return null;

  const filteredTerms = terms.filter(
    (t) =>
      t.term.toLowerCase().includes(searchTerm.toLowerCase()) ||
      t.definition.toLowerCase().includes(searchTerm.toLowerCase()) ||
      t.category.toLowerCase().includes(searchTerm.toLowerCase())
  );

  return (
    <div className="glossary-backdrop" onClick={onClose}>
      <div className="glossary-modal glass-card" onClick={(e) => e.stopPropagation()}>
        <div className="glossary-modal-header">
          <div className="glossary-title-group">
            <BookOpen size={20} className="text-purple" />
            <h3>Financial Graph AI Research Glossary</h3>
          </div>
          <button onClick={onClose} className="tour-close-btn">
            <X size={20} />
          </button>
        </div>

        <div className="glossary-search-wrapper">
          <Search size={16} className="search-icon" />
          <input
            type="text"
            placeholder="Search concepts (e.g. GraphSAGE, PR-AUC, Attention, R_k)..."
            value={searchTerm}
            onChange={(e) => setSearchTerm(e.target.value)}
            className="glossary-search-input"
            autoFocus
          />
        </div>

        <div className="glossary-terms-list">
          {filteredTerms.length > 0 ? (
            filteredTerms.map((t) => (
              <div key={t.term} className="glossary-term-item">
                <div className="term-item-top">
                  <span className="term-name">{t.term}</span>
                  <span className="term-category-badge">{t.category}</span>
                </div>
                <p className="term-def">{t.definition}</p>
              </div>
            ))
          ) : (
            <p className="no-terms-msg">No matching terminology found.</p>
          )}
        </div>
      </div>
    </div>
  );
}