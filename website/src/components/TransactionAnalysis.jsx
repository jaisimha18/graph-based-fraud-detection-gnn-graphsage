import React, { useState, useRef, useEffect } from 'react';
import ForceGraph2D from 'react-force-graph-2d';
import { 
  Upload, AlertTriangle, CheckCircle, Activity, Repeat, Eye, EyeOff, 
  ShieldAlert, Info, Download, Play, RefreshCw, FileText, Sparkles, Filter 
} from 'lucide-react';
import './TransactionAnalysis.css';

const PRELOADED_SCENARIOS = [
  {
    id: 'sample_test_cycle_ring.csv',
    name: '4-Hop Circular Laundering Ring',
    desc: '4 accounts in a closed loop (A ➔ B ➔ C ➔ D ➔ A) circulating $894K across 4 banks.',
    tag: 'Closed Cycle',
    color: '#c084fc',
    recommendedThreshold: 0.60
  },
  {
    id: 'sample_trans.csv',
    name: 'Multi-Bank Transaction Stream',
    desc: '167 streaming transfers across real commercial banks containing benign & anomalous flows.',
    tag: 'Production Stream',
    color: '#38bdf8',
    recommendedThreshold: 0.95
  },
  {
    id: 'sample_trans_4_with_fraud.csv',
    name: 'Structuring & Smurfing Scheme',
    desc: '540 transfers showing structured fragmentation designed to stay below single-transaction reporting limits.',
    tag: 'Smurfing & Layering',
    color: '#fb7185',
    recommendedThreshold: 0.70
  },
  {
    id: 'sample_trans_5_subgraph.csv',
    name: 'Dense Multi-Account Subgraph',
    desc: '115 interconnected transactions forming a dense cluster of interacting counterparties.',
    tag: 'Dense Subgraph',
    color: '#34d399',
    recommendedThreshold: 0.75
  }
];

export default function TransactionAnalysis() {
  const [file, setFile] = useState(null);
  const [loading, setLoading] = useState(false);
  const [activeScenario, setActiveScenario] = useState('sample_test_cycle_ring.csv');
  const [graphData, setGraphData] = useState(null);
  const [stats, setStats] = useState(null);
  const [cycleData, setCycleData] = useState(null);
  const [error, setError] = useState(null);
  const [dimensions, setDimensions] = useState({ width: 800, height: 520 });
  const [customThreshold, setCustomThreshold] = useState(0.60);
  const [highlightCyclesOnly, setHighlightCyclesOnly] = useState(false);
  const [hideLegit, setHideLegit] = useState(false);
  const [hideFraud, setHideFraud] = useState(false);
  const [selectedItem, setSelectedItem] = useState(null);
  const [modelInfo, setModelInfo] = useState({ active_model: 'cycle_attn', status: 'ready', threshold: 0.95 });
  const containerRef = useRef(null);
  const graphRef = useRef(null);

  const API_BASE = import.meta.env.VITE_API_URL || 'http://localhost:8000';

  useEffect(() => {
    // Fetch active model info on mount
    fetch(`${API_BASE}/api/model_info`)
      .then(res => res.json())
      .then(data => {
        if (data.active_model) {
          setModelInfo({
            active_model: data.active_model,
            status: 'Loaded & Active',
            threshold: data.threshold || 0.95,
            device: data.device || 'CPU'
          });
        }
      })
      .catch(() => {
        // Fallback default info
        setModelInfo({ active_model: 'cycle_attn', status: 'Ready (Local Mode)', threshold: 0.95, device: 'PyTorch' });
      });
  }, [API_BASE]);

  useEffect(() => {
    if (containerRef.current) {
      setDimensions({
        width: containerRef.current.offsetWidth,
        height: 520
      });
    }
    
    const handleResize = () => {
      if (containerRef.current) {
        setDimensions({
          width: containerRef.current.offsetWidth,
          height: 520
        });
      }
    };
    
    window.addEventListener('resize', handleResize);
    return () => window.removeEventListener('resize', handleResize);
  }, [graphData]);

  const handleFileChange = (e) => {
    if (e.target.files && e.target.files[0]) {
      setFile(e.target.files[0]);
      setActiveScenario(null);
    }
  };

  const handleAnalyze = async () => {
    if (!file) return;

    setLoading(true);
    setError(null);
    setGraphData(null);
    setCycleData(null);
    setSelectedItem(null);

    const formData = new FormData();
    formData.append('file', file);

    try {
      const response = await fetch(`${API_BASE}/api/predict`, {
        method: 'POST',
        body: formData,
      });

      const data = await response.json();

      if (data.error) {
        setError(data.error);
        setLoading(false);
        return;
      }

      applyInferenceResults(data, file.name);

    } catch (err) {
      console.error(err);
      setError("Failed to connect to the backend API. Make sure the FastAPI server is running on port 8000.");
    } finally {
      setLoading(false);
    }
  };

  const handleLoadScenario = async (scenario) => {
    setActiveScenario(scenario.id);
    setFile(null);
    setLoading(true);
    setError(null);
    setGraphData(null);
    setCycleData(null);
    setSelectedItem(null);

    try {
      const response = await fetch(`${API_BASE}/api/predict_sample/${scenario.id}`, {
        method: 'POST'
      });

      const data = await response.json();

      if (data.error) {
        setError(data.error);
        setLoading(false);
        return;
      }

      applyInferenceResults(data, scenario.name, scenario.recommendedThreshold);

    } catch (err) {
      console.error(err);
      setError("Failed to connect to the backend API. Make sure the FastAPI server is running on port 8000.");
    } finally {
      setLoading(false);
    }
  };

  const applyInferenceResults = (data, sourceName, suggestedThresh = null) => {
    setGraphData({
      nodes: data.nodes,
      links: data.links
    });

    const initThresh = suggestedThresh !== null ? suggestedThresh : (data.threshold || 0.60);
    setCustomThreshold(initThresh);
    
    const fraudCount = data.links.filter(l => l.prediction > initThresh).length;
    setStats({
      sourceName,
      totalTransactions: data.links.length,
      fraudTransactions: fraudCount,
      actualFraudTransactions: data.actualFraudCount || 0,
      totalAccounts: data.nodes.length,
      threshold: data.threshold,
      modelUsed: data.modelUsed || "cycle_attn"
    });

    if (data.cycleAnalysis) {
      setCycleData(data.cycleAnalysis);
    }

    setTimeout(() => {
      if (graphRef.current) {
        graphRef.current.zoomToFit(400, 40);
      }
    }, 200);
  };

  const handleDownloadSample = (scenarioId) => {
    window.open(`${API_BASE}/api/sample/${scenarioId}/download`, '_blank');
  };

  const handleResetZoom = () => {
    if (graphRef.current) {
      graphRef.current.zoomToFit(400, 40);
    }
  };

  const handleDownloadReport = () => {
    if (!stats || !graphData) return;

    const flaggedLinks = graphData.links.filter(l => l.prediction > customThreshold);
    const reportText = `================================================================================
           CYCLE-AWARE AML FORENSIC INVESTIGATION REPORT
================================================================================
Analysis Date       : ${new Date().toLocaleString()}
Transaction Source  : ${stats.sourceName}
Model Architecture  : Cycle-Aware GraphSAGE (Attention Active)
Decision Threshold  : ${customThreshold.toFixed(2)}
Total Accounts      : ${stats.totalAccounts}
Total Transfers     : ${stats.totalTransactions}
Flagged Transfers   : ${flaggedLinks.length} (${((flaggedLinks.length / Math.max(stats.totalTransactions, 1)) * 100).toFixed(1)}%)
Actual Fraud Count  : ${stats.actualFraudTransactions}
Closed Rings Found  : ${cycleData?.totalRingsFound || 0}
Cycle Edges (R_k>0) : ${cycleData?.totalCycleEdges || 0}

--------------------------------------------------------------------------------
DETECTED CIRCULAR MONEY-LAUNDERING RINGS
--------------------------------------------------------------------------------
${cycleData?.detectedRings?.length > 0 ? cycleData.detectedRings.map(r => `
[Ring #${r.ringId}] ${r.hops}-Hop Closed Cycle
Path               : ${r.pathDisplay}
Circulated Capital : $${r.totalAmount?.toLocaleString()} ${r.currency}
Flagged Hops       : ${r.flaggedEdges} / ${r.totalEdges} (${((r.flaggedEdges / Math.max(r.totalEdges, 1)) * 100).toFixed(0)}%)
Average Risk Score : ${(r.avgPrediction * 100).toFixed(1)}%
Status             : CONFIRMED CIRCULAR EVASION PATTERN
`).join('\n') : 'No circular laundering rings detected in this batch.'}

--------------------------------------------------------------------------------
FLAGGED HIGH-RISK TRANSFERS (SAMPLE LIST)
--------------------------------------------------------------------------------
${flaggedLinks.slice(0, 50).map((l, i) => `
#${i + 1} Route: ${l.source} ➔ ${l.target}
   Amount: $${l.amount?.toLocaleString()} ${l.currency}
   Risk Score: ${(l.prediction * 100).toFixed(1)}%
   In Cycle?: ${l.inCycle ? `YES (${l.cycleHop}-hop, R_k=${l.cycleProb})` : 'NO'}
   Timestamp: ${l.timestamp}
`).join('')}

================================================================================
Report generated by Cycle-Aware GraphSAGE Financial Forensics Platform.
================================================================================
`;

    const blob = new Blob([reportText], { type: 'text/plain;charset=utf-8' });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.download = `AML_Forensic_Report_${stats.sourceName.replace(/[^a-z0-9]/gi, '_')}.txt`;
    link.click();
    URL.revokeObjectURL(url);
  };

  // Node drawing (circles + halo for nodes in cycles)
  const paintNode = (node, ctx, globalScale) => {
    const label = node.id;
    const fontSize = 11 / globalScale;
    
    // Check if node is part of any detected cycle ring
    const inRing = cycleData?.detectedRings?.some(r => r.path.includes(node.id));

    if (inRing) {
      // Draw outer glowing halo
      ctx.beginPath();
      ctx.arc(node.x, node.y, 8, 0, 2 * Math.PI, false);
      ctx.fillStyle = 'rgba(192, 132, 252, 0.35)'; // purple glow
      ctx.fill();
      ctx.lineWidth = 1.5;
      ctx.strokeStyle = '#c084fc';
      ctx.stroke();

      // Node center
      ctx.beginPath();
      ctx.arc(node.x, node.y, 5, 0, 2 * Math.PI, false);
      ctx.fillStyle = '#a855f7';
      ctx.fill();
    } else {
      ctx.beginPath();
      ctx.arc(node.x, node.y, 5, 0, 2 * Math.PI, false);
      ctx.fillStyle = '#3b82f6'; // blue
      ctx.fill();
    }
    
    ctx.font = `${fontSize}px Sans-Serif`;
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.fillStyle = '#f8fafc';
    ctx.fillText(label, node.x, node.y + 9);
  };

  // Filter links for display
  const getFilteredLinks = () => {
    if (!graphData) return [];
    return graphData.links.filter(link => {
      const isFraud = link.prediction > customThreshold;
      if (hideLegit && !isFraud) return false;
      if (hideFraud && isFraud) return false;
      return true;
    });
  };

  const getLinkColor = (link) => {
    const isFraud = link.prediction > customThreshold;
    const isCycle = link.inCycle;

    if (highlightCyclesOnly) {
      if (isCycle && isFraud) return '#f43f5e'; // Bright Rose/Magenta for laundering cycle
      if (isCycle) return '#c084fc'; // Purple for topological cycle
      return 'rgba(148, 163, 184, 0.06)'; // Heavily Dimmed
    }

    if (isCycle && isFraud) {
      return '#f43f5e'; // Vibrant Rose: Closed Laundering Cycle
    }
    if (isCycle) {
      return '#c084fc'; // Purple: Structural Cycle
    }
    if (isFraud) {
      return '#ef4444'; // Red: High-risk Fraud
    }
    return '#64748b'; // Slate: Legitimate
  };

  const getLinkWidth = (link) => {
    const isFraud = link.prediction > customThreshold;
    const isCycle = link.inCycle;
    if (isCycle && isFraud) return 3.2;
    if (isCycle || isFraud) return 2.2;
    return 1;
  };

  return (
    <section id="transaction-analysis" className="transaction-analysis-section section-bg">
      <div className="container">
        <div className="section-header-row">
          <div>
            <h2 className="section-title">Interactive Transaction &amp; Cycle Analysis</h2>
            <p className="section-subtitle">
              Upload custom banking logs or 1-click test pre-loaded forensic crime scenarios to run real-time inductive GraphSAGE inference.
            </p>
          </div>
          <div className="model-status-chip">
            <span className="live-dot"></span>
            <span>Model: <strong>Cycle-Aware GraphSAGE (Attention Active)</strong></span>
          </div>
        </div>

        {/* Pre-loaded Sample Scenarios 1-Click Bar */}
        <div className="scenario-selector-card glass-card">
          <div className="scenario-bar-header">
            <div className="scenario-bar-title">
              <Sparkles size={16} className="text-purple" />
              <span>1-Click Forensic Scenarios (IBM HI-Small AML Data):</span>
            </div>
            <span className="scenario-bar-hint">Click any scenario to instantly run the neural network</span>
          </div>

          <div className="scenario-buttons-grid">
            {PRELOADED_SCENARIOS.map(sc => (
              <div
                key={sc.id}
                className={`scenario-btn-box ${activeScenario === sc.id ? 'active' : ''}`}
                onClick={() => handleLoadScenario(sc)}
                style={{ '--sc-color': sc.color }}
              >
                <div className="sc-box-top">
                  <span className="sc-tag" style={{ color: sc.color }}>{sc.tag}</span>
                  <button
                    className="sc-dl-icon"
                    onClick={(e) => { e.stopPropagation(); handleDownloadSample(sc.id); }}
                    title="Download raw CSV file"
                  >
                    <Download size={13} />
                  </button>
                </div>
                <strong className="sc-name">{sc.name}</strong>
                <p className="sc-desc">{sc.desc}</p>
              </div>
            ))}
          </div>
        </div>
        
        <div className="analysis-layout">
          {/* Sidebar / Controls */}
          <div className="analysis-sidebar card">
            <h3>Custom CSV Ingestion</h3>
            <p className="sidebar-desc">
              Upload your own transaction CSV. Required columns: <code>Timestamp</code>, <code>From Bank</code>, <code>Account</code>, <code>To Bank</code>, <code>Account.1</code>, <code>Amount Paid</code>, <code>Payment Format</code>.
            </p>
            
            <div className="upload-zone">
              <input 
                type="file" 
                accept=".csv" 
                id="file-upload" 
                onChange={handleFileChange}
                className="file-input"
              />
              <label htmlFor="file-upload" className="file-label">
                <Upload className="upload-icon" />
                <span>{file ? file.name : "Choose custom CSV file..."}</span>
              </label>
            </div>

            <button 
              className="btn btn-primary analyze-btn" 
              onClick={handleAnalyze} 
              disabled={!file || loading}
            >
              {loading ? "Running Inductive Inference..." : "Analyze Custom CSV"}
            </button>

            {error && (
              <div className="error-message">
                <AlertTriangle size={18} />
                <span>{error}</span>
              </div>
            )}

            {stats && (
              <div className="stats-box">
                <h4>Inference Scorecard</h4>
                <div className="stat-row">
                  <span className="stat-label">Source:</span>
                  <span className="stat-value text-sky" style={{ maxWidth: '170px', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                    {stats.sourceName}
                  </span>
                </div>
                <div className="stat-row">
                  <span className="stat-label">Total Accounts:</span>
                  <span className="stat-value">{stats.totalAccounts}</span>
                </div>
                <div className="stat-row">
                  <span className="stat-label">Total Transfers:</span>
                  <span className="stat-value">{stats.totalTransactions}</span>
                </div>
                <div className="stat-row highlight-danger">
                  <span className="stat-label">Flagged Fraud Edges:</span>
                  <span className="stat-value">
                    {graphData.links.filter(l => l.prediction > customThreshold).length}
                  </span>
                </div>

                {/* Cycle Detection Highlight Box */}
                {cycleData && (
                  <div className="cycle-highlight-card">
                    <div className="cycle-card-header">
                      <Repeat size={16} className="text-purple" />
                      <strong>Laundering Cycle Detection</strong>
                    </div>
                    <div className="stat-row" style={{ marginTop: '0.4rem' }}>
                      <span className="stat-label">Closed Rings Found:</span>
                      <span className="stat-value text-purple">{cycleData.totalRingsFound}</span>
                    </div>
                    <div className="stat-row">
                      <span className="stat-label">Cycle Edges (R_k &gt; 0):</span>
                      <span className="stat-value text-purple">{cycleData.totalCycleEdges}</span>
                    </div>
                    {cycleData.totalRingsFound > 0 && (
                      <p className="cycle-alert-text">
                        ⚠️ <strong>{cycleData.totalRingsFound} circular scheme(s)</strong> detected where funds route through intermediary accounts and return to the origin.
                      </p>
                    )}
                  </div>
                )}
                
                {/* Threshold Controls */}
                <div className="threshold-control">
                  <div className="stat-row">
                    <span className="stat-label" style={{ color: '#f8fafc', fontWeight: 500 }}>Decision Threshold:</span>
                    <span className="stat-value" style={{ color: '#38bdf8', fontWeight: 600 }}>{customThreshold.toFixed(2)}</span>
                  </div>
                  <input 
                    type="range" 
                    min="0" 
                    max="1" 
                    step="0.01" 
                    value={customThreshold} 
                    onChange={(e) => setCustomThreshold(parseFloat(e.target.value))}
                    className="threshold-slider"
                  />
                  
                  <div className="preset-buttons">
                    <button 
                      type="button"
                      onClick={() => setCustomThreshold(0.60)}
                      className={`preset-btn ${customThreshold <= 0.65 ? 'active' : ''}`}
                    >
                      Mini Sample (0.60)
                    </button>
                    <button 
                      type="button"
                      onClick={() => setCustomThreshold(stats.threshold || 0.95)}
                      className={`preset-btn ${customThreshold >= 0.90 ? 'active' : ''}`}
                    >
                      Full Graph ({(stats.threshold || 0.95).toFixed(2)})
                    </button>
                  </div>

                  {/* Toggle Controls */}
                  <div className="graph-filter-toggles">
                    {cycleData && cycleData.totalRingsFound > 0 && (
                      <button 
                        type="button"
                        className={`cycle-toggle-btn ${highlightCyclesOnly ? 'active' : ''}`}
                        onClick={() => setHighlightCyclesOnly(!highlightCyclesOnly)}
                      >
                        {highlightCyclesOnly ? <EyeOff size={14} /> : <Eye size={14} />}
                        <span>{highlightCyclesOnly ? "Show All Transfers" : "Spotlight Circular Cycles Only"}</span>
                      </button>
                    )}

                    <div className="subtle-checkbox-row">
                      <label className="checkbox-label">
                        <input
                          type="checkbox"
                          checked={hideLegit}
                          onChange={(e) => setHideLegit(e.target.checked)}
                        />
                        <span>Hide Legit Transfers</span>
                      </label>
                      <label className="checkbox-label">
                        <input
                          type="checkbox"
                          checked={hideFraud}
                          onChange={(e) => setHideFraud(e.target.checked)}
                        />
                        <span>Hide Flagged Fraud</span>
                      </label>
                    </div>

                    <div className="action-buttons-row">
                      <button type="button" onClick={handleResetZoom} className="btn-action-small">
                        <RefreshCw size={12} /> Reset Zoom
                      </button>
                      <button type="button" onClick={handleDownloadReport} className="btn-action-small highlight">
                        <FileText size={12} /> Export Report
                      </button>
                    </div>
                  </div>
                </div>
              </div>
            )}
          </div>

          {/* Graph Visualization Container */}
          <div className="analysis-graph-container card" ref={containerRef}>
            {!graphData && !loading && (
              <div className="empty-graph-state">
                <Activity size={48} className="empty-icon" />
                <p>Select a 1-click scenario above or upload a CSV to inspect the live transaction graph.</p>
                <p className="empty-hint">Try clicking <strong>"4-Hop Circular Laundering Ring"</strong> above to see closed-loop detection live!</p>
              </div>
            )}
            
            {loading && (
              <div className="empty-graph-state">
                <div className="spinner"></div>
                <p>Computing random-walk cycle features &amp; running Cycle-Aware GraphSAGE...</p>
              </div>
            )}
            
            {graphData && (
              <>
                <div className="graph-legend">
                  <div className="legend-item">
                    <span className="legend-color node-color"></span> Normal Account
                  </div>
                  <div className="legend-item">
                    <span className="legend-color node-cycle-color"></span> Cycle Mule Account
                  </div>
                  <div className="legend-item">
                    <span className="legend-color edge-legit"></span> Legit Transfer
                  </div>
                  <div className="legend-item">
                    <span className="legend-color edge-fraud"></span> High-Risk Transfer
                  </div>
                  <div className="legend-item">
                    <span className="legend-color edge-cycle"></span> 🔄 Laundering Cycle (A ➔ B ➔ A)
                  </div>
                </div>

                <ForceGraph2D
                  ref={graphRef}
                  width={dimensions.width}
                  height={dimensions.height}
                  graphData={{ nodes: graphData.nodes, links: getFilteredLinks() }}
                  nodeCanvasObject={paintNode}
                  linkColor={getLinkColor}
                  linkWidth={getLinkWidth}
                  linkDirectionalArrowLength={4}
                  linkDirectionalArrowRelPos={1}
                  backgroundColor="#0a0f1d"
                  onNodeClick={node => setSelectedItem({ type: 'node', data: node })}
                  onLinkClick={link => setSelectedItem({ type: 'link', data: link })}
                />

                {/* Inspect Selected Element Modal */}
                {selectedItem && (
                  <div className="item-detail-popover">
                    <div className="popover-header">
                      <strong>{selectedItem.type === 'node' ? 'Account Node Details' : 'Transfer Details'}</strong>
                      <button onClick={() => setSelectedItem(null)} className="popover-close">✕</button>
                    </div>
                    {selectedItem.type === 'node' ? (
                      <div className="popover-body">
                        <div><strong>Account ID:</strong> <code>{selectedItem.data.id}</code></div>
                        <div><strong>Role:</strong> {cycleData?.detectedRings?.some(r => r.path.includes(selectedItem.data.id)) ? '🚨 Identified Participant in Closed Laundering Ring' : 'Standard Transactor'}</div>
                      </div>
                    ) : (
                      <div className="popover-body">
                        <div><strong>Route:</strong> <code>{selectedItem.data.source.id || selectedItem.data.source} ➔ {selectedItem.data.target.id || selectedItem.data.target}</code></div>
                        <div><strong>Amount Paid:</strong> ${selectedItem.data.amount?.toLocaleString()} {selectedItem.data.currency}</div>
                        <div><strong>Fraud Score:</strong> {(selectedItem.data.prediction * 100).toFixed(1)}% ({selectedItem.data.prediction > customThreshold ? '🚨 High Risk' : '✅ Normal'})</div>
                        <div><strong>In Laundering Cycle?:</strong> {selectedItem.data.inCycle ? `🔄 YES (${selectedItem.data.cycleHop}-Hop Loop, R_k=${selectedItem.data.cycleProb})` : 'NO'}</div>
                        <div><strong>Timestamp:</strong> {selectedItem.data.timestamp}</div>
                      </div>
                    )}
                  </div>
                )}
              </>
            )}
          </div>
        </div>

        {/* Module 5: Forensic Investigation — Detected Laundering Rings */}
        <div className="forensic-rings-wrapper" id="forensic-investigation">
          {cycleData && cycleData.detectedRings && cycleData.detectedRings.length > 0 ? (
            <div className="detected-rings-section glass-card">
              <div className="rings-header">
                <ShieldAlert className="text-rose" size={26} />
                <div>
                  <h3>Forensic Investigation: Detected Closed Laundering Rings ({cycleData.totalRingsFound})</h3>
                  <p className="rings-subtitle">
                    The model identified the following closed transaction cycles where funds circulate through intermediary accounts across multiple banks and return to the originating beneficial owner.
                  </p>
                </div>
              </div>

              <div className="rings-grid">
                {cycleData.detectedRings.map((ring) => (
                  <div key={ring.ringId} className="ring-card">
                    <div className="ring-card-top">
                      <span className="badge badge-rose">Ring #{ring.ringId}: {ring.hops}-Hop Closed Cycle</span>
                      <span className="ring-amount">${ring.totalAmount.toLocaleString()} {ring.currency}</span>
                    </div>
                    <div className="ring-path-box">
                      <p className="ring-path-label">Flow Topology Sequence:</p>
                      <div className="ring-path-flow">
                        {ring.path.map((acc, idx) => (
                          <span key={idx} className="path-node-step">
                            <code>{acc}</code>
                            {idx < ring.path.length - 1 && <span className="path-arrow">➔</span>}
                          </span>
                        ))}
                      </div>
                    </div>
                    <div className="ring-stats-footer">
                      <div>
                        <span className="subtle-label">Flagged Hops:</span>
                        <strong>{ring.flaggedEdges} / {ring.totalEdges} ({(ring.flaggedEdges / Math.max(ring.totalEdges, 1) * 100).toFixed(0)}%)</strong>
                      </div>
                      <div>
                        <span className="subtle-label">Avg Fraud Score:</span>
                        <strong className="text-rose">{(ring.avgPrediction * 100).toFixed(1)}%</strong>
                      </div>
                      <div>
                        <span className="subtle-label">Forensic Classification:</span>
                        <strong className="text-emerald">Captured by Cycle Attention</strong>
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          ) : (
            <div className="empty-rings-card glass-card">
              <Info size={20} className="text-sky" />
              <div>
                <strong>Forensic Loop Status: No Circular Laundering Rings Detected in Current Selection</strong>
                <p>
                  Transfers in this batch do not form closed directed return loops. Click the <strong>"4-Hop Circular Laundering Ring"</strong> scenario above to observe live ring identification.
                </p>
              </div>
            </div>
          )}
        </div>
      </div>
    </section>
  );
}