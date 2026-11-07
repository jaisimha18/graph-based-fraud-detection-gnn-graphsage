import React, { useState, useRef, useEffect } from 'react';
import ForceGraph2D from 'react-force-graph-2d';
import { Upload, AlertTriangle, CheckCircle, Activity, Repeat, Eye, EyeOff, ShieldAlert, Info } from 'lucide-react';
import './TransactionAnalysis.css';

export default function TransactionAnalysis() {
  const [file, setFile] = useState(null);
  const [loading, setLoading] = useState(false);
  const [graphData, setGraphData] = useState(null);
  const [stats, setStats] = useState(null);
  const [cycleData, setCycleData] = useState(null);
  const [error, setError] = useState(null);
  const [dimensions, setDimensions] = useState({ width: 800, height: 500 });
  const [customThreshold, setCustomThreshold] = useState(0.5);
  const [highlightCyclesOnly, setHighlightCyclesOnly] = useState(false);
  const [selectedItem, setSelectedItem] = useState(null);
  const containerRef = useRef(null);

  useEffect(() => {
    if (containerRef.current) {
      setDimensions({
        width: containerRef.current.offsetWidth,
        height: 500
      });
    }
    
    const handleResize = () => {
      if (containerRef.current) {
        setDimensions({
          width: containerRef.current.offsetWidth,
          height: 500
        });
      }
    };
    
    window.addEventListener('resize', handleResize);
    return () => window.removeEventListener('resize', handleResize);
  }, [graphData]);

  const handleFileChange = (e) => {
    if (e.target.files && e.target.files[0]) {
      setFile(e.target.files[0]);
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
      const API_BASE = import.meta.env.VITE_API_URL || 'http://localhost:8000';
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

      setGraphData({
        nodes: data.nodes,
        links: data.links
      });

      setCustomThreshold(data.threshold || 0.60);
      
      const fraudCount = data.links.filter(l => l.prediction > data.threshold).length;
      setStats({
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

    } catch (err) {
      console.error(err);
      setError("Failed to connect to the backend API. Make sure the FastAPI server is running on port 8000.");
    } finally {
      setLoading(false);
    }
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

  // Get edge color based on risk & cycle membership
  const getLinkColor = (link) => {
    const isFraud = link.prediction > customThreshold;
    const isCycle = link.inCycle;

    if (highlightCyclesOnly) {
      if (isCycle && isFraud) return '#f43f5e'; // Bright Rose/Magenta for laundering cycle
      if (isCycle) return '#c084fc'; // Purple for topological cycle
      return 'rgba(148, 163, 184, 0.08)'; // Dimmed
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
    if (isCycle && isFraud) return 3;
    if (isCycle || isFraud) return 2.2;
    return 1;
  };

  return (
    <section id="transaction-analysis" className="transaction-analysis-section section-bg">
      <div className="container">
        <div className="section-header-row">
          <div>
            <h2 className="section-title">Interactive Transaction & Cycle Analysis</h2>
            <p className="section-subtitle">
              Upload a transaction CSV to visualize graph topology, run inductive inference, and automatically detect circular money laundering schemes.
            </p>
          </div>
          <span className="badge badge-purple-glow">
            <Repeat size={14} /> Cycle-Aware Attention Active
          </span>
        </div>
        
        <div className="analysis-layout">
          {/* Sidebar / Controls */}
          <div className="analysis-sidebar card">
            <h3>Upload Data</h3>
            <p className="sidebar-desc">
              Upload a transaction CSV (e.g. <code>sample_test_cycle_ring.csv</code> or <code>sample_trans.csv</code>).
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
                <span>{file ? file.name : "Choose CSV file..."}</span>
              </label>
            </div>

            <button 
              className="btn btn-primary analyze-btn" 
              onClick={handleAnalyze} 
              disabled={!file || loading}
            >
              {loading ? "Analyzing Topology & Cycles..." : "Analyze Transactions"}
            </button>

            {error && (
              <div className="error-message">
                <AlertTriangle size={18} />
                <span>{error}</span>
              </div>
            )}

            {stats && (
              <div className="stats-box">
                <h4>Inference Results</h4>
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

                {/* Cycle Detection Banner */}
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
                
                {/* Threshold Control */}
                <div className="threshold-control">
                  <div className="stat-row">
                    <span className="stat-label" style={{ color: '#f8fafc', fontWeight: 500 }}>Detection Threshold:</span>
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
                </div>
              </div>
            )}
          </div>

          {/* Graph Visualization Container */}
          <div className="analysis-graph-container card" ref={containerRef}>
            {!graphData && !loading && (
              <div className="empty-graph-state">
                <Activity size={48} className="empty-icon" />
                <p>Upload a transaction CSV and click "Analyze Transactions" to inspect the graph.</p>
                <p className="empty-hint">Try with <code>data/samples/sample_test_cycle_ring.csv</code> to view a 4-hop circular laundering ring!</p>
              </div>
            )}
            
            {loading && (
              <div className="empty-graph-state">
                <div className="spinner"></div>
                <p>Computing random-walk cycle features & running Cycle-Aware GraphSAGE...</p>
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
                    <span className="legend-color edge-cycle"></span> 🔄 Laundering Cycle ($A \to B \to A$)
                  </div>
                </div>

                <ForceGraph2D
                  width={dimensions.width}
                  height={dimensions.height}
                  graphData={graphData}
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
                      <strong>{selectedItem.type === 'node' ? 'Account Node' : 'Transaction Transfer'}</strong>
                      <button onClick={() => setSelectedItem(null)} className="popover-close">✕</button>
                    </div>
                    {selectedItem.type === 'node' ? (
                      <div className="popover-body">
                        <div><strong>Account ID:</strong> <code>{selectedItem.data.id}</code></div>
                        <div><strong>Role:</strong> {cycleData?.detectedRings?.some(r => r.path.includes(selectedItem.data.id)) ? '🚨 Participates in Circular Laundering Ring' : 'Standard Transactor'}</div>
                      </div>
                    ) : (
                      <div className="popover-body">
                        <div><strong>Route:</strong> <code>{selectedItem.data.source} ➔ {selectedItem.data.target}</code></div>
                        <div><strong>Amount Paid:</strong> ${selectedItem.data.amount?.toLocaleString()} {selectedItem.data.currency}</div>
                        <div><strong>Fraud Score:</strong> {(selectedItem.data.prediction * 100).toFixed(1)}% ({selectedItem.data.prediction > customThreshold ? '🚨 High Risk' : '✅ Normal'})</div>
                        <div><strong>Part of Cycle:</strong> {selectedItem.data.inCycle ? `🔄 Yes (${selectedItem.data.cycleHop}-Hop Loop, $R_k$=${selectedItem.data.cycleProb})` : 'No'}</div>
                        <div><strong>Timestamp:</strong> {selectedItem.data.timestamp}</div>
                      </div>
                    )}
                  </div>
                )}
              </>
            )}
          </div>
        </div>

        {/* Detected Laundering Rings Breakdown */}
        {cycleData && cycleData.detectedRings && cycleData.detectedRings.length > 0 && (
          <div className="detected-rings-section glass-card">
            <div className="rings-header">
              <ShieldAlert className="text-rose" size={24} />
              <div>
                <h3>Forensic Investigation: Detected Closed Laundering Rings ({cycleData.totalRingsFound})</h3>
                <p className="rings-subtitle">
                  The model identified the following closed transaction cycles where funds circulate through intermediaries back to the beneficial owner.
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
                    <p className="ring-path-label">Flow Topology:</p>
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
                      <span className="subtle-label">Flagged Edges:</span>
                      <strong>{ring.flaggedEdges} / {ring.totalEdges} ({(ring.flaggedEdges / Math.max(ring.totalEdges, 1) * 100).toFixed(0)}%)</strong>
                    </div>
                    <div>
                      <span className="subtle-label">Avg Fraud Score:</span>
                      <strong className="text-rose">{(ring.avgPrediction * 100).toFixed(1)}%</strong>
                    </div>
                    <div>
                      <span className="subtle-label">Status:</span>
                      <strong className="text-emerald">Identified by Cycle-Aware GNN</strong>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}
      </div>
    </section>
  );
}