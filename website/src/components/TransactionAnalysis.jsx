import React, { useState, useRef, useEffect } from 'react';
import ForceGraph2D from 'react-force-graph-2d';
import { Upload, AlertTriangle, CheckCircle, Activity } from 'lucide-react';
import './TransactionAnalysis.css';

export default function TransactionAnalysis() {
  const [file, setFile] = useState(null);
  const [loading, setLoading] = useState(false);
  const [graphData, setGraphData] = useState(null);
  const [stats, setStats] = useState(null);
  const [error, setError] = useState(null);
  const [dimensions, setDimensions] = useState({ width: 800, height: 500 });
  const [customThreshold, setCustomThreshold] = useState(0.5);
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

    const formData = new FormData();
    formData.append('file', file);

    try {
      // Assuming FastAPI is running on localhost:8000
      const response = await fetch('http://localhost:8000/api/predict', {
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

      setCustomThreshold(data.threshold);
      
      const fraudCount = data.links.filter(l => l.prediction > data.threshold).length;
      setStats({
        totalTransactions: data.links.length,
        fraudTransactions: fraudCount,
        actualFraudTransactions: data.actualFraudCount || 0,
        totalAccounts: data.nodes.length,
        threshold: data.threshold
      });

    } catch (err) {
      console.error(err);
      setError("Failed to connect to the backend API. Make sure the FastAPI server is running on port 8000.");
    } finally {
      setLoading(false);
    }
  };

  // Node drawing (circles)
  const paintNode = (node, ctx, globalScale) => {
    const label = node.id;
    const fontSize = 12 / globalScale;
    ctx.font = `${fontSize}px Sans-Serif`;
    ctx.fillStyle = '#3b82f6'; // blue
    
    ctx.beginPath();
    ctx.arc(node.x, node.y, 5, 0, 2 * Math.PI, false);
    ctx.fill();
    
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.fillStyle = '#f8fafc';
    ctx.fillText(label, node.x, node.y + 8);
  };

  return (
    <section id="transaction-analysis" className="transaction-analysis-section section-bg">
      <div className="container">
        <h2 className="section-title">Interactive Transaction Analysis</h2>
        <p className="section-subtitle">Upload a transactions CSV to visualize the network and run the GraphSAGE model.</p>
        
        <div className="analysis-layout">
          {/* Sidebar / Controls */}
          <div className="analysis-sidebar card">
            <h3>Upload Data</h3>
            <p className="sidebar-desc">
              Provide a standard CSV containing transaction records. Ensure columns like <code>Timestamp</code>, <code>Account</code>, <code>Account.1</code>, and <code>Amount Paid</code> exist.
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
              {loading ? "Analyzing..." : "Analyze Transactions"}
            </button>

            {error && (
              <div className="error-message">
                <AlertTriangle size={18} />
                <span>{error}</span>
              </div>
            )}

            {stats && (
              <div className="stats-box">
                <h4>Analysis Results</h4>
                <div className="stat-row">
                  <span className="stat-label">Total Accounts:</span>
                  <span className="stat-value">{stats.totalAccounts}</span>
                </div>
                <div className="stat-row">
                  <span className="stat-label">Total Transactions:</span>
                  <span className="stat-value">{stats.totalTransactions}</span>
                </div>
                <div className="stat-row highlight-danger">
                  <span className="stat-label">Flagged as Fraud:</span>
                  <span className="stat-value">
                    {graphData.links.filter(l => l.prediction > customThreshold).length}
                  </span>
                </div>
                <div className="stat-row">
                  <span className="stat-label">Actual Fraud (In Data):</span>
                  <span className="stat-value">{stats.actualFraudTransactions}</span>
                </div>
                
                <div className="threshold-control" style={{ marginTop: '1rem', paddingTop: '1rem', borderTop: '1px solid rgba(255,255,255,0.1)' }}>
                  <div className="stat-row">
                    <span className="stat-label" style={{ color: '#f8fafc', fontWeight: 500 }}>Adjust Threshold:</span>
                    <span className="stat-value" style={{ color: '#3b82f6' }}>{customThreshold.toFixed(2)}</span>
                  </div>
                  <input 
                    type="range" 
                    min="0" 
                    max="1" 
                    step="0.01" 
                    value={customThreshold} 
                    onChange={(e) => setCustomThreshold(parseFloat(e.target.value))}
                    style={{ width: '100%', marginTop: '0.5rem', cursor: 'pointer' }}
                  />
                  <p style={{ fontSize: '0.75rem', color: '#94a3b8', marginTop: '0.5rem', lineHeight: 1.3 }}>
                    Lowering the threshold will flag more transactions. The original optimized threshold was {stats.threshold.toFixed(2)}.
                  </p>
                </div>
              </div>
            )}
          </div>

          {/* Graph Visualization */}
          <div className="analysis-graph-container card" ref={containerRef}>
            {!graphData && !loading && (
              <div className="empty-graph-state">
                <Activity size={48} className="empty-icon" />
                <p>Upload a file and run the analysis to view the graph.</p>
              </div>
            )}
            
            {loading && (
              <div className="empty-graph-state">
                <div className="spinner"></div>
                <p>Processing transactions and running GraphSAGE inference...</p>
              </div>
            )}
            
            {graphData && (
              <>
                <div className="graph-legend">
                  <div className="legend-item">
                    <span className="legend-color node-color"></span> Account
                  </div>
                  <div className="legend-item">
                    <span className="legend-color edge-legit"></span> Legit Transaction
                  </div>
                  <div className="legend-item">
                    <span className="legend-color edge-fraud"></span> High-Risk (Fraud)
                  </div>
                </div>
                <ForceGraph2D
                  width={dimensions.width}
                  height={dimensions.height}
                  graphData={graphData}
                  nodeCanvasObject={paintNode}
                  linkColor={link => link.prediction > customThreshold ? '#ef4444' : '#94a3b8'}
                  linkWidth={link => link.prediction > customThreshold ? 2 : 1}
                  linkDirectionalArrowLength={3.5}
                  linkDirectionalArrowRelPos={1}
                  backgroundColor="#0f172a"
                  onNodeClick={node => console.log('Clicked node', node)}
                  onLinkClick={link => console.log('Clicked link', link)}
                />
              </>
            )}
          </div>
        </div>
      </div>
    </section>
  );
}
