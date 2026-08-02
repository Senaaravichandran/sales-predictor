import { useState, useEffect, useRef } from 'react'
import { 
  Activity, BarChart3, Database, Cpu, UploadCloud, 
  LayoutDashboard, BrainCircuit, TableProperties, CheckCircle2
} from 'lucide-react'
import { 
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Cell 
} from 'recharts'
import axios from 'axios'
import './App.css'

function App() {
  const [activeTab, setActiveTab] = useState('dashboard')
  const [apiStatus, setApiStatus] = useState('Connecting...')
  const [isUploading, setIsUploading] = useState(false)
  const [isTraining, setIsTraining] = useState(false)
  const [datasetStats, setDatasetStats] = useState(null)
  const [trainResults, setTrainResults] = useState(null)
  
  const [targetColumn, setTargetColumn] = useState('')
  const [taskType, setTaskType] = useState('auto')
  
  const fileInputRef = useRef(null)

  useEffect(() => {
    const fetchStatus = async () => {
      try {
        const response = await axios.get('http://localhost:8000/api/status')
        if (response.data.status === 'ok') {
          setApiStatus('Connected')
        } else {
          setApiStatus('Offline')
        }
      } catch (error) {
        setApiStatus('Offline')
      }
    }
    fetchStatus()
  }, [])

  const handleFileUpload = async (event) => {
    const file = event.target.files[0];
    if (!file) return;

    setIsUploading(true);
    setTrainResults(null);
    const formData = new FormData();
    formData.append('file', file);

    try {
      const response = await axios.post('http://localhost:8000/api/upload', formData, {
        headers: { 'Content-Type': 'multipart/form-data' }
      });
      if (response.data.status === 'success') {
        setDatasetStats(response.data.data_summary);
        if (response.data.data_summary.columns.length > 0) {
           setTargetColumn(response.data.data_summary.columns[response.data.data_summary.columns.length - 1]);
        }
      }
    } catch (error) {
      console.error("Error uploading file:", error);
      alert("Failed to upload dataset. Ensure backend is running.");
    } finally {
      setIsUploading(false);
      if (fileInputRef.current) fileInputRef.current.value = '';
    }
  };

  const handleTrainModel = async () => {
    if (!targetColumn) {
      alert("Please select a target column");
      return;
    }
    setIsTraining(true);
    try {
      const response = await axios.post('http://localhost:8000/api/train', {
        target_column: targetColumn,
        task_type: taskType
      });
      if (response.data.status === 'success') {
        setTrainResults(response.data);
      }
    } catch (error) {
      console.error("Error training model:", error);
      alert(error.response?.data?.detail || "Model training failed.");
    } finally {
      setIsTraining(false);
    }
  };

  const triggerFileInput = () => {
    if (fileInputRef.current) {
      fileInputRef.current.click();
    }
  };

  return (
    <div className="app-layout">
      {/* Sidebar */}
      <aside className="sidebar">
        <div className="sidebar-header">
          <Activity className="logo-icon" size={32} />
          <h2 style={{ fontSize: '1.2rem', margin: 0 }}>SaleSense <span style={{ color: 'var(--text-muted)', fontWeight: 400 }}>AI</span></h2>
        </div>
        
        <nav className="sidebar-nav">
          <div 
            className={`nav-item ${activeTab === 'dashboard' ? 'active' : ''}`}
            onClick={() => setActiveTab('dashboard')}
          >
            <LayoutDashboard size={20} />
            Data Explorer
          </div>
          <div 
            className={`nav-item ${activeTab === 'mllab' ? 'active' : ''}`}
            onClick={() => setActiveTab('mllab')}
          >
            <BrainCircuit size={20} />
            ML Studio
          </div>
        </nav>

        <div className="sidebar-footer">
          <div style={{ display: 'flex', flexDirection: 'column', gap: '1rem' }}>
            <span style={{ fontSize: '0.85rem', color: 'var(--text-muted)' }}>
              API Status: 
              <span style={{ color: apiStatus === 'Connected' ? '#10b981' : '#ef4444', fontWeight: 600, marginLeft: '0.5rem' }}>
                {apiStatus}
              </span>
            </span>
            <button className="btn-primary" onClick={triggerFileInput} disabled={isUploading} style={{ width: '100%' }}>
              <UploadCloud size={18} />
              {isUploading ? 'Uploading...' : 'New Dataset'}
            </button>
            <input type="file" accept=".csv" ref={fileInputRef} style={{ display: 'none' }} onChange={handleFileUpload} />
          </div>
        </div>
      </aside>

      {/* Main Content */}
      <main className="main-content">
        <header className="top-nav">
          <h1 className="page-title">{activeTab === 'dashboard' ? 'Data Explorer' : 'Machine Learning Studio'}</h1>
          {datasetStats && (
            <div className="badge numeric" style={{ padding: '0.5rem 1rem', display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
              <TableProperties size={16} />
              Active: {datasetStats.columns.length} columns, {datasetStats.total_rows.toLocaleString()} rows
            </div>
          )}
        </header>

        <div className="content-wrapper animate-fade-in">
          {!datasetStats ? (
            <div className="glass-panel" style={{ textAlign: 'center', padding: '4rem 2rem' }}>
              <Database size={64} style={{ color: 'var(--text-muted)', opacity: 0.5, marginBottom: '1rem' }} />
              <h2>No Dataset Loaded</h2>
              <p style={{ color: 'var(--text-muted)', marginBottom: '2rem' }}>
                Upload any CSV dataset to unlock enterprise-grade automated analytics and machine learning.
              </p>
              <button className="btn-primary" onClick={triggerFileInput}>
                 <UploadCloud size={18} /> Upload CSV Dataset
              </button>
            </div>
          ) : (
            <>
              {activeTab === 'dashboard' && (
                <>
                  <div className="stats-grid">
                    <div className="stat-card glass-panel">
                      <div className="stat-header">Total Rows</div>
                      <div className="stat-value">{datasetStats.total_rows.toLocaleString()}</div>
                    </div>
                    <div className="stat-card glass-panel">
                      <div className="stat-header">Total Columns</div>
                      <div className="stat-value">{datasetStats.total_cols}</div>
                    </div>
                    <div className="stat-card glass-panel">
                      <div className="stat-header">Numeric Features</div>
                      <div className="stat-value">{datasetStats.numeric_cols.length}</div>
                    </div>
                    <div className="stat-card glass-panel">
                      <div className="stat-header">Categorical Features</div>
                      <div className="stat-value">{datasetStats.categorical_cols.length}</div>
                    </div>
                  </div>

                  <div className="glass-panel" style={{ height: '400px', display: 'flex', flexDirection: 'column' }}>
                    <h3 style={{ marginBottom: '1rem' }}>{datasetStats.chart_title}</h3>
                    {datasetStats.chart_data && datasetStats.chart_data.length > 0 ? (
                      <div style={{ flex: 1 }}>
                        <ResponsiveContainer width="100%" height="100%">
                          <BarChart data={datasetStats.chart_data}>
                            <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" vertical={false} />
                            <XAxis dataKey="name" stroke="var(--text-muted)" fontSize={12} tickLine={false} axisLine={false} />
                            <YAxis stroke="var(--text-muted)" fontSize={12} tickLine={false} axisLine={false} />
                            <Tooltip contentStyle={{ backgroundColor: 'rgba(20, 20, 30, 0.9)', borderColor: 'var(--border-color)', borderRadius: '8px', color: 'white' }} />
                            <Bar dataKey="value" fill="var(--accent-1)" radius={[4, 4, 0, 0]}>
                              {datasetStats.chart_data.map((entry, index) => (
                                <Cell key={`cell-${index}`} fill={index % 2 === 0 ? 'var(--accent-1)' : 'var(--accent-2)'} />
                              ))}
                            </Bar>
                          </BarChart>
                        </ResponsiveContainer>
                      </div>
                    ) : (
                      <p style={{ color: 'var(--text-muted)' }}>No distribution data available.</p>
                    )}
                  </div>

                  <div className="glass-panel">
                    <h3 style={{ marginBottom: '1rem' }}>Dataset Schema</h3>
                    <div className="table-container">
                      <table>
                        <thead>
                          <tr>
                            <th>Column Name</th>
                            <th>Data Type</th>
                            <th>Unique Values</th>
                            <th>Missing Values</th>
                          </tr>
                        </thead>
                        <tbody>
                          {datasetStats.column_stats.map((col, idx) => (
                            <tr key={idx}>
                              <td style={{ fontWeight: 500 }}>{col.column}</td>
                              <td><span className={`badge ${col.type}`}>{col.type}</span></td>
                              <td>{col.unique.toLocaleString()}</td>
                              <td style={{ color: col.missing > 0 ? '#ef4444' : 'inherit' }}>{col.missing}</td>
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>
                  </div>
                </>
              )}

              {activeTab === 'mllab' && (
                <div style={{ display: 'flex', gap: '2rem', alignItems: 'flex-start' }}>
                  <div className="glass-panel" style={{ flex: '1', minWidth: '300px' }}>
                    <h3 style={{ marginBottom: '1.5rem', display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                      <Cpu size={24} color="var(--accent-1)" /> Model Configuration
                    </h3>
                    
                    <div className="form-group">
                      <label>Target Variable (to predict)</label>
                      <select value={targetColumn} onChange={(e) => setTargetColumn(e.target.value)}>
                        {datasetStats.columns.map(c => <option key={c} value={c}>{c}</option>)}
                      </select>
                    </div>

                    <div className="form-group">
                      <label>Task Type</label>
                      <select value={taskType} onChange={(e) => setTaskType(e.target.value)}>
                        <option value="auto">Auto-detect (Recommended)</option>
                        <option value="regression">Regression (Continuous values)</option>
                        <option value="classification">Classification (Categories)</option>
                      </select>
                    </div>

                    <button 
                      className="btn-primary" 
                      style={{ width: '100%', marginTop: '1rem' }}
                      onClick={handleTrainModel}
                      disabled={isTraining}
                    >
                      <BrainCircuit size={18} /> 
                      {isTraining ? 'Training Model...' : 'Train Model'}
                    </button>
                  </div>

                  <div className="glass-panel" style={{ flex: '2', minHeight: '350px' }}>
                    {isTraining ? (
                      <div style={{ display: 'flex', flexDirection: 'column', justifyContent: 'center', alignItems: 'center', height: '100%', color: 'var(--text-muted)' }}>
                        <Cpu size={48} className="animate-pulse" style={{ color: 'var(--accent-1)', marginBottom: '1rem' }} />
                        <h3>Training Custom Model...</h3>
                        <p>Running automated feature engineering and ensemble algorithms.</p>
                      </div>
                    ) : trainResults ? (
                      <div className="animate-fade-in">
                        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '2rem' }}>
                          <h3 style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', color: '#10b981' }}>
                            <CheckCircle2 size={24} /> Model Trained Successfully
                          </h3>
                          <span className="badge numeric">{trainResults.task_type} Task</span>
                        </div>

                        <div className="stats-grid" style={{ marginBottom: '2rem' }}>
                          {trainResults.task_type === 'Classification' ? (
                            <div className="stat-card" style={{ background: 'rgba(0,0,0,0.2)', padding: '1.5rem', borderRadius: '8px' }}>
                              <div className="stat-header">Model Accuracy</div>
                              <div className="stat-value" style={{ color: 'var(--accent-1)' }}>{(trainResults.metrics.accuracy * 100).toFixed(2)}%</div>
                            </div>
                          ) : (
                            <>
                              <div className="stat-card" style={{ background: 'rgba(0,0,0,0.2)', padding: '1.5rem', borderRadius: '8px' }}>
                                <div className="stat-header">R² Score (Accuracy)</div>
                                <div className="stat-value" style={{ color: 'var(--accent-1)' }}>{(trainResults.metrics.r2_score * 100).toFixed(2)}%</div>
                              </div>
                              <div className="stat-card" style={{ background: 'rgba(0,0,0,0.2)', padding: '1.5rem', borderRadius: '8px' }}>
                                <div className="stat-header">RMSE (Error)</div>
                                <div className="stat-value" style={{ color: '#ef4444' }}>{trainResults.metrics.rmse.toFixed(4)}</div>
                              </div>
                            </>
                          )}
                        </div>

                        <h4>Top Feature Importances</h4>
                        <div className="table-container" style={{ marginTop: '1rem' }}>
                          <table>
                            <thead>
                              <tr>
                                <th>Feature</th>
                                <th>Importance Weight</th>
                              </tr>
                            </thead>
                            <tbody>
                              {trainResults.top_features.map((feat, idx) => (
                                <tr key={idx}>
                                  <td>{feat.feature}</td>
                                  <td>
                                    <div style={{ display: 'flex', alignItems: 'center', gap: '1rem' }}>
                                      <div style={{ flex: 1, background: 'rgba(255,255,255,0.1)', height: '8px', borderRadius: '4px', overflow: 'hidden' }}>
                                        <div style={{ width: `${feat.importance * 100}%`, background: 'var(--accent-gradient)', height: '100%' }}></div>
                                      </div>
                                      <span style={{ fontSize: '0.85rem' }}>{(feat.importance * 100).toFixed(1)}%</span>
                                    </div>
                                  </td>
                                </tr>
                              ))}
                            </tbody>
                          </table>
                        </div>
                      </div>
                    ) : (
                      <div style={{ display: 'flex', flexDirection: 'column', justifyContent: 'center', alignItems: 'center', height: '100%', color: 'var(--text-muted)' }}>
                        <BrainCircuit size={48} style={{ opacity: 0.2, marginBottom: '1rem' }} />
                        <h3>ML Studio Ready</h3>
                        <p>Select your target variable and click train to auto-build a model.</p>
                      </div>
                    )}
                  </div>
                </div>
              )}
            </>
          )}
        </div>
      </main>
    </div>
  )
}

export default App
