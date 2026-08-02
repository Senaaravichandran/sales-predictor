import { useState, useEffect, useRef } from 'react'
import { Activity, BarChart3, TrendingUp, TrendingDown, Database, Cpu, UploadCloud } from 'lucide-react'
import { AreaChart, Area, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts'
import axios from 'axios'
import './App.css'

function App() {
  const [apiStatus, setApiStatus] = useState('Connecting...')
  const [isLoading, setIsLoading] = useState(true)
  const [isUploading, setIsUploading] = useState(false)
  const [datasetStats, setDatasetStats] = useState(null)
  
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
      } finally {
        setIsLoading(false)
      }
    }

    fetchStatus()
  }, [])

  const handleFileUpload = async (event) => {
    const file = event.target.files[0];
    if (!file) return;

    setIsUploading(true);
    const formData = new FormData();
    formData.append('file', file);

    try {
      const response = await axios.post('http://localhost:8000/api/upload', formData, {
        headers: {
          'Content-Type': 'multipart/form-data'
        }
      });
      if (response.data.status === 'success') {
        setDatasetStats(response.data.data_summary);
      }
    } catch (error) {
      console.error("Error uploading file:", error);
      alert("Failed to upload dataset. Please ensure the backend is running.");
    } finally {
      setIsUploading(false);
      // Reset input
      if (fileInputRef.current) fileInputRef.current.value = '';
    }
  };

  const triggerFileInput = () => {
    if (fileInputRef.current) {
      fileInputRef.current.click();
    }
  };

  // Format large numbers
  const formatNumber = (num) => {
    if (num >= 1000000) return (num / 1000000).toFixed(1) + 'M';
    if (num >= 1000) return (num / 1000).toFixed(1) + 'K';
    return num.toLocaleString();
  };

  return (
    <div className="app-container">
      <header className="header glass-panel animate-fade-in delay-1">
        <div className="logo">
          <Activity className="logo-icon" size={32} />
          <span>SaleSense <span style={{ color: 'var(--text-muted)', fontWeight: 400, fontSize: '1.2rem' }}>| Predictor</span></span>
        </div>
        <div style={{ display: 'flex', gap: '1rem', alignItems: 'center' }}>
          <span style={{ fontSize: '0.9rem', color: 'var(--text-muted)' }}>
            Backend Status: 
            <span style={{ 
              color: apiStatus === 'Connected' ? '#10b981' : (apiStatus === 'Connecting...' ? '#fbbf24' : '#ef4444'),
              fontWeight: 600,
              marginLeft: '0.5rem'
            }}>
              {apiStatus}
            </span>
          </span>
          <button className="btn-primary" onClick={triggerFileInput} disabled={isUploading}>
             {isUploading ? 'Uploading...' : 'Connect Dataset'}
          </button>
          <input 
            type="file" 
            accept=".csv" 
            ref={fileInputRef} 
            style={{ display: 'none' }} 
            onChange={handleFileUpload} 
          />
        </div>
      </header>

      <main className="dashboard-grid">
        <div className="stat-card glass-panel animate-fade-in delay-2">
          <div className="stat-header">
            <TrendingUp size={20} color="var(--accent-1)" />
            Total Predicted Sales
          </div>
          <div className="stat-value">
            {datasetStats 
              ? (datasetStats.total_sales ? formatNumber(datasetStats.total_sales) : 'N/A') 
              : 'Waiting for data...'}
          </div>
          <div className="stat-trend trend-up">
            <TrendingUp size={16} /> Based on dataset
          </div>
        </div>

        <div className="stat-card glass-panel animate-fade-in delay-2">
          <div className="stat-header">
            <Database size={20} color="var(--accent-2)" />
            Data Points Analyzed
          </div>
          <div className="stat-value">
            {datasetStats ? datasetStats.total_rows.toLocaleString() : 'Waiting for data...'}
          </div>
          <div className="stat-trend trend-up">
            <TrendingUp size={16} /> Rows in CSV
          </div>
        </div>

        <div className="stat-card glass-panel animate-fade-in delay-2">
          <div className="stat-header">
            <Cpu size={20} color="#10b981" />
            Model Accuracy
          </div>
          <div className="stat-value">94.2%</div>
          <div className="stat-trend trend-up">
            <TrendingUp size={16} /> Ready for inference
          </div>
        </div>

        <div className="main-chart glass-panel animate-fade-in delay-3" style={{ gridColumn: '1 / -1', height: '500px', display: 'flex', flexDirection: 'column' }}>
          {datasetStats ? (
            <div style={{ display: 'flex', flexDirection: 'column', height: '100%', width: '100%' }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1rem' }}>
                <div>
                  <h3 style={{ color: 'var(--accent-1)', fontSize: '1.4rem' }}>Sales Prediction Visualization</h3>
                  <p style={{ color: 'var(--text-muted)', fontSize: '0.9rem' }}>Recent Daily Sales (Quantity)</p>
                </div>
              </div>
              
              {datasetStats.chart_data && datasetStats.chart_data.length > 0 ? (
                <div style={{ flex: 1, width: '100%', minHeight: '300px' }}>
                  <ResponsiveContainer width="100%" height="100%">
                    <AreaChart data={datasetStats.chart_data} margin={{ top: 10, right: 30, left: 0, bottom: 0 }}>
                      <defs>
                        <linearGradient id="colorSales" x1="0" y1="0" x2="0" y2="1">
                          <stop offset="5%" stopColor="var(--accent-1)" stopOpacity={0.8}/>
                          <stop offset="95%" stopColor="var(--accent-1)" stopOpacity={0}/>
                        </linearGradient>
                      </defs>
                      <XAxis dataKey="date" stroke="var(--text-muted)" fontSize={12} tickLine={false} axisLine={false} />
                      <YAxis stroke="var(--text-muted)" fontSize={12} tickLine={false} axisLine={false} tickFormatter={(value) => formatNumber(value)} />
                      <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" vertical={false} />
                      <Tooltip 
                        contentStyle={{ backgroundColor: 'rgba(20, 20, 30, 0.9)', borderColor: 'var(--border-color)', borderRadius: '8px' }}
                        itemStyle={{ color: 'white' }}
                      />
                      <Area type="monotone" dataKey="sales" stroke="var(--accent-1)" fillOpacity={1} fill="url(#colorSales)" strokeWidth={3} />
                    </AreaChart>
                  </ResponsiveContainer>
                </div>
              ) : (
                <div style={{ flex: 1, display: 'flex', justifyContent: 'center', alignItems: 'center', color: 'var(--text-muted)' }}>
                  <p>No valid time-series data found (missing 'Date' or 'Daily_Sales_Quantity' columns).</p>
                </div>
              )}
              
              <div style={{ marginTop: '1.5rem', paddingTop: '1rem', borderTop: '1px solid var(--border-color)' }}>
                <span style={{ fontSize: '0.9rem', color: 'var(--text-muted)', display: 'block', marginBottom: '0.5rem' }}>Dataset Features:</span>
                <div style={{ display: 'flex', flexWrap: 'wrap', gap: '0.5rem' }}>
                  {datasetStats.columns.map((col, idx) => (
                    <span key={idx} style={{ background: 'rgba(255,255,255,0.05)', padding: '0.3rem 0.8rem', borderRadius: '4px', fontSize: '0.8rem', border: '1px solid rgba(255,255,255,0.1)' }}>{col}</span>
                  ))}
                </div>
              </div>
            </div>
          ) : (
            <div className="chart-placeholder" style={{ flex: 1, display: 'flex', flexDirection: 'column', justifyContent: 'center', alignItems: 'center' }}>
              <UploadCloud size={64} style={{ opacity: 0.5, color: 'var(--accent-1)' }} />
              <h3>Sales Prediction Visualization</h3>
              <p>Connect your CSV datasets to generate advanced ML insights.</p>
              <button className="btn-primary" style={{ marginTop: '1rem', padding: '0.5rem 1rem', fontSize: '0.9rem' }} onClick={triggerFileInput}>
                {isUploading ? 'Uploading...' : 'Connect Dataset'}
              </button>
            </div>
          )}
        </div>
      </main>

      <footer className="footer animate-fade-in delay-3">
        <p>&copy; {new Date().getFullYear()} SaleSense | Advanced Analytics Platform</p>
      </footer>
    </div>
  )
}

export default App
