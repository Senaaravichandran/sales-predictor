import { useState, useEffect } from 'react'
import { Activity, BarChart3, TrendingUp, TrendingDown, Database, Cpu } from 'lucide-react'
import axios from 'axios'
import './App.css'

function App() {
  const [apiStatus, setApiStatus] = useState('Connecting...')
  const [isLoading, setIsLoading] = useState(true)

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
          <button className="btn-primary">Generate Report</button>
        </div>
      </header>

      <main className="dashboard-grid">
        <div className="stat-card glass-panel animate-fade-in delay-2">
          <div className="stat-header">
            <TrendingUp size={20} color="var(--accent-1)" />
            Total Predicted Sales
          </div>
          <div className="stat-value">2.4M</div>
          <div className="stat-trend trend-up">
            <TrendingUp size={16} /> +14.5% from last month
          </div>
        </div>

        <div className="stat-card glass-panel animate-fade-in delay-2">
          <div className="stat-header">
            <Database size={20} color="var(--accent-2)" />
            Data Points Analyzed
          </div>
          <div className="stat-value">1.8M</div>
          <div className="stat-trend trend-up">
            <TrendingUp size={16} /> +120k new records
          </div>
        </div>

        <div className="stat-card glass-panel animate-fade-in delay-2">
          <div className="stat-header">
            <Cpu size={20} color="#10b981" />
            Model Accuracy
          </div>
          <div className="stat-value">94.2%</div>
          <div className="stat-trend trend-up">
            <TrendingUp size={16} /> +1.2% model improvement
          </div>
        </div>

        <div className="main-chart glass-panel animate-fade-in delay-3">
          <div className="chart-placeholder">
            <BarChart3 size={64} style={{ opacity: 0.5 }} />
            <h3>Sales Prediction Visualization</h3>
            <p>Connect your datasets to generate advanced ML insights.</p>
            <button className="btn-primary" style={{ marginTop: '1rem', padding: '0.5rem 1rem', fontSize: '0.9rem' }}>
              Connect Dataset
            </button>
          </div>
        </div>
      </main>

      <footer className="footer animate-fade-in delay-3">
        <p>&copy; {new Date().getFullYear()} SaleSense | Advanced Analytics Platform</p>
      </footer>
    </div>
  )
}

export default App
