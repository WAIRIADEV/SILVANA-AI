
import './App.css'
import { useBackendHealth } from './hooks/useBackendHealth'
import HolographicCore from './components/HolographicCore'
import ChatInterface from './components/ChatInterface'

function App() {
  const backendHealth = useBackendHealth()

  const backendStatus =
    backendHealth.status === 'operational'
      ? 'ONLINE'
      : backendHealth.status === 'checking'
        ? 'CONNECTING'
        : 'OFFLINE'

  const backendClass =
    backendHealth.status === 'operational'
      ? 'telemetry-value online'
      : backendHealth.status === 'checking'
        ? 'telemetry-value pending'
        : 'telemetry-value offline'

  return (
    <main className="silvana-interface">
      <div className="ambient-grid" />

      <header className="interface-header">
        <div className="identity">
          <div className="identity-symbol">◈</div>

          <div>
            <div className="identity-name">SILVANA</div>
            <div className="identity-subtitle">
              INTELLIGENCE SYSTEMS
            </div>
          </div>
        </div>

        <div className="header-center">
          <span className="live-indicator" />
          SYSTEM ACTIVE
        </div>

        <div className="header-meta">
          <span>LOCAL INSTANCE</span>
          <span className="meta-divider">/</span>
          <span>CORE 0.1.0</span>
        </div>
      </header>
<div className="workspace-layout">
      <section className="observation-stage">
        <div className="corner-mark top-left" />
        <div className="corner-mark top-right" />
        <div className="corner-mark bottom-left" />
        <div className="corner-mark bottom-right" />

        <div className="stage-topline">
          <span>ARTIFICIAL INTELLIGENCE</span>
          <span>VISUALIZATION UNIT / 001</span>
        </div>

        <div className="hologram-container">
          <div className="holo-backdrop" />

          <div className="scanner-ring ring-one" />
          <div className="scanner-ring ring-two" />
          <div className="scanner-ring ring-three" />

          <div className="vertical-beam" />

          <HolographicCore />

          <div className="scan-line" />
        </div>

        <div className="observation-message">
          <div className="message-overline">
            <span />
            PERCEPTION SYSTEM ENGAGED
            <span />
          </div>

          <h1>SILVANA IS WATCHING</h1>

          <p>
            OBSERVING ENVIRONMENT <span>·</span> AWAITING INPUT
          </p>
        </div>

        <div className="side-readout readout-left">
          <span className="readout-label">ENTITY</span>
          <span className="readout-value">SILVANA_01</span>
          <span className="readout-label">MODE</span>
          <span className="readout-value cyan">OBSERVATION</span>
        </div>

        <div className="side-readout readout-right">
          <span className="readout-label">CORE STATUS</span>
          <span className="readout-value cyan">ACTIVE</span>
          <span className="readout-label">RENDER ENGINE</span>
          <span className="readout-value">THREE.JS</span>
        </div>

        <div className="stage-bottomline">
          <span>OBSERVE</span>
          <span className="bottomline-track">
            <span />
          </span>
          <span>UNDERSTAND</span>
          <span className="bottomline-track">
            <span />
          </span>
          <span>EXECUTE</span>
          <span className="bottomline-track">
            <span />
          </span>
          <span>VERIFY</span>
        </div>
      </section>

      <ChatInterface />
    </div>

      <footer className="interface-footer">
        <div className="footer-brand">
          SILVANA INTELLIGENCE SYSTEMS
        </div>

        <div className="footer-status">
          <span className="footer-dot" />
          FRONTEND ONLINE

          <span className="footer-separator">/</span>

          BACKEND
          <span className={backendClass}>
            {backendStatus}
          </span>
        </div>

        <div className="footer-version">
          LOCAL INSTANCE · DEVELOPMENT MODE
        </div>
      </footer>
    </main>
  )
}

export default App
