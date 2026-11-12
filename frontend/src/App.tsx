
import './App.css'
import { useBackendHealth } from './hooks/useBackendHealth'

function App() {
  const backendHealth = useBackendHealth()

  const backendStatusLabel =
    backendHealth.status === 'operational'
      ? 'ONLINE'
      : backendHealth.status === 'checking'
        ? 'CHECKING'
        : 'OFFLINE'

  const backendStatusClass =
    backendHealth.status === 'operational'
      ? 'status-online'
      : 'status-pending'

  return (
    <main className="silvana-shell">
      <header className="topbar">
        <div className="brand">
          <div className="brand-mark">S</div>
          <div>
            <div className="brand-name">SILVANA</div>
            <div className="brand-subtitle">INTELLIGENT SYSTEMS</div>
          </div>
        </div>

        <div className="topbar-center">
          <span className="system-indicator" />
          INTERFACE INITIALIZED
        </div>

        <div className="topbar-right">
          <span className="environment-label">LOCAL ENVIRONMENT</span>
          <span className="version-label">v0.1.0</span>
        </div>
      </header>

      <div className="workspace">
        <aside className="left-panel">
          <div className="panel-heading">NAVIGATION</div>

          <nav className="navigation">
            <div className="nav-item active">
              <span>◈</span>
              Overview
            </div>

            <div className="nav-item">
              <span>⌘</span>
              Intelligence
            </div>

            <div className="nav-item">
              <span>◤</span>
              Documents
            </div>

            <div className="nav-item">
              <span>⌁</span>
              Workflows
            </div>

            <div className="nav-item">
              <span>◷</span>
              Activity
            </div>
          </nav>

          <div className="left-panel-footer">
            <div className="panel-heading">SYSTEM VERSION</div>
            <div className="muted-value">SILVANA CORE 0.1</div>
            <div className="muted-value">LOCAL INSTANCE</div>
          </div>
        </aside>

        <section className="main-stage">
          <div className="stage-heading">
            <div>
              <div className="eyebrow">
                PERSONAL INTELLIGENCE SYSTEM
              </div>

              <h1>Good to have you here.</h1>

              <p className="stage-description">
                Observe. Understand. Execute. Verify.
              </p>
            </div>

            <div className="session-indicator">
              <span className="system-indicator" />
              SESSION READY
            </div>
          </div>

          <div className="hologram-stage">
            <div className="stage-grid" />

            <div className="hologram">
              <div className="orbit orbit-outer" />
              <div className="orbit orbit-middle" />
              <div className="orbit orbit-inner" />

              <div className="hologram-core">
                <div className="core-inner">
                  <span>S</span>
                </div>
              </div>

              <div className="core-glow" />
            </div>

            <div className="hologram-caption">
              <span className="caption-line" />
              <span>HOLOGRAPHIC CORE</span>
              <span className="caption-line" />
            </div>

            <div className="stage-coordinate coordinate-left">
              CORE / 001
            </div>

            <div className="stage-coordinate coordinate-right">
              VISUALIZATION PENDING
            </div>
          </div>

          <div className="interaction-area">
            <div className="interaction-heading">
              <span>COMMAND INTERFACE</span>
              <span className="muted-value">VOICE / TEXT</span>
            </div>

            <div className="command-placeholder">
              <span className="command-symbol">›</span>
              <span>
                Awaiting connection to SILVANA intelligence...
              </span>
            </div>
          </div>
        </section>

        <aside className="right-panel">
          <div className="panel-section">
            <div className="panel-heading">SYSTEM STATUS</div>

            <div className="status-row">
              <span>Frontend</span>
              <span className="status-online">ONLINE</span>
            </div>

            <div className="status-row">
              <span>Backend API</span>

              <span
                className={backendStatusClass}
                title={
                  backendHealth.error ??
                  backendHealth.data?.message ??
                  ''
                }
              >
                {backendStatusLabel}
              </span>
            </div>

            <div className="status-row">
              <span>AI Providers</span>
              <span className="status-pending">PENDING</span>
            </div>

            <div className="status-row">
              <span>Voice Engine</span>
              <span className="status-pending">PENDING</span>
            </div>
          </div>

          <div className="panel-divider" />

          <div className="panel-section">
            <div className="panel-heading">INTELLIGENCE</div>

            <div className="metric-card">
              <div className="metric-label">ACTIVE PROVIDERS</div>
              <div className="metric-value">--</div>
              <div className="metric-description">
                Awaiting backend telemetry
              </div>
            </div>

            <div className="metric-card">
              <div className="metric-label">EXECUTIONS</div>
              <div className="metric-value">--</div>
              <div className="metric-description">
                No execution data received
              </div>
            </div>
          </div>

          <div className="panel-divider" />

          <div className="panel-section">
            <div className="panel-heading">SYSTEM PRINCIPLES</div>

            <div className="principle">01 / Observe</div>
            <div className="principle">02 / Understand</div>
            <div className="principle">03 / Execute</div>
            <div className="principle">04 / Verify</div>
          </div>
        </aside>
      </div>

      <footer className="footer">
        <span>SILVANA INTELLIGENCE SYSTEMS</span>
        <span>LOCAL INSTANCE · DEVELOPMENT MODE</span>
      </footer>
    </main>
  )
}

export default App
