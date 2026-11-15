
import { useState } from 'react'
import './App.css'
import { useBackendHealth } from './hooks/useBackendHealth'
import HolographicCore from './components/HolographicCore'
import ChatInterface from './components/ChatInterface'

type ExplorerSection =
  | 'intelligence'
  | 'workflows'
  | 'documents'
  | 'activity'

function App() {
  const backendHealth = useBackendHealth()
  const [activeSection, setActiveSection] =
    useState<ExplorerSection>('intelligence')

  const backendStatus =
    backendHealth.status === 'operational'
      ? 'ONLINE'
      : backendHealth.status === 'checking'
        ? 'CONNECTING'
        : 'OFFLINE'

  const backendClass =
    backendHealth.status === 'operational'
      ? 'online'
      : backendHealth.status === 'checking'
        ? 'pending'
        : 'offline'

  const explorerSections: {
    id: ExplorerSection
    label: string
    symbol: string
    description: string
  }[] = [
    {
      id: 'intelligence',
      label: 'Intelligence',
      symbol: '◇',
      description: 'System overview and intelligence status',
    },
    {
      id: 'workflows',
      label: 'Workflows',
      symbol: '⌘',
      description: 'Automation and workflow workspace',
    },
    {
      id: 'documents',
      label: 'Documents',
      symbol: '▤',
      description: 'Document intelligence workspace',
    },
    {
      id: 'activity',
      label: 'Activity',
      symbol: '◷',
      description: 'System activity and event history',
    },
  ]

  function renderExplorerContent() {
    switch (activeSection) {
      case 'intelligence':
        return (
          <div className="explorer-content">
            <div className="explorer-content-heading">
              <span className="section-eyebrow">SYSTEM OVERVIEW</span>
              <h2>Intelligence</h2>
              <p>
                Operational status of the SILVANA intelligence system.
              </p>
            </div>

            <div className="status-card">
              <div className="status-card-heading">
                <span>BACKEND SERVICE</span>
                <span className={`status-indicator ${backendClass}`} />
              </div>
              <strong className={backendClass}>{backendStatus}</strong>
              <p>FastAPI service connectivity</p>
            </div>

            <div className="status-card">
              <div className="status-card-heading">
                <span>FRONTEND</span>
                <span className="status-indicator online" />
              </div>
              <strong className="online">ACTIVE</strong>
              <p>React application loaded</p>
            </div>

            <div className="status-card">
              <div className="status-card-heading">
                <span>RENDER ENGINE</span>
                <span className="status-indicator online" />
              </div>
              <strong className="online">THREE.JS</strong>
              <p>3D visualization component</p>
            </div>

            <div className="status-card telemetry-card">
              <div className="status-card-heading">
                <span>HOST TELEMETRY</span>
                <span className="status-indicator unavailable" />
              </div>
              <strong className="muted-status">NOT CONNECTED</strong>
              <p>
                CPU, memory and storage metrics are not yet connected.
              </p>
            </div>

            <div className="explorer-note">
              <span className="note-symbol">i</span>
              <p>
                System readings will only be displayed when a verified
                telemetry source is available.
              </p>
            </div>
          </div>
        )

      case 'workflows':
        return (
          <div className="explorer-content placeholder-content">
            <span className="section-eyebrow">AUTOMATION SYSTEMS</span>
            <div className="placeholder-symbol">⌘</div>
            <h2>Workflows</h2>
            <p>
              A workspace for orchestrating multi-step tasks,
              integrations and automated processes.
            </p>
            <span className="feature-status">PLANNED MODULE</span>
            <div className="placeholder-divider" />
            <p className="placeholder-detail">
              Workflow execution, scheduling and integrations will be
              introduced in a later development sprint.
            </p>
          </div>
        )

      case 'documents':
        return (
          <div className="explorer-content placeholder-content">
            <span className="section-eyebrow">KNOWLEDGE SYSTEMS</span>
            <div className="placeholder-symbol">▤</div>
            <h2>Documents</h2>
            <p>
              A dedicated workspace for document understanding,
              knowledge retrieval and contextual answers.
            </p>
            <span className="feature-status">PLANNED MODULE</span>
            <div className="placeholder-divider" />
            <p className="placeholder-detail">
              Document ingestion, semantic retrieval and source
              citations are planned for the document intelligence sprint.
            </p>
          </div>
        )

      case 'activity':
        return (
          <div className="explorer-content placeholder-content">
            <span className="section-eyebrow">SYSTEM OBSERVABILITY</span>
            <div className="placeholder-symbol">◷</div>
            <h2>Activity</h2>
            <p>
              A centralized view of system events, operations and
              execution history.
            </p>
            <span className="feature-status">PLANNED MODULE</span>
            <div className="placeholder-divider" />
            <p className="placeholder-detail">
              Event history and execution tracking will become available
              as the activity and observability capabilities are developed.
            </p>
          </div>
        )
    }
  }

  return (
    <main className="silvana-interface">
      <div className="ambient-grid" />

      <header className="interface-header">
        <div className="identity">
          <div className="identity-symbol">◇</div>

          <div>
            <div className="identity-name">SILVANA</div>
            <div className="identity-subtitle">
              INTELLIGENCE SYSTEMS
            </div>
          </div>
        </div>

        <div className="header-center">
          <span className={`live-indicator ${backendClass}`} />
          SYSTEM {backendStatus}
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

            <h1>SILVANA</h1>

            <p>
              OBSERVING ENVIRONMENT <span>/</span> AWAITING INPUT
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
            <span className="bottomline-track"><span /></span>
            <span>UNDERSTAND</span>
            <span className="bottomline-track"><span /></span>
            <span>EXECUTE</span>
            <span className="bottomline-track"><span /></span>
            <span>VERIFY</span>
          </div>
        </section>

        <section className="interaction-panel">
          <ChatInterface />
        </section>

        <aside className="explorer-panel">
          <div className="explorer-header">
            <div>
              <span className="section-eyebrow">WORKSPACE</span>
              <h2>EXPLORER</h2>
            </div>
            <span className="explorer-menu-mark">•••</span>
          </div>

          <div className="explorer-navigation">
            <div className="navigation-label">SILVANA SYSTEMS</div>

            {explorerSections.map((section) => (
              <button
                key={section.id}
                type="button"
                className={`explorer-nav-item ${
                  activeSection === section.id ? 'active' : ''
                }`}
                onClick={() => setActiveSection(section.id)}
                aria-current={
                  activeSection === section.id ? 'page' : undefined
                }
                title={section.description}
              >
                <span className="nav-symbol">{section.symbol}</span>
                <span>{section.label}</span>
                <span className="nav-chevron">›</span>
              </button>
            ))}
          </div>

          <div className="explorer-divider" />

          <div className="explorer-view" key={activeSection}>
            {renderExplorerContent()}
          </div>

          <div className="explorer-footer">
            <span className="explorer-footer-dot" />
            LOCAL WORKSPACE
          </div>
        </aside>
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
          <span className={`telemetry-value ${backendClass}`}>
            {backendStatus}
          </span>
        </div>

        <div className="footer-version">
          LOCAL INSTANCE / DEVELOPMENT MODE
        </div>
      </footer>
    </main>
  )
}

export default App
