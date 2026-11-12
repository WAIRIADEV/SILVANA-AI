
import { useCallback, useEffect, useState } from 'react'

import { getBackendHealth } from '../services/api/health'

import type {
  HealthResponse,
  BackendConnectionStatus,
} from '../types/api'

interface BackendHealthState {
  status: BackendConnectionStatus
  data: HealthResponse | null
  error: string | null
}

export function useBackendHealth() {
  const [health, setHealth] = useState<BackendHealthState>({
    status: 'checking',
    data: null,
    error: null,
  })

  const checkHealth = useCallback(async () => {
    try {
      const data = await getBackendHealth()

      setHealth({
        status: 'operational',
        data,
        error: null,
      })
    } catch {
      setHealth({
        status: 'disconnected',
        data: null,
        error: 'Unable to reach SILVANA backend.',
      })
    }
  }, [])

  useEffect(() => {
    let active = true

    const runHealthCheck = async () => {
      try {
        const data = await getBackendHealth()

        if (!active) return

        setHealth({
          status: 'operational',
          data,
          error: null,
        })
      } catch {
        if (!active) return

        setHealth({
          status: 'disconnected',
          data: null,
          error: 'Unable to reach SILVANA backend.',
        })
      }
    }

    void runHealthCheck()

    const interval = window.setInterval(() => {
      void runHealthCheck()
    }, 30000)

    return () => {
      active = false
      window.clearInterval(interval)
    }
  }, [])

  return {
    ...health,
    refresh: checkHealth,
  }
}
