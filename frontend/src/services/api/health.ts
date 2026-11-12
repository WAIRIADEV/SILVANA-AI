import { apiGet } from './client'
import type { HealthResponse } from '../../types/api'

export async function getBackendHealth(): Promise<HealthResponse> {
  const health = await apiGet<HealthResponse>('/health')

  if (
    health.application !== 'SILVANA' ||
    health.status !== 'operational'
  ) {
    throw new Error('Backend health check failed.')
  }

  return health
}