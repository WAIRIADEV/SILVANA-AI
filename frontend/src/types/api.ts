export interface HealthResponse {
  status: string
  application: string
  version: string
  message: string
}

export type BackendConnectionStatus =
  | 'checking'
  | 'operational'
  | 'disconnected'