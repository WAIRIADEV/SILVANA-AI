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

export type ChatRole = 'system' | 'user' | 'assistant'

export interface ChatMessage {
  role: ChatRole
  content: string
}

export interface ChatRequest {
  messages: ChatMessage[]
  provider?: 'gemini' | 'groq'
  temperature?: number
  max_output_tokens?: number
}

export interface ChatResponse {
  content: string
  provider: string
  model: string
  input_tokens: number | null
  output_tokens: number | null
  finish_reason: string | null
}
