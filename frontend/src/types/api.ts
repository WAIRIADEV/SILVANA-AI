
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

export type ChatRequestRole = 'user' | 'assistant'

export interface ChatMessage {
  role: ChatRole
  content: string
}

export interface ChatRequestMessage {
  role: ChatRequestRole
  content: string
}

export type KnowledgeScope =
  | 'none'
  | 'conversation'
  | 'library'
  | 'both'

export interface ChatRequest {
  messages: ChatRequestMessage[]
  provider?: 'gemini' | 'groq'
  temperature?: number
  max_output_tokens?: number
  conversation_id?: string | null
  knowledge_scope?: KnowledgeScope
}

export interface ChatSourceCitation {
  ref: string
  document_id: string
  filename: string
  scope: 'temporary' | 'persistent'
  page_number: number | null
  chunk_index: number
}

export interface ChatResponse {
  content: string
  provider: string
  model: string
  input_tokens: number | null
  output_tokens: number | null
  finish_reason: string | null
  sources: ChatSourceCitation[]
}
