
export type AssistantState =
  | 'idle'
  | 'listening'
  | 'thinking'
  | 'executing'
  | 'verifying'
  | 'completed'
  | 'error'

export interface AssistantStateConfig {
  label: string
  description: string
  color: number
  cssColor: string
}

export const ASSISTANT_STATES: Record<
  AssistantState,
  AssistantStateConfig
> = {
  idle: {
    label: 'IDLE',
    description: 'Awaiting input',
    color: 0x22d3ee,
    cssColor: '#22d3ee',
  },
  listening: {
    label: 'LISTENING',
    description: 'Receiving user input',
    color: 0x3b82f6,
    cssColor: '#3b82f6',
  },
  thinking: {
    label: 'THINKING',
    description: 'Processing request',
    color: 0xa78bfa,
    cssColor: '#a78bfa',
  },
  executing: {
    label: 'EXECUTING',
    description: 'Performing an operation',
    color: 0xf59e0b,
    cssColor: '#f59e0b',
  },
  verifying: {
    label: 'VERIFYING',
    description: 'Checking operation results',
    color: 0x14b8a6,
    cssColor: '#14b8a6',
  },
  completed: {
    label: 'COMPLETED',
    description: 'Request completed',
    color: 0x22c55e,
    cssColor: '#22c55e',
  },
  error: {
    label: 'ERROR',
    description: 'Request failed',
    color: 0xef4444,
    cssColor: '#ef4444',
  },
}
