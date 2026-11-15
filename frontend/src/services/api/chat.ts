
import type { AssistantState } from '../../types/assistant'
import type { ChatRequest, ChatResponse } from '../../types/api'

const API_BASE_URL = '/api/v1'

const VALID_ASSISTANT_STATES = new Set<AssistantState>([
  'idle',
  'listening',
  'thinking',
  'executing',
  'verifying',
  'completed',
  'error',
])

type StreamEvent =
  | { event: 'state'; data: { state: AssistantState } }
  | { event: 'response'; data: ChatResponse }
  | { event: 'error'; data: { detail?: string; status_code?: number } }

function parseEvent(block: string): StreamEvent | null {
  let eventName = ''
  const dataLines: string[] = []

  for (const line of block.split(/\r?\n/)) {
    if (!line || line.startsWith(':')) continue

    const separator = line.indexOf(':')
    const field = separator === -1 ? line : line.slice(0, separator)
    let value = separator === -1 ? '' : line.slice(separator + 1)

    if (value.startsWith(' ')) {
      value = value.slice(1)
    }

    if (field === 'event') {
      eventName = value
    } else if (field === 'data') {
      dataLines.push(value)
    }
  }

  if (!eventName || dataLines.length === 0) return null

  let data: unknown

  try {
    data = JSON.parse(dataLines.join('\n'))
  } catch {
    throw new Error('SILVANA received an invalid streaming event.')
  }

  if (!data || typeof data !== 'object') {
    throw new Error('SILVANA received an invalid streaming payload.')
  }

  if (eventName === 'state') {
    const state = (data as { state?: unknown }).state

    if (
      typeof state !== 'string' ||
      !VALID_ASSISTANT_STATES.has(state as AssistantState)
    ) {
      throw new Error('SILVANA received an unknown assistant state.')
    }

    return {
      event: 'state',
      data: { state: state as AssistantState },
    }
  }

  if (eventName === 'response') {
    const response = data as Partial<ChatResponse>

    if (
      typeof response.content !== 'string' ||
      typeof response.provider !== 'string' ||
      typeof response.model !== 'string'
    ) {
      throw new Error('SILVANA received an invalid chat response.')
    }

    return {
      event: 'response',
      data: response as ChatResponse,
    }
  }

  if (eventName === 'error') {
    const error = data as {
      detail?: unknown
      status_code?: unknown
    }

    return {
      event: 'error',
      data: {
        detail:
          typeof error.detail === 'string'
            ? error.detail
            : 'SILVANA could not complete the request.',
        status_code:
          typeof error.status_code === 'number'
            ? error.status_code
            : undefined,
      },
    }
  }

  return null
}

export async function sendStreamingChatMessage(
  request: ChatRequest,
  onStateChange: (state: AssistantState) => void,
): Promise<ChatResponse> {
  const response = await fetch(`${API_BASE_URL}/chat/stream`, {
    method: 'POST',
    headers: {
      Accept: 'text/event-stream',
      'Content-Type': 'application/json',
    },
    body: JSON.stringify(request),
  })

  if (!response.ok) {
    let message = `API request failed: ${response.status}`

    try {
      const body = await response.json()

      if (typeof body.detail === 'string') {
        message = body.detail
      }
    } catch {
      // Retain the original HTTP error.
    }

    throw new Error(message)
  }

  if (!response.body) {
    throw new Error('SILVANA could not establish a streaming connection.')
  }

  const reader = response.body.getReader()
  const decoder = new TextDecoder()
  let buffer = ''
  let finalResponse: ChatResponse | null = null

  function handleEvent(block: string): void {
    const event = parseEvent(block)

    if (!event) return

    if (event.event === 'state') {
      onStateChange(event.data.state)
      return
    }

    if (event.event === 'error') {
      throw new Error(event.data.detail)
    }

    if (event.event === 'response') {
      finalResponse = event.data
    }
  }

  try {
    while (true) {
      const { value, done } = await reader.read()

      if (done) break

      buffer += decoder.decode(value, { stream: true })

      const blocks = buffer.split(/\r?\n\r?\n/)
      buffer = blocks.pop() ?? ''

      for (const block of blocks) {
        handleEvent(block)
      }
    }

    buffer += decoder.decode()

    if (buffer.trim()) {
      handleEvent(buffer)
    }
  } finally {
    reader.releaseLock()
  }

  if (!finalResponse) {
    throw new Error(
      'The streaming connection ended without a final response.',
    )
  }

  return finalResponse
}
