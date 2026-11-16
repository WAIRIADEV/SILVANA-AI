
import type { AssistantState } from '../../types/assistant'
import type {
  ChatRequest,
  ChatResponse,
  ChatSourceCitation,
} from '../../types/api'

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

function isRecord(value: unknown): value is Record<string, unknown> {
  return (
    typeof value === 'object' &&
    value !== null &&
    !Array.isArray(value)
  )
}

function isNullableNumber(value: unknown): value is number | null {
  return value === null || typeof value === 'number'
}

function isValidCitation(value: unknown): value is ChatSourceCitation {
  if (!isRecord(value)) return false

  return (
    typeof value.ref === 'string' &&
    typeof value.document_id === 'string' &&
    typeof value.filename === 'string' &&
    (value.scope === 'temporary' || value.scope === 'persistent') &&
    (value.page_number === null ||
      (typeof value.page_number === 'number' &&
        Number.isInteger(value.page_number))) &&
    typeof value.chunk_index === 'number' &&
    Number.isInteger(value.chunk_index)
  )
}

function isValidChatResponse(value: unknown): value is ChatResponse {
  if (!isRecord(value)) return false

  return (
    typeof value.content === 'string' &&
    typeof value.provider === 'string' &&
    typeof value.model === 'string' &&
    isNullableNumber(value.input_tokens) &&
    isNullableNumber(value.output_tokens) &&
    (typeof value.finish_reason === 'string' ||
      value.finish_reason === null) &&
    Array.isArray(value.sources) &&
    value.sources.every(isValidCitation)
  )
}

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

  if (!isRecord(data)) {
    throw new Error('SILVANA received an invalid streaming payload.')
  }

  if (eventName === 'state') {
    const state = data.state

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
    if (!isValidChatResponse(data)) {
      throw new Error(
        'SILVANA received an invalid chat response or citation payload.',
      )
    }

    return {
      event: 'response',
      data,
    }
  }

  if (eventName === 'error') {
    return {
      event: 'error',
      data: {
        detail:
          typeof data.detail === 'string'
            ? data.detail
            : 'SILVANA could not complete the request.',
        status_code:
          typeof data.status_code === 'number'
            ? data.status_code
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
      const body: unknown = await response.json()

      if (isRecord(body) && typeof body.detail === 'string') {
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
