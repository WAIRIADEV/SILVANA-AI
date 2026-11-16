
import { useEffect, useRef, useState } from 'react'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import { sendStreamingChatMessage } from '../services/api/chat'
import { deleteConversationDocuments } from '../services/api/documents'
import type {
  ChatMessage,
  ChatResponse,
  ChatSourceCitation,
  KnowledgeScope,
} from '../types/api'
import type { AssistantState } from '../types/assistant'
import './ChatInterface.css'

interface DisplayMessage extends ChatMessage {
  id: string
  metadata?: Pick<ChatResponse, 'provider' | 'model'>
  sources?: ChatSourceCitation[]
}

interface ChatInterfaceProps {
  onStateChange: (state: AssistantState) => void
  conversationId: string
  onConversationCleared: () => void
}

const KNOWLEDGE_SCOPE_OPTIONS: {
  value: KnowledgeScope
  label: string
  description: string
}[] = [
  {
    value: 'none',
    label: 'GENERAL CHAT',
    description: 'Answer without document retrieval',
  },
  {
    value: 'conversation',
    label: 'CONVERSATION',
    description: 'Search temporary documents in this conversation',
  },
  {
    value: 'library',
    label: 'KNOWLEDGE LIBRARY',
    description: 'Search persistent library documents',
  },
  {
    value: 'both',
    label: 'BOTH SOURCES',
    description: 'Search conversation documents and the library',
  },
]

function ChatInterface({
  onStateChange,
  conversationId,
  onConversationCleared,
}: ChatInterfaceProps) {
  const [messages, setMessages] = useState<DisplayMessage[]>([])
  const [input, setInput] = useState('')
  const [isSending, setIsSending] = useState(false)
  const [isClearing, setIsClearing] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [clearError, setClearError] = useState<string | null>(null)
  const [lastResponse, setLastResponse] = useState<ChatResponse | null>(null)
  const [knowledgeScope, setKnowledgeScope] =
    useState<KnowledgeScope>('none')

  const textareaRef = useRef<HTMLTextAreaElement>(null)
  const messagesEndRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({
      behavior: 'smooth',
      block: 'end',
    })
  }, [messages, isSending, isClearing, error, clearError])

  async function submitMessage(content: string, retry = false) {
    const trimmedContent = content.trim()

    if (!trimmedContent || isSending || isClearing) return

    const userMessage: DisplayMessage = {
      id: crypto.randomUUID(),
      role: 'user',
      content: trimmedContent,
    }

    const conversation = retry
      ? messages
      : [...messages, userMessage]

    if (!retry) {
      setMessages(conversation)
      setInput('')
    }

    setError(null)
    setClearError(null)
    setIsSending(true)
    onStateChange('thinking')

    try {
      const requestMessages = conversation
        .slice(-50)
        .filter(
          (
            message,
          ): message is DisplayMessage & {
            role: 'user' | 'assistant'
          } =>
            message.role === 'user' ||
            message.role === 'assistant',
        )
        .map(({ role, content: messageContent }) => ({
          role,
          content: messageContent,
        }))

      const response = await sendStreamingChatMessage(
        {
          messages: requestMessages,
          provider: 'groq',
          temperature: 0.7,
          max_output_tokens: 1024,
          conversation_id: conversationId,
          knowledge_scope: knowledgeScope,
        },
        onStateChange,
      )

      const assistantMessage: DisplayMessage = {
        id: crypto.randomUUID(),
        role: 'assistant',
        content: response.content,
        metadata: {
          provider: response.provider,
          model: response.model,
        },
        sources: response.sources,
      }

      setMessages((current) => [...current, assistantMessage])
      setLastResponse(response)
    } catch (err) {
      setError(
        err instanceof Error
          ? err.message
          : 'SILVANA could not complete the request.',
      )

      onStateChange('error')
    } finally {
      setIsSending(false)
      textareaRef.current?.focus()
    }
  }

  function handleSubmit(event: React.FormEvent<HTMLFormElement>) {
    event.preventDefault()
    void submitMessage(input)
  }

  function handleKeyDown(event: React.KeyboardEvent<HTMLTextAreaElement>) {
    if (event.key === 'Enter' && !event.shiftKey) {
      event.preventDefault()

      if (!isSending && !isClearing) {
        void submitMessage(input)
      }
    }
  }

  function handleInputChange(value: string) {
    setInput(value)

    if (isSending || isClearing) return

    onStateChange(value.trim() ? 'listening' : 'idle')
  }

  function retryLastMessage() {
    if (isSending || isClearing) return

    const lastUserMessage = [...messages]
      .reverse()
      .find((message) => message.role === 'user')

    if (lastUserMessage) {
      void submitMessage(lastUserMessage.content, true)
    }
  }

  async function clearConversation() {
    if (isSending || isClearing || !hasMessages) return

    setIsClearing(true)
    setClearError(null)

    try {
      await deleteConversationDocuments(conversationId)

      setMessages([])
      setInput('')
      setError(null)
      setLastResponse(null)

      onConversationCleared()
      onStateChange('idle')
    } catch (err) {
      setClearError(
        err instanceof Error
          ? err.message
          : 'SILVANA could not clear this conversation.',
      )
    } finally {
      setIsClearing(false)
      textareaRef.current?.focus()
    }
  }

  function renderSources(sources: ChatSourceCitation[]) {
    if (sources.length === 0) return null

    return (
      <div className="message-sources">
        <div className="message-sources-heading">
          SOURCES / {sources.length}
        </div>

        <ul>
          {sources.map((source) => (
            <li
              key={`${source.document_id}-${source.chunk_index}-${source.ref}`}
            >
              <span className="source-reference">{source.ref}</span>

              <span className="source-details">
                <strong>{source.filename}</strong>

                <small>
                  {source.scope === 'temporary'
                    ? 'CONVERSATION DOCUMENT'
                    : 'KNOWLEDGE LIBRARY'}

                  {source.page_number !== null &&
                    ` / PAGE ${source.page_number}`}

                  {` / CHUNK ${source.chunk_index}`}
                </small>
              </span>
            </li>
          ))}
        </ul>
      </div>
    )
  }

  const hasMessages = messages.length > 0

  return (
    <section className="chat-interface">
      <div className="chat-heading">
        <div>
          <div className="chat-overline">
            INTERACTION TERMINAL / 001
          </div>

          <h2>CONVERSATION</h2>

          <p>
            Communicate with SILVANA through natural language.
          </p>
        </div>

        <button
          className="clear-chat-button"
          type="button"
          onClick={() => void clearConversation()}
          disabled={!hasMessages || isSending || isClearing}
        >
          {isClearing ? 'CLEARING...' : 'CLEAR'}
        </button>
      </div>

      <div
        className="conversation-window"
        aria-live="polite"
        aria-label="Conversation with SILVANA"
      >
        {!hasMessages && (
          <div className="conversation-empty">
            <div className="empty-symbol">◇</div>

            <h3>READY WHEN YOU ARE</h3>

            <p>
              Start a conversation. Ask a question, explore an idea,
              or give SILVANA a task.
            </p>
          </div>
        )}

        {messages.map((message) => (
          <article
            className={`chat-message ${message.role}`}
            key={message.id}
          >
            <div className="message-avatar">
              {message.role === 'user' ? 'YOU' : 'S'}
            </div>

            <div className="message-content">
              <div className="message-author">
                {message.role === 'user' ? 'YOU' : 'SILVANA'}
              </div>

              <div className="message-text">
                {message.role === 'assistant' ? (
                  <ReactMarkdown remarkPlugins={[remarkGfm]}>
                    {message.content}
                  </ReactMarkdown>
                ) : (
                  message.content
                )}
              </div>

              {message.metadata && (
                <div className="message-metadata">
                  {message.metadata.provider.toUpperCase()}
                  {' / '}
                  {message.metadata.model}
                </div>
              )}

              {message.sources && renderSources(message.sources)}
            </div>
          </article>
        ))}

        {isSending && (
          <div className="chat-message assistant">
            <div className="message-avatar">S</div>

            <div className="message-content">
              <div className="message-author">SILVANA</div>

              <div className="thinking-indicator">
                <span />
                <span />
                <span />
                <small>PROCESSING REQUEST</small>
              </div>
            </div>
          </div>
        )}

        {error && (
          <div className="chat-error" role="alert">
            <div>
              <strong>TRANSMISSION FAILED</strong>
              <p>{error}</p>
            </div>

            <button
              type="button"
              onClick={retryLastMessage}
              disabled={isSending || isClearing}
            >
              RETRY
            </button>
          </div>
        )}

        {clearError && (
          <div className="chat-error" role="alert">
            <div>
              <strong>CLEAR FAILED</strong>
              <p>{clearError}</p>
            </div>
          </div>
        )}

        <div ref={messagesEndRef} />
      </div>

      <form className="chat-composer" onSubmit={handleSubmit}>
        <div className="composer-status">
          <span className="composer-status-dot" />
          TEXT CHANNEL ACTIVE
        </div>

        <div className="knowledge-scope-control">
          <label htmlFor="knowledge-scope">
            KNOWLEDGE CONTEXT
          </label>

          <select
            id="knowledge-scope"
            value={knowledgeScope}
            onChange={(event) =>
              setKnowledgeScope(event.target.value as KnowledgeScope)
            }
            disabled={isSending || isClearing}
            aria-describedby="knowledge-scope-description"
          >
            {KNOWLEDGE_SCOPE_OPTIONS.map((option) => (
              <option key={option.value} value={option.value}>
                {option.label}
              </option>
            ))}
          </select>

          <small id="knowledge-scope-description">
            {
              KNOWLEDGE_SCOPE_OPTIONS.find(
                (option) => option.value === knowledgeScope,
              )?.description
            }
          </small>
        </div>

        <div className="composer-input-row">
          <textarea
            ref={textareaRef}
            value={input}
            onChange={(event) => handleInputChange(event.target.value)}
            onKeyDown={handleKeyDown}
            placeholder="Message SILVANA..."
            rows={2}
            maxLength={12000}
            disabled={isSending || isClearing}
            aria-label="Type your message"
          />

          <button
            className="send-message-button"
            type="submit"
            disabled={!input.trim() || isSending || isClearing}
            aria-label="Send message"
          >
            {isSending ? '...' : 'SEND'}
          </button>
        </div>

        <div className="composer-footer">
          <span>ENTER TO SEND</span>
          <span>SHIFT + ENTER FOR NEW LINE</span>
          <span>{input.length.toLocaleString()} / 12,000</span>
        </div>
      </form>

      {lastResponse && !isSending && (
        <div className="chat-session-info">
          LAST RESPONSE: {lastResponse.provider.toUpperCase()}
          {' / '}
          {lastResponse.model}

          {lastResponse.output_tokens !== null &&
            ` / ${lastResponse.output_tokens} TOKENS`}
        </div>
      )}
    </section>
  )
}

export default ChatInterface
