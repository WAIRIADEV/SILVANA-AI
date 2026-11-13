
import { useEffect, useRef, useState } from 'react'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import { sendChatMessage } from '../services/api/chat'
import type { ChatMessage, ChatResponse } from '../types/api'
import './ChatInterface.css'

interface DisplayMessage extends ChatMessage {
  id: string
  metadata?: Pick<ChatResponse, 'provider' | 'model'>
}

function ChatInterface() {
  const [messages, setMessages] = useState<DisplayMessage[]>([])
  const [input, setInput] = useState('')
  const [isSending, setIsSending] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [lastResponse, setLastResponse] = useState<ChatResponse | null>(null)

  const textareaRef = useRef<HTMLTextAreaElement>(null)
  const messagesEndRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({
      behavior: 'smooth',
      block: 'end',
    })
  }, [messages, isSending, error])

  async function submitMessage(content: string, retry = false) {
    const trimmedContent = content.trim()

    if (!trimmedContent || isSending) return

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
    setIsSending(true)

    try {
      const response = await sendChatMessage({
        messages: conversation.slice(-50).map(({ role, content }) => ({
          role,
          content,
        })),
        provider: 'groq',
        temperature: 0.7,
        max_output_tokens: 1024,
      })

      const assistantMessage: DisplayMessage = {
        id: crypto.randomUUID(),
        role: 'assistant',
        content: response.content,
        metadata: {
          provider: response.provider,
          model: response.model,
        },
      }

      setMessages((current) => [...current, assistantMessage])
      setLastResponse(response)
    } catch (err) {
      setError(
        err instanceof Error
          ? err.message
          : 'SILVANA could not complete the request.',
      )
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

      if (!isSending) {
        void submitMessage(input)
      }
    }
  }

  function retryLastMessage() {
    const lastUserMessage = [...messages]
      .reverse()
      .find((message) => message.role === 'user')

    if (lastUserMessage) {
      void submitMessage(lastUserMessage.content, true)
    }
  }

  function clearConversation() {
    if (isSending) return

    setMessages([])
    setInput('')
    setError(null)
    setLastResponse(null)

    textareaRef.current?.focus()
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
          onClick={clearConversation}
          disabled={!hasMessages || isSending}
        >
          <span aria-hidden="true">â†º</span>
          CLEAR
        </button>
      </div>

      <div
        className="conversation-window"
        aria-live="polite"
        aria-label="Conversation with SILVANA"
      >
        {!hasMessages && (
          <div className="conversation-empty">
            <div className="empty-symbol">â—ˆ</div>

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
              disabled={isSending}
            >
              RETRY
            </button>
          </div>
        )}

        <div ref={messagesEndRef} />
      </div>

      <form className="chat-composer" onSubmit={handleSubmit}>
        <div className="composer-status">
          <span className="composer-status-dot" />
          TEXT CHANNEL ACTIVE
        </div>

        <div className="composer-input-row">
          <textarea
            ref={textareaRef}
            value={input}
            onChange={(event) => setInput(event.target.value)}
            onKeyDown={handleKeyDown}
            placeholder="Message SILVANA..."
            rows={2}
            maxLength={12000}
            disabled={isSending}
            aria-label="Type your message"
          />

          <button
            className="send-message-button"
            type="submit"
            disabled={!input.trim() || isSending}
            aria-label="Send message"
          >
            {isSending ? '...' : 'SEND â†—'}
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
