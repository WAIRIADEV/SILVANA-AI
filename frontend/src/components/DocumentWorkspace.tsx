
import { useEffect, useRef, useState } from 'react'
import {
  deleteDocument,
  listDocuments,
  reindexDocument,
  uploadDocument,
} from '../services/api/documents'
import type {
  DocumentResponse,
  DocumentScope,
  DocumentStatus,
} from '../services/api/documents'
import './DocumentWorkspace.css'

interface DocumentWorkspaceProps {
  conversationId: string
}

function formatFileSize(bytes: number): string {
  if (bytes < 1024) {
    return `${bytes} B`
  }

  if (bytes < 1024 * 1024) {
    return `${(bytes / 1024).toFixed(1)} KB`
  }

  return `${(bytes / (1024 * 1024)).toFixed(2)} MB`
}

function formatDate(value: string | null): string {
  if (!value) return 'DATE UNAVAILABLE'

  const date = new Date(value)

  if (Number.isNaN(date.getTime())) {
    return 'DATE UNAVAILABLE'
  }

  return new Intl.DateTimeFormat(undefined, {
    dateStyle: 'medium',
    timeStyle: 'short',
  }).format(date)
}

function getStatusLabel(status: DocumentStatus): string {
  switch (status) {
    case 'uploaded':
      return 'UPLOADED'
    case 'processing':
      return 'PROCESSING'
    case 'indexed':
      return 'INDEXED'
    case 'failed':
      return 'FAILED'
  }
}

function DocumentWorkspace({
  conversationId,
}: DocumentWorkspaceProps) {
  const [scope, setScope] =
    useState<DocumentScope>('temporary')

  const [documents, setDocuments] =
    useState<DocumentResponse[]>([])

  const [isLoading, setIsLoading] = useState(true)
  const [isUploading, setIsUploading] = useState(false)

  const [busyDocumentId, setBusyDocumentId] =
    useState<string | null>(null)

  const [error, setError] = useState<string | null>(null)
  const [notice, setNotice] = useState<string | null>(null)

  const [refreshKey, setRefreshKey] = useState(0)

  const fileInputRef = useRef<HTMLInputElement>(null)

  const isTemporary = scope === 'temporary'

  useEffect(() => {
    let isCurrentRequest = true

    async function loadDocuments() {
      setIsLoading(true)
      setError(null)
      setNotice(null)

      if (isTemporary && !conversationId.trim()) {
        setDocuments([])
        setIsLoading(false)
        return
      }

      try {
        const result = await listDocuments(
          scope,
          isTemporary ? conversationId : undefined,
        )

        if (isCurrentRequest) {
          setDocuments(result)
        }
      } catch (requestError) {
        if (isCurrentRequest) {
          setDocuments([])
          setError(
            requestError instanceof Error
              ? requestError.message
              : 'Unable to retrieve documents.',
          )
        }
      } finally {
        if (isCurrentRequest) {
          setIsLoading(false)
        }
      }
    }

    void loadDocuments()

    return () => {
      isCurrentRequest = false
    }
  }, [scope, conversationId, refreshKey, isTemporary])

  function refreshDocuments() {
    setRefreshKey((current) => current + 1)
  }

  async function handleFileSelection(
    event: React.ChangeEvent<HTMLInputElement>,
  ) {
    const file = event.target.files?.[0]

    if (!file) return

    setError(null)
    setNotice(null)

    try {
      setIsUploading(true)

      const result = await uploadDocument(
        file,
        scope,
        isTemporary ? conversationId : undefined,
      )

      if (result.status === 'indexed') {
        setNotice(
          `${result.filename} uploaded and indexed successfully.`,
        )
      } else if (result.status === 'failed') {
        setNotice(
          `${result.filename} was uploaded, but indexing failed. Check its status before using it for retrieval.`,
        )
      } else {
        setNotice(
          `${result.filename} uploaded. Current status: ${getStatusLabel(result.status)}.`,
        )
      }

      refreshDocuments()
    } catch (uploadError) {
      setError(
        uploadError instanceof Error
          ? uploadError.message
          : 'Document upload failed.',
      )
    } finally {
      setIsUploading(false)

      if (fileInputRef.current) {
        fileInputRef.current.value = ''
      }
    }
  }

  async function handleReindex(document: DocumentResponse) {
    setError(null)
    setNotice(null)
    setBusyDocumentId(document.id)

    try {
      const result = await reindexDocument(
        document.id,
        scope,
        isTemporary ? conversationId : undefined,
      )

      setNotice(
        `${result.filename} reindex request completed. Status: ${getStatusLabel(result.status)}.`,
      )

      refreshDocuments()
    } catch (reindexError) {
      setError(
        reindexError instanceof Error
          ? reindexError.message
          : 'Unable to reindex this document.',
      )
    } finally {
      setBusyDocumentId(null)
    }
  }

  async function handleDelete(document: DocumentResponse) {
    const confirmed = window.confirm(
      `Delete "${document.filename}" from the ${
        isTemporary ? 'conversation' : 'knowledge library'
      }? This action cannot be undone.`,
    )

    if (!confirmed) return

    setError(null)
    setNotice(null)
    setBusyDocumentId(document.id)

    try {
      await deleteDocument(
        document.id,
        scope,
        isTemporary ? conversationId : undefined,
      )

      setNotice(`${document.filename} was deleted.`)
      refreshDocuments()
    } catch (deleteError) {
      setError(
        deleteError instanceof Error
          ? deleteError.message
          : 'Unable to delete this document.',
      )
    } finally {
      setBusyDocumentId(null)
    }
  }

  const indexedCount = documents.filter(
    (document) => document.status === 'indexed',
  ).length

  const failedCount = documents.filter(
    (document) => document.status === 'failed',
  ).length

  return (
    <div className="explorer-content document-workspace">
      <div className="explorer-content-heading">
        <span className="section-eyebrow">KNOWLEDGE SYSTEMS</span>
        <h2>Documents</h2>
        <p>
          Manage source material used by SILVANA's retrieval
          and knowledge systems.
        </p>
      </div>

      <div className="document-scope-switch">
        <button
          type="button"
          className={isTemporary ? 'active' : ''}
          onClick={() => setScope('temporary')}
          aria-pressed={isTemporary}
        >
          <span>CONVERSATION</span>
          <small>TEMPORARY</small>
        </button>

        <button
          type="button"
          className={!isTemporary ? 'active' : ''}
          onClick={() => setScope('persistent')}
          aria-pressed={!isTemporary}
        >
          <span>KNOWLEDGE LIBRARY</span>
          <small>PERSISTENT</small>
        </button>
      </div>

      <div className="document-scope-description">
        {isTemporary
          ? 'Files in this workspace belong to the active conversation and are removed when that conversation is cleared. Inactive temporary documents expire according to the retention policy.'
          : 'Files in this library remain available across conversations until explicitly deleted.'}
      </div>

      <div className="document-summary-grid">
        <div className="document-summary-card">
          <span>DOCUMENTS</span>
          <strong>{documents.length}</strong>
        </div>

        <div className="document-summary-card">
          <span>INDEXED</span>
          <strong className="document-count-indexed">
            {indexedCount}
          </strong>
        </div>

        <div className="document-summary-card">
          <span>FAILED</span>
          <strong className="document-count-failed">
            {failedCount}
          </strong>
        </div>
      </div>

      <div className="document-upload-panel">
        <div className="document-upload-heading">
          <div>
            <span className="section-eyebrow">INGESTION</span>
            <h3>Add a document</h3>
          </div>

          <span className="document-upload-symbol">+</span>
        </div>

        <p>
          Supported formats: PDF, DOCX and TXT.
        </p>

        <input
          ref={fileInputRef}
          type="file"
          accept=".pdf,.docx,.txt,application/pdf,text/plain,application/vnd.openxmlformats-officedocument.wordprocessingml.document"
          className="document-file-input"
          onChange={handleFileSelection}
          disabled={isUploading}
        />

        <button
          type="button"
          className="document-upload-button"
          onClick={() => fileInputRef.current?.click()}
          disabled={isUploading || (isTemporary && !conversationId)}
        >
          {isUploading
            ? 'UPLOADING DOCUMENT...'
            : 'SELECT FILE TO UPLOAD'}
        </button>

        <span className="document-upload-hint">
          {isTemporary
            ? 'UPLOAD TARGET: ACTIVE CONVERSATION'
            : 'UPLOAD TARGET: PERSISTENT LIBRARY'}
        </span>
      </div>

      {error && (
        <div className="document-feedback document-feedback-error" role="alert">
          <span>ERROR</span>
          <p>{error}</p>
        </div>
      )}

      {notice && (
        <div className="document-feedback document-feedback-success" role="status">
          <span>NOTICE</span>
          <p>{notice}</p>
        </div>
      )}

      <div className="document-list-heading">
        <div>
          <span className="section-eyebrow">STORED MATERIAL</span>
          <h3>
            {isTemporary ? 'Conversation files' : 'Library files'}
          </h3>
        </div>

        <button
          type="button"
          className="document-refresh-button"
          onClick={refreshDocuments}
          disabled={isLoading || isUploading || busyDocumentId !== null}
          title="Refresh document list"
        >
          ↻
        </button>
      </div>

      {isLoading ? (
        <div className="document-empty-state">
          <span className="document-loading-mark" />
          <strong>RETRIEVING DOCUMENTS</strong>
          <p>Connecting to the document repository.</p>
        </div>
      ) : documents.length === 0 ? (
        <div className="document-empty-state">
          <span className="document-empty-symbol">▤</span>
          <strong>NO DOCUMENTS FOUND</strong>
          <p>
            {isTemporary
              ? 'Upload a file to make it available to this conversation.'
              : 'Your persistent knowledge library is currently empty.'}
          </p>
        </div>
      ) : (
        <div className="document-list">
          {documents.map((document) => {
            const isBusy = busyDocumentId === document.id

            return (
              <article
                className="document-item"
                key={document.id}
              >
                <div className="document-item-top">
                  <div className="document-file-icon">
                    {document.filename.split('.').pop()?.toUpperCase() ?? 'FILE'}
                  </div>

                  <div className="document-file-heading">
                    <strong title={document.filename}>
                      {document.filename}
                    </strong>

                    <span>
                      {formatFileSize(document.size_bytes)}
                      {' · '}
                      {document.mime_type}
                    </span>
                  </div>
                </div>

                <div className="document-status-row">
                  <span
                    className={`document-status document-status-${document.status}`}
                  >
                    <span />
                    {getStatusLabel(document.status)}
                  </span>

                  <span className="document-created">
                    {formatDate(document.created_at)}
                  </span>
                </div>

                <div className="document-metadata">
                  <span>
                    CHUNKS <strong>{document.chunk_count}</strong>
                  </span>

                  {document.page_count !== null && (
                    <span>
                      PAGES <strong>{document.page_count}</strong>
                    </span>
                  )}
                </div>

                {document.expires_at && (
                  <div className="document-expiry">
                    EXPIRES {formatDate(document.expires_at)}
                  </div>
                )}

                <div className="document-actions">
                  <button
                    type="button"
                    onClick={() => void handleReindex(document)}
                    disabled={isBusy || isUploading || document.status === 'processing'}
                  >
                    {isBusy ? 'WORKING...' : 'REINDEX'}
                  </button>

                  <button
                    type="button"
                    className="document-delete-button"
                    onClick={() => void handleDelete(document)}
                    disabled={isBusy || isUploading}
                  >
                    DELETE
                  </button>
                </div>
              </article>
            )
          })}
        </div>
      )}

      <div className="document-workspace-footer">
        <span>RETRIEVAL ENGINE</span>
        <strong>DOCUMENT RAG</strong>
      </div>
    </div>
  )
}

export default DocumentWorkspace
