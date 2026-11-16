
import {
  apiDelete,
  apiGet,
  apiPostWithoutBody,
  apiUpload,
} from './client'

export type DocumentScope = 'temporary' | 'persistent'

export type DocumentStatus =
  | 'uploaded'
  | 'processing'
  | 'indexed'
  | 'failed'

export interface DocumentResponse {
  id: string
  scope: DocumentScope
  conversation_id: string | null

  filename: string
  content_hash: string
  mime_type: string
  size_bytes: number

  status: DocumentStatus

  page_count: number | null
  chunk_count: number

  created_at: string | null
  last_activity_at: string | null
  expires_at: string | null
  indexed_at: string | null
}

export interface DeleteConversationDocumentsResponse {
  deleted_count: number
}

function buildDocumentQuery(
  scope: DocumentScope,
  conversationId?: string,
): string {
  const params = new URLSearchParams()
  params.set('scope', scope)

  if (scope === 'temporary') {
    if (!conversationId?.trim()) {
      throw new Error(
        'A conversation ID is required for temporary documents.',
      )
    }

    params.set('conversation_id', conversationId)
  }

  return params.toString()
}

function validateConversationId(
  scope: DocumentScope,
  conversationId?: string,
): void {
  if (scope === 'temporary' && !conversationId?.trim()) {
    throw new Error(
      'A conversation ID is required for temporary documents.',
    )
  }

  if (scope === 'persistent' && conversationId) {
    throw new Error(
      'Persistent documents cannot belong to a conversation.',
    )
  }
}

export async function listDocuments(
  scope: DocumentScope,
  conversationId?: string,
): Promise<DocumentResponse[]> {
  const query = buildDocumentQuery(scope, conversationId)

  return apiGet<DocumentResponse[]>(`/documents?${query}`)
}

export async function uploadDocument(
  file: File,
  scope: DocumentScope,
  conversationId?: string,
): Promise<DocumentResponse> {
  validateConversationId(scope, conversationId)

  const formData = new FormData()

  formData.append('file', file)
  formData.append('scope', scope)

  if (scope === 'temporary' && conversationId) {
    formData.append('conversation_id', conversationId)
  }

  return apiUpload<DocumentResponse>(
    '/documents/upload',
    formData,
  )
}

export async function reindexDocument(
  documentId: string,
  scope: DocumentScope,
  conversationId?: string,
): Promise<DocumentResponse> {
  validateConversationId(scope, conversationId)

  const query = buildDocumentQuery(scope, conversationId)

  return apiPostWithoutBody<DocumentResponse>(
    `/documents/${encodeURIComponent(documentId)}/reindex?${query}`,
  )
}

export async function deleteDocument(
  documentId: string,
  scope: DocumentScope,
  conversationId?: string,
): Promise<void> {
  validateConversationId(scope, conversationId)

  const query = buildDocumentQuery(scope, conversationId)

  return apiDelete<void>(
    `/documents/${encodeURIComponent(documentId)}?${query}`,
  )
}

export async function deleteConversationDocuments(
  conversationId: string,
): Promise<DeleteConversationDocumentsResponse> {
  if (!conversationId.trim()) {
    throw new Error('A conversation ID is required.')
  }

  return apiDelete<DeleteConversationDocumentsResponse>(
    `/documents/conversation/${encodeURIComponent(conversationId)}`,
  )
}
