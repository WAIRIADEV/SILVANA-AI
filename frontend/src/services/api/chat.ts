import { apiPost } from './client'
import type { ChatRequest, ChatResponse } from '../../types/api'

export async function sendChatMessage(
  request: ChatRequest,
): Promise<ChatResponse> {
  return apiPost<ChatRequest, ChatResponse>('/chat', request)
}