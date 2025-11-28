import type { ChatMessage } from '../types/chat';
import type { ChatResponse } from '../types/api';
import { apiPost } from './client';

export function sendChat(messages: ChatMessage[], mode?: string) {
  return apiPost<ChatResponse>('/api/chat', { messages, mode });
}