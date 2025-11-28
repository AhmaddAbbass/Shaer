import type { ChatMessage } from '../../../types/chat';
import './ChatMessageBubble.css';

interface ChatMessageBubbleProps {
  message: ChatMessage;
}

export function ChatMessageBubble({ message }: ChatMessageBubbleProps) {
  const isUser = message.role === 'user';
  
  return (
    <div className={`chat-message-bubble ${isUser ? 'chat-message--user' : 'chat-message--assistant'}`}>
      <div className="message-content" dir="rtl">
        {message.content.split('\n').map((line, idx) => (
          <span key={idx}>
            {line}
            {idx < message.content.split('\n').length - 1 && <br />}
          </span>
        ))}
      </div>
    </div>
  );
}