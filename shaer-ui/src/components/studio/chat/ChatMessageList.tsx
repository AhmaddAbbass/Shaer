import { useEffect, useRef } from 'react';
import { useChat } from '../../../state/ChatContext';
import { ChatMessageBubble } from './ChatMessageBubble';
import './ChatMessageList.css';

export function ChatMessageList() {
  const { messages, isLoading } = useChat();
  const scrollRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }
  }, [messages, isLoading]);

  return (
    <div className="chat-message-list" ref={scrollRef}>
      <div className="messages-container">
        {messages.length === 0 && (
          <div className="empty-state">
            <div className="empty-icon">✒️</div>
            <h3 className="empty-title">مرحباً بك في Shaer Studio</h3>
            <p className="empty-subtitle">ابدأ محادثة جديدة أو استخدم أحد الاقتراحات</p>
          </div>
        )}
        {messages.map((msg) => (
          <ChatMessageBubble key={msg.id} message={msg} />
        ))}
        {isLoading && (
          <div className="loading-indicator">
            <div className="loading-dots">
              <span></span>
              <span></span>
              <span></span>
            </div>
            <p className="loading-text">Shaer يفكر...</p>
          </div>
        )}
      </div>
    </div>
  );
}