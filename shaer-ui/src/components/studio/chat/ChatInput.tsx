import { useState, useEffect } from 'react';
import type {KeyboardEvent} from 'react';
import { useChat } from '../../../state/ChatContext';
import './ChatInput.css';

export function ChatInput() {
  const { sendMessage, isLoading, suggestedPrompt, setSuggestedPrompt } = useChat();
  const [input, setInput] = useState('');

  useEffect(() => {
    if (suggestedPrompt) {
      setInput(suggestedPrompt);
      setSuggestedPrompt('');
    }
  }, [suggestedPrompt, setSuggestedPrompt]);

  const handleSubmit = async () => {
    if (!input.trim() || isLoading) return;
    
    await sendMessage(input);
    setInput('');
  };

  const handleKeyDown = (e: KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSubmit();
    }
  };

  return (
    <div className="chat-input-container">
      <div className="chat-input-wrapper">
        <textarea
          className="chat-input"
          placeholder="اكتب رسالتك هنا..."
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={handleKeyDown}
          disabled={isLoading}
          dir="rtl"
          rows={1}
        />
        <button
          className="chat-send-button"
          onClick={handleSubmit}
          disabled={!input.trim() || isLoading}
        >
          ➤
        </button>
      </div>
    </div>
  );
}