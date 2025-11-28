import { ChatMessageList } from './ChatMessageList';
import { ChatInput } from './ChatInput';
import './ChatWindow.css';

export function ChatWindow() {
  return (
    <div className="chat-window">
      <ChatMessageList />
      <ChatInput />
    </div>
  );
}