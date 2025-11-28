import { createContext, useContext, useState} from 'react';
import type {ReactNode} from 'react';
import type { ChatMessage } from '../types/chat';
import type { PoemSpec, PoemVersion } from '../types/poem';
import type { AgentStep, LibraryItem } from '../types/agent';
import { sendChat } from '../api/chatApi';

interface ChatContextType {
  messages: ChatMessage[];
  mode: 'generate' | 'fix' | 'search' | 'explain';
  poemSpec: PoemSpec | null;
  poemVersion: PoemVersion | null;
  agentSteps: AgentStep[];
  libraryItems: LibraryItem[];
  isLoading: boolean;
  setMode: (mode: 'generate' | 'fix' | 'search' | 'explain') => void;
  sendMessage: (text: string) => Promise<void>;
  setSuggestedPrompt: (text: string) => void;
  suggestedPrompt: string;
}

const ChatContext = createContext<ChatContextType | undefined>(undefined);

export function ChatProvider({ children }: { children: ReactNode }) {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [mode, setMode] = useState<'generate' | 'fix' | 'search' | 'explain'>('generate');
  const [poemSpec, setPoemSpec] = useState<PoemSpec | null>(null);
  const [poemVersion, setPoemVersion] = useState<PoemVersion | null>(null);
  const [agentSteps, setAgentSteps] = useState<AgentStep[]>([]);
  const [libraryItems, setLibraryItems] = useState<LibraryItem[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [suggestedPrompt, setSuggestedPrompt] = useState('');

  const sendMessage = async (text: string) => {
    const userMessage: ChatMessage = {
      id: Date.now().toString(),
      role: 'user',
      content: text,
    };

    const newMessages = [...messages, userMessage];
    setMessages(newMessages);
    setIsLoading(true);

    try {
      const response = await sendChat(newMessages, mode);
      
      const assistantMessage: ChatMessage = {
        id: (Date.now() + 1).toString(),
        role: 'assistant',
        content: response.reply,
      };

      setMessages([...newMessages, assistantMessage]);
      setPoemSpec(response.poem_spec || null);
      setPoemVersion(response.poem_version || null);
      setAgentSteps(response.agent_trace || []);
      setLibraryItems(response.library_context || []);
    } catch (error) {
      console.error('Error sending message:', error);
      const errorMessage: ChatMessage = {
        id: (Date.now() + 1).toString(),
        role: 'assistant',
        content: 'عذراً، حدث خطأ. يرجى المحاولة مرة أخرى.',
      };
      setMessages([...newMessages, errorMessage]);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <ChatContext.Provider
      value={{
        messages,
        mode,
        poemSpec,
        poemVersion,
        agentSteps,
        libraryItems,
        isLoading,
        setMode,
        sendMessage,
        setSuggestedPrompt,
        suggestedPrompt,
      }}
    >
      {children}
    </ChatContext.Provider>
  );
}

export function useChat() {
  const context = useContext(ChatContext);
  if (!context) {
    throw new Error('useChat must be used within ChatProvider');
  }
  return context;
}