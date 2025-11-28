import { ChatProvider } from '../../state/ChatContext';
import { TopBar } from '../../components/studio/layout/TopBar';
import { StudioLeftSidebar } from '../../components/studio/layout/StudioLeftSidebar';
import { StudioRightSidebar } from '../../components/studio/layout/StudioRightSidebar';
import { ChatWindow } from '../../components/studio/chat/ChatWindow';
import './StudioPage.css';

export function StudioPage() {
  return (
    <ChatProvider>
      <div className="studio-page">
        <TopBar />
        <StudioLeftSidebar />
        <ChatWindow />
        <StudioRightSidebar />
      </div>
    </ChatProvider>
  );
}