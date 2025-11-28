import { useChat } from '../../../state/ChatContext';
import { Pill } from '../../common/Pill';
import './StudioLeftSidebar.css';

export function StudioLeftSidebar() {
  const { mode, setMode, setSuggestedPrompt } = useChat();

  const modes = [
    { id: 'generate' as const, label: 'ألّف' },
    { id: 'fix' as const, label: 'صحّح' },
    { id: 'search' as const, label: 'ابحث' },
    { id: 'explain' as const, label: 'اشرح' }
  ];

  const suggestedPrompts = [
    'اكتب قصيدة عن الحنين',
    'صحّح هذا البيت',
    'ابحث عن قصائد الغزل',
    'ما هو بحر الكامل؟'
  ];

  return (
    <div className="studio-left-sidebar">
      <div className="sidebar-section">
        <h3 className="sidebar-title">الأنماط</h3>
        <div className="modes-list">
          {modes.map((m) => (
            <Pill
              key={m.id}
              active={mode === m.id}
              onClick={() => setMode(m.id)}
            >
              {m.label}
            </Pill>
          ))}
        </div>
      </div>

      <div className="sidebar-section">
        <h3 className="sidebar-title">اقتراحات</h3>
        <div className="prompts-list">
          {suggestedPrompts.map((prompt, idx) => (
            <button
              key={idx}
              className="prompt-chip"
              onClick={() => setSuggestedPrompt(prompt)}
            >
              {prompt}
            </button>
          ))}
        </div>
      </div>
    </div>
  );
}