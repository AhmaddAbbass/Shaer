import { Pill } from '../../common/Pill';
import './InspectorTabs.css';

interface InspectorTabsProps {
  activeTab: 'spec' | 'steps' | 'library';
  onTabChange: (tab: 'spec' | 'steps' | 'library') => void;
}

export function InspectorTabs({ activeTab, onTabChange }: InspectorTabsProps) {
  const tabs = [
    { id: 'spec' as const, label: 'المواصفات' },
    { id: 'steps' as const, label: 'خطوات العمل' },
    { id: 'library' as const, label: 'المكتبة' }
  ];

  return (
    <div className="inspector-tabs">
      {tabs.map((tab) => (
        <Pill
          key={tab.id}
          variant="small"
          active={activeTab === tab.id}
          onClick={() => onTabChange(tab.id)}
        >
          {tab.label}
        </Pill>
      ))}
    </div>
  );
}