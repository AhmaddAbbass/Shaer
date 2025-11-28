import { useState } from 'react';
import { InspectorTabs } from '../inspector/InspectorTabs';
import { SpecPanel } from '../inspector/SpecPanel';
import { AgentStepsPanel } from '../inspector/AgentStepsPanel';
import { LibraryPanel } from '../inspector/LibraryPanel';
import './StudioRightSidebar.css';

export function StudioRightSidebar() {
  const [activeTab, setActiveTab] = useState<'spec' | 'steps' | 'library'>('steps');

  return (
    <div className="studio-right-sidebar">
      <InspectorTabs activeTab={activeTab} onTabChange={setActiveTab} />
      <div className="inspector-content">
        {activeTab === 'spec' && <SpecPanel />}
        {activeTab === 'steps' && <AgentStepsPanel />}
        {activeTab === 'library' && <LibraryPanel />}
      </div>
    </div>
  );
}