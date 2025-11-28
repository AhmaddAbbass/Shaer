import { useChat } from '../../../state/ChatContext';
import './AgentStepsPanel.css';

export function AgentStepsPanel() {
  const { agentSteps } = useChat();

  if (!agentSteps || agentSteps.length === 0) {
    return (
      <div className="steps-empty">
        <p>لا توجد خطوات داخلية مسجلة لهذا الرد</p>
      </div>
    );
  }

  return (
    <div className="agent-steps-panel">
      <div className="steps-timeline">
        {agentSteps.map((step) => (
          <div key={step.step} className="step-item">
            <div className="step-marker">
              <div className="step-dot"></div>
              {step.step < agentSteps.length && <div className="step-line"></div>}
            </div>
            <div className="step-content">
              <div className="step-header">
                {step.agent} · {step.tool}
              </div>
              <div className="step-summary">{step.summary}</div>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}