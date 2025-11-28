import './EngineSection.css';

export function EngineSection() {
  return (
    <section className="engine-section" id="engine-section">
      <div className="engine-content">
        <h2 className="section-title">
          داخل المحرك
          <div className="title-underline"></div>
        </h2>
        <div className="engine-diagram">
          <div className="diagram-flow">
            <div className="flow-node">المستخدم</div>
            <div className="flow-arrow">→</div>
            <div className="flow-node">Orchestrator</div>
            <div className="flow-arrow">→</div>
            <div className="flow-node">المكتبة (RAG)</div>
            <div className="flow-arrow">→</div>
            <div className="flow-node">Yehia</div>
            <div className="flow-arrow">→</div>
            <div className="flow-node">Shaer</div>
            <div className="flow-arrow">→</div>
            <div className="flow-node">Scansion</div>
            <div className="flow-arrow">→</div>
            <div className="flow-node">المستخدم</div>
          </div>
        </div>
        <div className="engine-features">
          <div className="engine-feature">
            <div className="feature-bullet">•</div>
            <p>بحث رسومي ومتجهات على وصف القصائد</p>
          </div>
          <div className="engine-feature">
            <div className="feature-bullet">•</div>
            <p>نموذج شاعر عربي مُدرَّب بدقة</p>
          </div>
          <div className="engine-feature">
            <div className="feature-bullet">•</div>
            <p>فحص تلقائي للوزن والأسلوب</p>
          </div>
        </div>
      </div>
    </section>
  );
}