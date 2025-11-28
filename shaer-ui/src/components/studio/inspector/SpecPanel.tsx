import { useChat } from '../../../state/ChatContext';
import './SpecPanel.css';

export function SpecPanel() {
  const { poemSpec } = useChat();

  if (!poemSpec) {
    return (
      <div className="spec-empty">
        <p>لا توجد مواصفات قصيدة نشطة لهذا الرد</p>
      </div>
    );
  }

  const fields = [
    { label: 'العنوان', value: poemSpec.poem_title },
    { label: 'البحر', value: poemSpec.poem_meter },
    { label: 'الغرض', value: poemSpec.poem_theme },
    { label: 'العصر', value: poemSpec.poem_era },
    { label: 'الشاعر/الأسلوب', value: poemSpec.poet_name },
    { label: 'عدد الأبيات', value: poemSpec.num_verses?.toString() },
    { label: 'الوصف', value: poemSpec.poem_description }
  ];

  return (
    <div className="spec-panel">
      <div className="spec-card">
        <h3 className="spec-title">مواصفات القصيدة</h3>
        <div className="spec-fields">
          {fields.map((field, idx) => 
            field.value ? (
              <div key={idx} className="spec-field">
                <div className="spec-label">{field.label}</div>
                <div className="spec-value">{field.value}</div>
              </div>
            ) : null
          )}
        </div>
      </div>
    </div>
  );
}