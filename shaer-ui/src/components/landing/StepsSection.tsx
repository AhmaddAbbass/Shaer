import { Card } from '../common/Card';
import './StepsSection.css';

export function StepsSection() {
  const steps = [
    {
      number: 1,
      title: 'صِف ما تريد',
      description: 'اكتب فكرتك أو الحالة المزاجية للقصيدة'
    },
    {
      number: 2,
      title: 'Shaer يستشير المكتبة',
      description: 'يستخدم البحث الدلالي والنماذج اللغوية'
    },
    {
      number: 3,
      title: 'تستلم قصيدة موزونة',
      description: 'قصيدة جديدة أو نسخة مصححة من قصيدتك'
    }
  ];

  return (
    <section className="steps-section">
      <div className="steps-content">
        <h2 className="section-title">
          كيف يعمل؟
          <div className="title-underline"></div>
        </h2>
        <div className="steps-grid">
          {steps.map((step) => (
            <div key={step.number} className="step-item">
              <div className="step-number">{step.number}</div>
              <Card>
                <h3 className="step-title">{step.title}</h3>
                <p className="step-description">{step.description}</p>
              </Card>
            </div>
          ))}
        </div>
      </div>
    </section>
  );
}