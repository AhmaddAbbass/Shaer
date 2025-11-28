import { Card } from '../common/Card';
import './WhatShaerDoesSection.css';

export function WhatShaerDoesSection() {
  const features = [
    {
      icon: '✨',
      title: 'ألّف',
      description: 'اكتب قصائد عربية جديدة بأوزان مختلفة وأغراض متنوعة'
    },
    {
      icon: '🔧',
      title: 'صحّح',
      description: 'صحّح الوزن والنحو في قصائدك الموجودة'
    },
    {
      icon: '🔍',
      title: 'استكشف',
      description: 'ابحث في مكتبة الشعر الكلاسيكي والحديث'
    },
    {
      icon: '📚',
      title: 'تعلّم',
      description: 'اكتشف الأوزان والأخطاء الشائعة في الشعر العربي'
    }
  ];

  return (
    <section className="what-shaer-section">
      <div className="what-shaer-content">
        <h2 className="section-title">
          ماذا يفعل Shaer-AI؟
          <div className="title-underline"></div>
        </h2>
        <div className="features-grid">
          {features.map((feature, idx) => (
            <Card key={idx}>
              <div className="feature-icon">{feature.icon}</div>
              <h3 className="feature-title">{feature.title}</h3>
              <p className="feature-description">{feature.description}</p>
            </Card>
          ))}
        </div>
      </div>
    </section>
  );
}