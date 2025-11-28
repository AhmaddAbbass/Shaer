import { useNavigate } from 'react-router-dom';
import { Button } from '../common/Button';
import './HeroSection.css';

export function HeroSection() {
  const navigate = useNavigate();

  return (
    <section className="hero-section">
      <div className="hero-content">
        <div className="hero-left">
          <div className="hero-logo">✒️</div>
          <h1 className="hero-title">Shaer-AI</h1>
          <p className="hero-subtitle">
            مولّد الشعر العربي الذكي - اكتب، صحّح، واستكشف الشعر العربي بمساعدة الذكاء الاصطناعي
          </p>
          <div className="hero-buttons">
            <Button variant="primary" onClick={() => navigate('/studio')}>
              جرّب Shaer-AI
            </Button>
            <Button
              variant="secondary"
              onClick={() => {
                document.getElementById('engine-section')?.scrollIntoView({ behavior: 'smooth' });
              }}
            >
              كيف يعمل؟
            </Button>
          </div>
        </div>
        <div className="hero-right">
          <div className="hero-studio-preview">
            <div className="preview-header">
              <div className="preview-dots">
                <span></span>
                <span></span>
                <span></span>
              </div>
            </div>
            <div className="preview-messages" dir="rtl">
              <div className="preview-message preview-message--user">
                اكتب لي قصيدة عن الحنين للوطن
              </div>
              <div className="preview-message preview-message--assistant">
                <div className="arabic">
                  أحنُّ إلى بلادي والديار<br />
                  وتشتاقُ الفؤادَ بها الأزهار<br />
                  فكم ليلٍ بعيدٍ قد مضى<br />
                  وفي القلب اشتياقٌ واحترار
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </section>
  );
}