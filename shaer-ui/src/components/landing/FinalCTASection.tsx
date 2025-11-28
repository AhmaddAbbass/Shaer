import { useNavigate } from 'react-router-dom';
import { Button } from '../common/Button';
import './FinalCTASection.css';

export function FinalCTASection() {
  const navigate = useNavigate();

  return (
    <section className="final-cta-section">
      <div className="final-cta-content">
        <h2 className="cta-title">هل أنت مستعد لتجربة Shaer-AI؟</h2>
        <Button variant="primary" onClick={() => navigate('/studio')}>
          ادخل إلى الاستوديو
        </Button>
        <p className="cta-footer">Shaer-AI • نموذج أولي</p>
      </div>
    </section>
  );
}