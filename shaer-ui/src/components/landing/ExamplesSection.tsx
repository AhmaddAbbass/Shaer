import './ExamplesSection.css';

export function ExamplesSection() {
  return (
    <section className="examples-section">
      <div className="examples-content">
        <h2 className="section-title">
          أمثلة
          <div className="title-underline"></div>
        </h2>
        <div className="examples-grid">
          <div className="example-card">
            <div className="example-header">توليد قصيدة</div>
            <div className="example-body">
              <p className="example-prompt">
                <strong>الطلب:</strong> اكتب قصيدة عن الربيع
              </p>
              <div className="example-result arabic">
                جاء الربيعُ بنورهِ المتألقِ<br />
                والطيرُ غنّى في الغصون العالقِ<br />
                والوردُ فاحَ عبيرهُ في روضةٍ<br />
                يهدي النسيمُ شذاهُ للعاشقِ
              </div>
            </div>
          </div>

          <div className="example-card">
            <div className="example-header">تصحيح قصيدة</div>
            <div className="example-body">
              <p className="example-prompt">
                <strong>قبل:</strong> <span className="arabic">أحبك يا وطني الجميل</span>
              </p>
              <p className="example-prompt">
                <strong>بعد:</strong> <span className="arabic">أحبُّكَ يا وطني الجميلا</span>
              </p>
              <p className="example-note">تم تصحيح الوزن والقافية</p>
            </div>
          </div>

          <div className="example-card">
            <div className="example-header">البحث في المكتبة</div>
            <div className="example-body">
              <p className="example-prompt">
                <strong>البحث:</strong> قصائد عن الحب في العصر العباسي
              </p>
              <div className="example-results">
                <div className="result-item">
                  <div className="result-title arabic">قف بالديار</div>
                  <div className="result-meta">
                    <span className="meta-tag">البحث: الطويل</span>
                    <span className="meta-tag">الشاعر: ابن الرومي</span>
                  </div>
                </div>
                <div className="result-item">
                  <div className="result-title arabic">أراك عصي الدمع</div>
                  <div className="result-meta">
                    <span className="meta-tag">البحث: الطويل</span>
                    <span className="meta-tag">الشاعر: أبو فراس الحمداني</span>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </section>
  );
}