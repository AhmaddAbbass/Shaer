import { useChat } from '../../../state/ChatContext';
import './LibraryPanel.css';

export function LibraryPanel() {
  const { libraryItems } = useChat();

  if (!libraryItems || libraryItems.length === 0) {
    return (
      <div className="library-empty">
        <p>لم يتم استخدام أي سياق من المكتبة لهذا الرد</p>
      </div>
    );
  }

  return (
    <div className="library-panel">
      <div className="library-items">
        {libraryItems.map((item) => (
          <div key={item.poem_id} className="library-item">
            <div className="library-title">{item.title}</div>
            {item.poet_name && (
              <div className="library-poet">{item.poet_name}</div>
            )}
            <div className="library-tags">
              {item.poem_meter && (
                <span className="library-tag">{item.poem_meter}</span>
              )}
              {item.poem_era && (
                <span className="library-tag">{item.poem_era}</span>
              )}
              {item.poem_theme && (
                <span className="library-tag">{item.poem_theme}</span>
              )}
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}
