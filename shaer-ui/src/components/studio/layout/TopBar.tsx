import { useNavigate } from 'react-router-dom';
import './TopBar.css';

export function TopBar() {
  const navigate = useNavigate();

  return (
    <div className="studio-topbar">
      <div className="topbar-left" onClick={() => navigate('/')}>
        <div className="topbar-logo">✒️</div>
        <span className="topbar-title">Shaer Studio</span>
      </div>
      <div className="topbar-right">
        <span className="topbar-user">Master</span>
      </div>
    </div>
  );
}