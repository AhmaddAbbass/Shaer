import type { ReactNode } from 'react';
import './Pill.css';

interface PillProps {
  children: ReactNode;
  active?: boolean;
  onClick?: () => void;
  variant?: 'default' | 'small';
}

export function Pill({ children, active = false, onClick, variant = 'default' }: PillProps) {
  return (
    <button
      className={`shaer-pill shaer-pill--${variant} ${active ? 'shaer-pill--active' : ''}`}
      onClick={onClick}
    >
      {children}
    </button>
  );
}