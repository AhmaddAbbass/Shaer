import type { ReactNode } from 'react';
import './Card.css';

interface CardProps {
  children: ReactNode;
  className?: string;
  variant?: 'default' | 'emerald' | 'cream';
}

export function Card({ children, className = '', variant = 'default' }: CardProps) {
  return (
    <div className={`shaer-card shaer-card--${variant} ${className}`}>
      {children}
    </div>
  );
}