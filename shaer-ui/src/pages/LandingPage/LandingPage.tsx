import { HeroSection } from '../../components/landing/HeroSection';
import { WhatShaerDoesSection } from '../../components/landing/WhatShaerDoesSection';
import { StepsSection } from '../../components/landing/StepsSection';
import { EngineSection } from '../../components/landing/EngineSection';
import { ExamplesSection } from '../../components/landing/ExamplesSection';
import { FinalCTASection } from '../../components/landing/FinalCTASection';
import './LandingPage.css';

export function LandingPage() {
  return (
    <div className="landing-page">
      <HeroSection />
      <WhatShaerDoesSection />
      <StepsSection />
      <EngineSection />
      <ExamplesSection />
      <FinalCTASection />
    </div>
  );
}