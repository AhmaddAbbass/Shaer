import { BrowserRouter, Routes, Route } from 'react-router-dom';
import  { LandingPage } from './pages/LandingPage/LandingPage';
import { StudioPage } from './pages/StudioPage/StudioPage';

export function AppRouter() {
  return (
    <BrowserRouter>
      <Routes>
        <Route path="/" element={<LandingPage />} />
        <Route path="/studio" element={<StudioPage />} />
      </Routes>
    </BrowserRouter>
  );
}