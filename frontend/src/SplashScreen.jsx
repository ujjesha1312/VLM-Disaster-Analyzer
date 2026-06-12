import { useState, useEffect } from "react";

// Timings (ms)
const HOLD_MS  = 2000; // how long the splash stays fully visible
const EXIT_MS  = 600;  // fade-out duration (must match CSS)

export default function SplashScreen({ onComplete }) {
  const [exiting, setExiting] = useState(false);

  useEffect(() => {
    const exitTimer = setTimeout(() => setExiting(true), HOLD_MS);
    const doneTimer = setTimeout(() => onComplete(), HOLD_MS + EXIT_MS);
    return () => { clearTimeout(exitTimer); clearTimeout(doneTimer); };
  }, [onComplete]);

  return (
    <div className={`splash-screen${exiting ? " splash-exiting" : ""}`}>

      {/* Radial atmosphere behind the icon */}
      <div className="splash-atmosphere" aria-hidden="true" />

      <div className="splash-content">

        {/* Icon */}
        <div className="splash-icon-wrapper" aria-hidden="true">
          <div className="splash-icon-ring" />
          <div className="splash-icon">
            <span
              className="material-symbols-outlined"
              style={{ fontVariationSettings: "'FILL' 1, 'wght' 300, 'GRAD' 0, 'opsz' 48" }}
            >
              radar
            </span>
          </div>
        </div>

        {/* Word-mark */}
        <div className="splash-text-group">
          <h1 className="splash-title">Disaster Analyzer Assistant</h1>
          <p className="splash-subtitle">Preparing intelligence workspace</p>
        </div>

      </div>
    </div>
  );
}
