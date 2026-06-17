import { useEffect, useRef, useState } from "react";
import "./IntroAnimation.css";

const SKIP_KEY = "dia_intro_seen";

const ELEMENTS = [
  { id: "water",  color: "#4DA6FF", glow: "rgba(77,166,255,0.55)",  dir: "left"   },
  { id: "fire",   color: "#FF5A36", glow: "rgba(255,90,54,0.55)",   dir: "right"  },
  { id: "forest", color: "#4CAF50", glow: "rgba(76,175,80,0.55)",   dir: "top"    },
  { id: "earth",  color: "#C8956A", glow: "rgba(200,149,106,0.55)", dir: "bottom" },
];

function buildStreamStyle(dir, color, glow, phase) {
  const horiz = dir === "left" || dir === "right";

  const base = {
    position: "absolute",
    top: "50%",
    left: "50%",
    borderRadius: "4px",
    pointerEvents: "none",
    transition: "transform 1.35s cubic-bezier(0.16,1,0.3,1), opacity 0.35s ease",
    opacity: phase >= 3 ? 0 : 1,
    willChange: "transform",
  };

  if (horiz) {
    const offscreen = dir === "left" ? "translateX(-115%)" : "translateX(115%)";
    Object.assign(base, {
      width: "52vw",
      height: "5px",
      marginTop: "-2.5px",
      marginLeft: dir === "left" ? "-52vw" : "0px",
      background:
        dir === "left"
          ? `linear-gradient(to right, transparent 0%, ${color} 55%, #ffffff 100%)`
          : `linear-gradient(to left, transparent 0%, ${color} 55%, #ffffff 100%)`,
      boxShadow: `0 0 20px 6px ${glow}`,
      transform: phase < 1 ? offscreen : "translateX(0)",
    });
  } else {
    const offscreen = dir === "top" ? "translateY(-115%)" : "translateY(115%)";
    Object.assign(base, {
      width: "5px",
      height: "48vh",
      marginLeft: "-2.5px",
      marginTop: dir === "top" ? "-48vh" : "0px",
      background:
        dir === "top"
          ? `linear-gradient(to bottom, transparent 0%, ${color} 55%, #ffffff 100%)`
          : `linear-gradient(to top, transparent 0%, ${color} 55%, #ffffff 100%)`,
      boxShadow: `0 0 20px 6px ${glow}`,
      transform: phase < 1 ? offscreen : "translateY(0)",
    });
  }

  return base;
}

export default function IntroAnimation({ onComplete }) {
  const canvasRef     = useRef(null);
  const rafRef        = useRef(null);
  const phaseRef      = useRef(0);
  const particlesRef  = useRef([]);
  const explodedRef   = useRef(false);
  const [phase,   setPhase]   = useState(0);
  const [exiting, setExiting] = useState(false);

  // ── Skip / phase timeline ──────────────────────────────────────────────────
  useEffect(() => {
    const seen         = sessionStorage.getItem(SKIP_KEY);
    const reducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;

    if (seen || reducedMotion) {
      sessionStorage.setItem(SKIP_KEY, "1");
      onComplete();
      return;
    }
    sessionStorage.setItem(SKIP_KEY, "1");

    const advance = (n, ms) =>
      setTimeout(() => { setPhase(n); phaseRef.current = n; }, ms);

    const timers = [
      advance(1,   80),    // streams slide in
      advance(2, 1600),    // streams converge / glow pulse
      advance(3, 3100),    // collision burst + shockwave
      advance(4, 4100),    // logo materialises
      setTimeout(() => setExiting(true), 5100),
      setTimeout(() => onComplete(), 5700),
    ];
    return () => timers.forEach(clearTimeout);
  }, [onComplete]);

  // ── Canvas particle field ──────────────────────────────────────────────────
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");

    const resize = () => {
      canvas.width  = window.innerWidth;
      canvas.height = window.innerHeight;
    };
    resize();
    window.addEventListener("resize", resize);

    const TINTS = [
      "#ffffff", "#ffffff", "#ffffff", "#ffffff",
      "#4DA6FF", "#FF5A36", "#4CAF50", "#C8956A",
    ];

    particlesRef.current = Array.from({ length: 90 }, () => ({
      x:     Math.random() * window.innerWidth,
      y:     Math.random() * window.innerHeight,
      vx:    (Math.random() - 0.5) * 0.5,
      vy:    (Math.random() - 0.5) * 0.5,
      r:     Math.random() * 1.4 + 0.25,
      color: TINTS[Math.floor(Math.random() * TINTS.length)],
      alpha: Math.random() * 0.38 + 0.06,
    }));

    let frame = 0;

    const tick = () => {
      const w  = canvas.width;
      const h  = canvas.height;
      const cx = w / 2;
      const cy = h / 2;
      const p  = phaseRef.current;

      ctx.clearRect(0, 0, w, h);

      // Explosion burst: fire particles outward at phase 3
      if (p === 3 && !explodedRef.current) {
        explodedRef.current = true;
        for (const pt of particlesRef.current) {
          const dx  = pt.x - cx;
          const dy  = pt.y - cy;
          const len = Math.sqrt(dx * dx + dy * dy) || 1;
          const spd = 3 + Math.random() * 7;
          pt.vx = (dx / len) * spd;
          pt.vy = (dy / len) * spd;
        }
      }

      // Phase 4: decelerate
      if (p === 4) {
        for (const pt of particlesRef.current) {
          pt.vx *= 0.94;
          pt.vy *= 0.94;
        }
      }

      // Phase 2: drift toward center
      if (p === 2) {
        for (const pt of particlesRef.current) {
          const dx   = cx - pt.x;
          const dy   = cy - pt.y;
          const dist = Math.sqrt(dx * dx + dy * dy) || 1;
          if (dist > 60) {
            pt.vx += (dx / dist) * 0.006;
            pt.vy += (dy / dist) * 0.006;
          }
          const spd = Math.sqrt(pt.vx * pt.vx + pt.vy * pt.vy);
          if (spd > 1.6) { pt.vx = (pt.vx / spd) * 1.6; pt.vy = (pt.vy / spd) * 1.6; }
        }
      }

      for (const pt of particlesRef.current) {
        pt.x += pt.vx;
        pt.y += pt.vy;
        if (pt.x < -10) pt.x = w + 10;
        if (pt.x > w + 10) pt.x = -10;
        if (pt.y < -10) pt.y = h + 10;
        if (pt.y > h + 10) pt.y = -10;

        ctx.beginPath();
        ctx.arc(pt.x, pt.y, pt.r, 0, Math.PI * 2);
        ctx.fillStyle = pt.color;
        ctx.globalAlpha = pt.alpha;
        ctx.fill();
      }
      ctx.globalAlpha = 1;

      // Central radial glow in phases 2-4
      if (p >= 2) {
        const pulse  = p === 2 ? 0.22 + Math.sin(frame * 0.07) * 0.10 : p === 3 ? 0.55 : 0.18;
        const radius = p === 3 ? 160 : 100;
        const gr = ctx.createRadialGradient(cx, cy, 0, cx, cy, radius);
        gr.addColorStop(0, `rgba(255,255,255,${pulse})`);
        gr.addColorStop(1, "transparent");
        ctx.fillStyle = gr;
        ctx.fillRect(0, 0, w, h);
      }

      frame++;
      rafRef.current = requestAnimationFrame(tick);
    };

    tick();

    return () => {
      cancelAnimationFrame(rafRef.current);
      window.removeEventListener("resize", resize);
    };
  }, []);

  return (
    <div className={`intro-root${exiting ? " intro-exiting" : ""}`} aria-hidden="true">
      <canvas ref={canvasRef} className="intro-canvas" />

      {/* Elemental streams */}
      {ELEMENTS.map((el) => (
        <div
          key={el.id}
          style={buildStreamStyle(el.dir, el.color, el.glow, phase)}
          className={phase === 2 ? "ia-stream-pulse" : ""}
        />
      ))}

      {/* Phase 3: Collision burst + shockwave */}
      {phase >= 3 && (
        <div className="ia-collision">
          <div className="ia-shockwave" />
          <div className="ia-shockwave ia-shockwave-2" />
          <div className="ia-energy-core" />
          <p className={`ia-collision-text${phase >= 4 ? " ia-text-out" : ""}`}>
            Monitoring. Understanding. Responding.
          </p>
        </div>
      )}

      {/* Phase 4: Logo */}
      {phase >= 4 && (
        <div className="ia-logo">
          <div className="ia-emblem">
            <div className="ia-orbit ia-orbit-1" />
            <div className="ia-orbit ia-orbit-2" />
            <span
              className="material-symbols-outlined ia-logo-icon"
              style={{ fontVariationSettings: "'FILL' 1, 'wght' 300, 'GRAD' 0, 'opsz' 48" }}
            >
              radar
            </span>
          </div>
          <h1 className="ia-logo-title">VLM Disaster Analyzer</h1>
          <p className="ia-logo-subtitle">AI-Powered Disaster Intelligence</p>
        </div>
      )}
    </div>
  );
}
