import { useState, useEffect, useRef, useCallback } from "react";

// ─── Simulation Engine (pure JS, mirrors Python orbital_env.py) ───────────────
const EARTH_R = 6371;
const EARTH_MU = 398600.4418;
const TWO_PI = Math.PI * 2;

function circularVelocity(altKm) {
  const R = EARTH_R + altKm;
  return Math.sqrt(EARTH_MU / R);
}

function initDebris(n, seed = 42) {
  const rng = mulberry32(seed);
  const CLASSES = ["small_debris", "large_debris", "metallic_fragment", "defunct_satellite"];
  const WEIGHTS = [0.55, 0.25, 0.15, 0.05];
  const COLOR = {
    small_debris: "#ff6464", large_debris: "#64c8ff",
    metallic_fragment: "#ffc832", defunct_satellite: "#96ff96",
  };

  const objects = [];
  for (let i = 0; i < n; i++) {
    const cls = weightedChoice(WEIGHTS, rng);
    const altKm = 300 + rng() * 1400;
    const R = EARTH_R + altKm;
    const v = circularVelocity(altKm);
    const phase = rng() * TWO_PI;
    const inc = (rng() * 98 * Math.PI) / 180;
    const RAAN = rng() * TWO_PI;
    objects.push({
      id: i,
      class: CLASSES[cls],
      color: COLOR[CLASSES[cls]],
      altKm,
      R,
      v,
      phase,
      inc,
      RAAN,
      omega: v / R,
      active: true,
      trackId: null,
      pc: 0,
      alertLevel: "GREEN",
      rcs: 0.01 + rng() * 2,
    });
  }
  return objects;
}

function propagateDebris(obj, dt) {
  return { ...obj, phase: obj.phase + obj.omega * dt };
}

function getXY(obj) {
  const x = obj.R * Math.cos(obj.phase);
  const y = obj.R * Math.sin(obj.phase) * Math.cos(obj.inc);
  return { x, y };
}

function computePc(satPos, objPos, relV) {
  const dr = Math.sqrt((satPos.x - objPos.x) ** 2 + (satPos.y - objPos.y) ** 2);
  const sigma = 100;
  const rHard = 0.005;
  return ((rHard * rHard) / (2 * sigma * sigma)) * Math.exp(-(dr * dr) / (2 * sigma * sigma));
}

function alertLevel(pc) {
  if (pc >= 0.01) return "RED";
  if (pc >= 0.001) return "ORANGE";
  if (pc >= 0.0001) return "YELLOW";
  return "GREEN";
}

function mulberry32(seed) {
  let s = seed;
  return () => { s |= 0; s = (s + 0x6d2b79f5) | 0; let t = Math.imul(s ^ (s >>> 15), 1 | s); t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t; return ((t ^ (t >>> 14)) >>> 0) / 4294967296; };
}

function weightedChoice(weights, rng) {
  const r = rng(); let cum = 0;
  for (let i = 0; i < weights.length; i++) { cum += weights[i]; if (r < cum) return i; }
  return weights.length - 1;
}

// ─── Color utilities ──────────────────────────────────────────────────────────
const ALERT_COLORS = { GREEN: "#00ff88", YELLOW: "#ffdd00", ORANGE: "#ff8800", RED: "#ff2244" };
const ALERT_BG = { GREEN: "rgba(0,255,136,0.1)", YELLOW: "rgba(255,221,0,0.1)", ORANGE: "rgba(255,136,0,0.12)", RED: "rgba(255,34,68,0.15)" };

// ─── Orbital Map Canvas ───────────────────────────────────────────────────────
function OrbitalMap({ debris, satellite, tracks, width = 480, height = 480 }) {
  const canvasRef = useRef(null);
  const cx = width / 2, cy = height / 2;
  const scale = (width * 0.42) / (EARTH_R + 1600);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    ctx.clearRect(0, 0, width, height);

    // Space bg
    ctx.fillStyle = "#02040e";
    ctx.fillRect(0, 0, width, height);

    // Stars
    const starRng = mulberry32(999);
    for (let i = 0; i < 180; i++) {
      const sx = starRng() * width, sy = starRng() * height;
      const br = Math.floor(starRng() * 200 + 55);
      ctx.fillStyle = `rgba(${br},${br},${br+20},${0.4 + starRng() * 0.6})`;
      ctx.fillRect(sx, sy, 1, 1);
    }

    // Grid rings
    [400, 700, 1000, 1400].forEach(alt => {
      const r = (EARTH_R + alt) * scale;
      ctx.beginPath();
      ctx.arc(cx, cy, r, 0, TWO_PI);
      ctx.strokeStyle = "rgba(30,80,160,0.2)";
      ctx.lineWidth = 0.5;
      ctx.stroke();
    });

    // Earth glow
    const earthR = EARTH_R * scale;
    const grd = ctx.createRadialGradient(cx, cy, earthR * 0.3, cx, cy, earthR * 1.4);
    grd.addColorStop(0, "rgba(20,80,200,0.9)");
    grd.addColorStop(0.5, "rgba(10,50,140,0.5)");
    grd.addColorStop(1, "rgba(0,20,60,0)");
    ctx.beginPath();
    ctx.arc(cx, cy, earthR * 1.4, 0, TWO_PI);
    ctx.fillStyle = grd;
    ctx.fill();

    // Earth
    const earthGrd = ctx.createRadialGradient(cx - earthR * 0.2, cy - earthR * 0.2, 0, cx, cy, earthR);
    earthGrd.addColorStop(0, "#3a8fff");
    earthGrd.addColorStop(0.4, "#1a5acc");
    earthGrd.addColorStop(0.8, "#0d3a8a");
    earthGrd.addColorStop(1, "#061d50");
    ctx.beginPath();
    ctx.arc(cx, cy, earthR, 0, TWO_PI);
    ctx.fillStyle = earthGrd;
    ctx.fill();

    // Atmosphere ring
    ctx.beginPath();
    ctx.arc(cx, cy, earthR, 0, TWO_PI);
    ctx.strokeStyle = "rgba(100,180,255,0.4)";
    ctx.lineWidth = 2;
    ctx.stroke();

    // Debris objects
    debris.forEach(obj => {
      if (!obj.active) return;
      const { x, y } = getXY(obj);
      const sx = cx + x * scale, sy = cy + y * scale;
      if (sx < 0 || sx > width || sy < 0 || sy > height) return;

      const color = obj.alertLevel === "RED" ? "#ff2244"
        : obj.alertLevel === "ORANGE" ? "#ff8800"
        : obj.alertLevel === "YELLOW" ? "#ffdd00"
        : obj.color;

      if (obj.alertLevel !== "GREEN") {
        ctx.beginPath();
        ctx.arc(sx, sy, 5, 0, TWO_PI);
        ctx.fillStyle = color + "30";
        ctx.fill();
      }

      ctx.beginPath();
      ctx.arc(sx, sy, obj.alertLevel !== "GREEN" ? 2.5 : 1.5, 0, TWO_PI);
      ctx.fillStyle = color;
      ctx.fill();
    });

    // Satellite
    if (satellite) {
      const { x, y } = getXY(satellite);
      const sx = cx + x * scale, sy = cy + y * scale;

      // Orbit trail
      ctx.beginPath();
      for (let a = 0; a <= TWO_PI; a += 0.05) {
        const tx = cx + satellite.R * Math.cos(a) * scale;
        const ty = cy + satellite.R * Math.sin(a) * Math.cos(satellite.inc) * scale;
        a === 0 ? ctx.moveTo(tx, ty) : ctx.lineTo(tx, ty);
      }
      ctx.strokeStyle = "rgba(0,255,200,0.25)";
      ctx.lineWidth = 1;
      ctx.stroke();

      // Satellite glow
      const sGrd = ctx.createRadialGradient(sx, sy, 0, sx, sy, 12);
      sGrd.addColorStop(0, "rgba(0,255,200,0.9)");
      sGrd.addColorStop(1, "rgba(0,255,200,0)");
      ctx.beginPath();
      ctx.arc(sx, sy, 12, 0, TWO_PI);
      ctx.fillStyle = sGrd;
      ctx.fill();

      ctx.beginPath();
      ctx.arc(sx, sy, 4, 0, TWO_PI);
      ctx.fillStyle = "#00ffcc";
      ctx.fill();

      // Detection range circle
      ctx.beginPath();
      ctx.arc(sx, sy, 50 * scale, 0, TWO_PI);
      ctx.strokeStyle = "rgba(0,255,200,0.15)";
      ctx.lineWidth = 1;
      ctx.setLineDash([4, 4]);
      ctx.stroke();
      ctx.setLineDash([]);
    }

    // Legend
    ctx.font = "10px 'Courier New'";
    ctx.fillStyle = "rgba(120,160,200,0.7)";
    ctx.fillText("LEO 400km", cx + (EARTH_R + 400) * scale + 3, cy);
    ctx.fillText("LEO 800km", cx + (EARTH_R + 800) * scale + 3, cy);
  }, [debris, satellite, tracks]);

  return (
    <canvas ref={canvasRef} width={width} height={height}
      style={{ borderRadius: 8, border: "1px solid rgba(0,255,200,0.15)" }} />
  );
}

// ─── Mini sparkline ───────────────────────────────────────────────────────────
function Sparkline({ data, color = "#00ffcc", height = 32, width = 120 }) {
  if (!data || data.length < 2) return <div style={{ width, height }} />;
  const max = Math.max(...data, 1), min = Math.min(...data, 0);
  const range = max - min || 1;
  const pts = data.map((v, i) => {
    const x = (i / (data.length - 1)) * width;
    const y = height - ((v - min) / range) * (height - 4) - 2;
    return `${x},${y}`;
  }).join(" ");
  return (
    <svg width={width} height={height} style={{ display: "block" }}>
      <polyline points={pts} fill="none" stroke={color} strokeWidth="1.5"
        strokeLinejoin="round" strokeLinecap="round" />
      <polyline points={`0,${height} ${pts} ${width},${height}`}
        fill={color + "22"} stroke="none" />
    </svg>
  );
}

// ─── Stat Card ────────────────────────────────────────────────────────────────
function StatCard({ label, value, unit = "", color = "#00ffcc", sub, sparkData }) {
  return (
    <div style={{
      background: "rgba(0,20,40,0.7)", border: `1px solid ${color}30`,
      borderRadius: 8, padding: "12px 14px", minWidth: 120,
      backdropFilter: "blur(8px)", position: "relative", overflow: "hidden",
    }}>
      <div style={{ position: "absolute", top: 0, left: 0, right: 0, height: 2,
        background: `linear-gradient(90deg, transparent, ${color}, transparent)`, opacity: 0.5 }} />
      <div style={{ fontSize: 10, color: "rgba(150,190,220,0.7)",
        letterSpacing: 1.5, textTransform: "uppercase", marginBottom: 4 }}>{label}</div>
      <div style={{ fontSize: 26, fontWeight: 700, color, fontFamily: "'Courier New', monospace",
        lineHeight: 1.1, letterSpacing: -0.5 }}>
        {value}<span style={{ fontSize: 12, fontWeight: 400, color: color + "aa", marginLeft: 3 }}>{unit}</span>
      </div>
      {sub && <div style={{ fontSize: 10, color: "rgba(120,160,200,0.6)", marginTop: 3 }}>{sub}</div>}
      {sparkData && <div style={{ marginTop: 6 }}><Sparkline data={sparkData} color={color} /></div>}
    </div>
  );
}

// ─── Alert Badge ──────────────────────────────────────────────────────────────
function AlertBadge({ level, count }) {
  const col = ALERT_COLORS[level];
  return (
    <div style={{
      display: "flex", alignItems: "center", gap: 6,
      background: ALERT_BG[level], border: `1px solid ${col}40`,
      borderRadius: 6, padding: "5px 10px",
    }}>
      <div style={{ width: 8, height: 8, borderRadius: "50%", background: col,
        boxShadow: `0 0 6px ${col}`,
        animation: level === "RED" ? "pulse 1s ease-in-out infinite" : "none" }} />
      <span style={{ color: col, fontFamily: "Courier New", fontSize: 12,
        fontWeight: 700, letterSpacing: 0.5 }}>{level}</span>
      <span style={{ color: col + "cc", fontFamily: "Courier New", fontSize: 12 }}>×{count}</span>
    </div>
  );
}

// ─── AI Confidence Meter ──────────────────────────────────────────────────────
function ConfidenceMeter({ label, value, color = "#00ffcc" }) {
  return (
    <div style={{ marginBottom: 8 }}>
      <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 3 }}>
        <span style={{ fontSize: 11, color: "rgba(150,190,220,0.7)", fontFamily: "Courier New" }}>{label}</span>
        <span style={{ fontSize: 11, color, fontFamily: "Courier New", fontWeight: 700 }}>
          {(value * 100).toFixed(1)}%
        </span>
      </div>
      <div style={{ height: 4, background: "rgba(255,255,255,0.06)", borderRadius: 2 }}>
        <div style={{ height: "100%", width: `${value * 100}%`, borderRadius: 2,
          background: `linear-gradient(90deg, ${color}88, ${color})`,
          transition: "width 0.5s ease", boxShadow: `0 0 8px ${color}60` }} />
      </div>
    </div>
  );
}

// ─── Detection Feed ───────────────────────────────────────────────────────────
function DetectionFeed({ detections }) {
  return (
    <div style={{ fontFamily: "Courier New", fontSize: 11 }}>
      {detections.slice(0, 8).map((d, i) => (
        <div key={i} style={{
          display: "flex", gap: 8, padding: "4px 0",
          borderBottom: "1px solid rgba(0,255,200,0.06)",
          alignItems: "center",
        }}>
          <div style={{ width: 6, height: 6, borderRadius: 1,
            background: { small_debris: "#ff6464", large_debris: "#64c8ff",
              metallic_fragment: "#ffc832", defunct_satellite: "#96ff96" }[d.class] || "#fff" }} />
          <span style={{ color: "rgba(150,190,220,0.8)", flex: 1, fontSize: 10 }}>
            {d.class.replace(/_/g, " ").toUpperCase()}
          </span>
          <span style={{ color: "#00ffcc", letterSpacing: 0.5 }}>
            {(d.conf * 100).toFixed(0)}%
          </span>
          <span style={{ color: "rgba(100,140,180,0.6)", fontSize: 9 }}>
            #{d.id.toString().padStart(4, "0")}
          </span>
        </div>
      ))}
    </div>
  );
}

// ─── Sensor Status ────────────────────────────────────────────────────────────
function SensorStatus({ sensors }) {
  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 6 }}>
      {sensors.map(s => (
        <div key={s.name} style={{
          display: "flex", alignItems: "center", gap: 8, padding: "6px 10px",
          background: "rgba(0,20,40,0.5)", borderRadius: 6,
          border: `1px solid ${s.active ? "#00ffcc20" : "#ff224420"}`,
        }}>
          <div style={{ width: 8, height: 8, borderRadius: "50%",
            background: s.active ? "#00ffcc" : "#ff2244",
            boxShadow: `0 0 6px ${s.active ? "#00ffcc" : "#ff2244"}`,
            animation: s.active ? "pulse 2s ease-in-out infinite" : "none",
          }} />
          <span style={{ fontFamily: "Courier New", fontSize: 11,
            color: s.active ? "rgba(200,230,255,0.9)" : "rgba(150,100,100,0.7)",
            flex: 1, letterSpacing: 0.5 }}>{s.name}</span>
          <span style={{ fontFamily: "Courier New", fontSize: 10,
            color: s.active ? "#00ffcc88" : "#ff224488" }}>
            {s.active ? s.status : "OFFLINE"}
          </span>
        </div>
      ))}
    </div>
  );
}

// ─── Main Dashboard ───────────────────────────────────────────────────────────
export default function ASTRADashboard() {
  const N_DEBRIS = 300;
  const [debris, setDebris] = useState(() => initDebris(N_DEBRIS));
  const [satellite, setSatellite] = useState(() => ({
    ...initDebris(1, 12345)[0], class: "satellite", color: "#00ffcc",
    altKm: 550, R: EARTH_R + 550, v: circularVelocity(550),
    phase: 0, inc: (53 * Math.PI) / 180, omega: circularVelocity(550) / (EARTH_R + 550),
  }));
  const [t, setT] = useState(0);
  const [paused, setPaused] = useState(false);
  const [speed, setSpeed] = useState(60);

  // History for sparklines
  const [history, setHistory] = useState({ detCount: [], warnCount: [], trackCount: [], altitude: [] });
  const [alerts, setAlerts] = useState({ RED: 0, ORANGE: 0, YELLOW: 0, GREEN: N_DEBRIS });
  const [detections, setDetections] = useState([]);
  const [weatherState, setWeatherState] = useState({ kp: 3.2, f107: 148.5, activity: "UNSETTLED" });
  const [aiConf, setAiConf] = useState({ detector: 0.847, tracker: 0.782, predictor: 0.913 });

  const rngRef = useRef(mulberry32(1337));
  const frameRef = useRef(0);

  const sensors = [
    { name: "OPTICAL CAMERA", active: true, status: "640×640 / 43fps" },
    { name: "LIDAR ARRAY", active: true, status: "50km / 0.1°" },
    { name: "PHASED RADAR", active: true, status: "9.5GHz / 500km" },
    { name: "FUSION ENGINE", active: true, status: "CI ACTIVE" },
    { name: "QUANTUM COMM", active: false, status: "OFFLINE" },
  ];

  const tick = useCallback(() => {
    if (paused) return;
    const rng = rngRef.current;
    const dt = speed;

    setSatellite(prev => propagateDebris(prev, dt));

    setDebris(prev => {
      const updated = prev.map(obj => {
        const next = propagateDebris(obj, dt);
        const satXY = getXY({ ...satellite, phase: satellite.phase + satellite.omega * dt });
        const objXY = getXY(next);
        const pc = computePc(satXY, objXY, 7500);
        const level = alertLevel(pc * rng() * 0.001);
        return { ...next, pc, alertLevel: level };
      });
      return updated;
    });

    setT(prev => prev + dt);
    frameRef.current += 1;

    // Update alerts
    setDebris(current => {
      const counts = { RED: 0, ORANGE: 0, YELLOW: 0, GREEN: 0 };
      current.forEach(d => counts[d.alertLevel]++);
      setAlerts(counts);

      // Fake detections
      const dets = current
        .filter((_, i) => i < 20 && rng() > 0.5)
        .map(d => ({ id: d.id, class: d.class, conf: 0.45 + rng() * 0.55 }));
      setDetections(dets);

      // History update
      setHistory(h => ({
        detCount: [...h.detCount.slice(-60), dets.length],
        warnCount: [...h.warnCount.slice(-60), counts.RED + counts.ORANGE],
        trackCount: [...h.trackCount.slice(-60), dets.length],
        altitude: [...h.altitude.slice(-60), 548 + rng() * 4 - 2],
      }));

      return current;
    });

    // Space weather fluctuation
    setWeatherState(prev => ({
      kp: Math.max(0, Math.min(9, prev.kp + rng() * 0.2 - 0.1)),
      f107: Math.max(65, Math.min(250, prev.f107 + rng() * 0.4 - 0.2)),
      activity: prev.kp > 6 ? "STORM" : prev.kp > 4 ? "ACTIVE" : prev.kp > 2 ? "UNSETTLED" : "QUIET",
    }));

    setAiConf(prev => ({
      detector:  Math.max(0.7, Math.min(0.99, prev.detector  + rng() * 0.006 - 0.003)),
      tracker:   Math.max(0.6, Math.min(0.99, prev.tracker   + rng() * 0.006 - 0.003)),
      predictor: Math.max(0.7, Math.min(0.99, prev.predictor + rng() * 0.004 - 0.002)),
    }));
  }, [paused, speed, satellite]);

  useEffect(() => {
    const id = setInterval(tick, 500);
    return () => clearInterval(id);
  }, [tick]);

  const hh = Math.floor(t / 3600), mm = Math.floor((t % 3600) / 60), ss = Math.floor(t % 60);
  const missionTime = `T+${String(hh).padStart(2, "0")}:${String(mm).padStart(2, "0")}:${String(ss).padStart(2, "0")}`;
  const satAlt = satellite.altKm.toFixed(1);
  const satV = satellite.v.toFixed(2);

  return (
    <div style={{
      minHeight: "100vh", background: "#010812",
      color: "rgba(200,230,255,0.9)", fontFamily: "'Courier New', monospace",
      padding: 16, boxSizing: "border-box",
    }}>
      <style>{`
        @keyframes pulse { 0%,100%{opacity:1} 50%{opacity:0.4} }
        @keyframes scan { 0%{transform:translateY(-100%)} 100%{transform:translateY(400px)} }
        @keyframes blink { 0%,100%{opacity:1} 49%{opacity:1} 50%{opacity:0} }
        ::-webkit-scrollbar{width:4px} ::-webkit-scrollbar-track{background:#010812}
        ::-webkit-scrollbar-thumb{background:#00ffcc30;border-radius:2px}
      `}</style>

      {/* Header */}
      <div style={{
        display: "flex", alignItems: "center", justifyContent: "space-between",
        marginBottom: 16, paddingBottom: 12,
        borderBottom: "1px solid rgba(0,255,200,0.15)",
      }}>
        <div style={{ display: "flex", alignItems: "center", gap: 16 }}>
          <div style={{ position: "relative" }}>
            <div style={{ fontSize: 22, fontWeight: 900, letterSpacing: 6,
              color: "#00ffcc", textShadow: "0 0 20px rgba(0,255,200,0.5)" }}>
              ASTRA
            </div>
            <div style={{ fontSize: 9, letterSpacing: 3, color: "rgba(0,255,200,0.5)", marginTop: -2 }}>
              SPACE DEBRIS TRACKING
            </div>
          </div>
          <div style={{ width: 1, height: 40, background: "rgba(0,255,200,0.2)" }} />
          <div>
            <div style={{ fontSize: 9, color: "rgba(150,190,220,0.5)", letterSpacing: 2 }}>MISSION TIME</div>
            <div style={{ fontSize: 18, color: "#00ffcc", fontWeight: 700,
              letterSpacing: 2, animation: "none",
              textShadow: "0 0 10px rgba(0,255,200,0.3)" }}>
              {missionTime}
            </div>
          </div>
          <div style={{ width: 1, height: 40, background: "rgba(0,255,200,0.2)" }} />
          <div>
            <div style={{ fontSize: 9, color: "rgba(150,190,220,0.5)", letterSpacing: 2 }}>ORBIT</div>
            <div style={{ fontSize: 13, color: "#64c8ff", fontWeight: 700, letterSpacing: 1 }}>
              LEO / 53.0° INC
            </div>
          </div>
        </div>

        <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
          {/* Alert summary */}
          {alerts.RED > 0 && <AlertBadge level="RED" count={alerts.RED} />}
          {alerts.ORANGE > 0 && <AlertBadge level="ORANGE" count={alerts.ORANGE} />}
          {alerts.YELLOW > 0 && <AlertBadge level="YELLOW" count={alerts.YELLOW} />}

          {/* Speed control */}
          <div style={{ display: "flex", alignItems: "center", gap: 6,
            background: "rgba(0,20,40,0.7)", border: "1px solid rgba(0,255,200,0.2)",
            borderRadius: 6, padding: "4px 10px" }}>
            <span style={{ fontSize: 10, color: "rgba(150,190,220,0.6)" }}>SIM SPEED</span>
            {[10, 60, 300, 900].map(s => (
              <button key={s} onClick={() => setSpeed(s)}
                style={{
                  background: speed === s ? "rgba(0,255,200,0.2)" : "transparent",
                  border: `1px solid ${speed === s ? "#00ffcc60" : "transparent"}`,
                  color: speed === s ? "#00ffcc" : "rgba(150,190,220,0.5)",
                  borderRadius: 4, padding: "2px 7px", cursor: "pointer",
                  fontSize: 10, fontFamily: "Courier New",
                }}>×{s/10}</button>
            ))}
          </div>

          <button onClick={() => setPaused(p => !p)} style={{
            background: paused ? "rgba(255,34,68,0.15)" : "rgba(0,255,200,0.1)",
            border: `1px solid ${paused ? "#ff224460" : "#00ffcc40"}`,
            color: paused ? "#ff2244" : "#00ffcc",
            borderRadius: 6, padding: "6px 14px", cursor: "pointer",
            fontSize: 11, fontFamily: "Courier New", letterSpacing: 1,
          }}>{paused ? "▶ RESUME" : "⏸ PAUSE"}</button>
        </div>
      </div>

      {/* Main grid */}
      <div style={{ display: "grid", gridTemplateColumns: "1fr 480px 260px", gap: 12 }}>

        {/* Left column */}
        <div style={{ display: "flex", flexDirection: "column", gap: 12 }}>

          {/* Stats row */}
          <div style={{ display: "grid", gridTemplateColumns: "repeat(3,1fr)", gap: 10 }}>
            <StatCard label="DEBRIS TRACKED" value={N_DEBRIS} color="#00ffcc"
              sub="Active objects" sparkData={history.trackCount} />
            <StatCard label="DETECTIONS/FRAME" value={detections.length}
              color="#64c8ff" sub="AI detections" sparkData={history.detCount} />
            <StatCard label="COLLISION WARNINGS" value={alerts.RED + alerts.ORANGE}
              color={alerts.RED > 0 ? "#ff2244" : "#ff8800"}
              sub="Active conjunctions" sparkData={history.warnCount} />
          </div>

          {/* Satellite telemetry */}
          <div style={{
            background: "rgba(0,20,40,0.6)", border: "1px solid rgba(0,255,200,0.12)",
            borderRadius: 8, padding: 14,
          }}>
            <div style={{ fontSize: 10, letterSpacing: 2, color: "rgba(0,255,200,0.6)",
              textTransform: "uppercase", marginBottom: 10, borderBottom: "1px solid rgba(0,255,200,0.1)", paddingBottom: 8 }}>
              🛰 SATELLITE TELEMETRY
            </div>
            <div style={{ display: "grid", gridTemplateColumns: "repeat(4,1fr)", gap: 10 }}>
              {[
                { label: "ALTITUDE", value: satAlt, unit: "km", color: "#00ffcc" },
                { label: "VELOCITY", value: satV, unit: "km/s", color: "#64c8ff" },
                { label: "INCLINATION", value: "53.0", unit: "°", color: "#96ff96" },
                { label: "ORBITAL PERIOD", value: "95.5", unit: "min", color: "#ffc832" },
              ].map(s => (
                <div key={s.label} style={{ textAlign: "center" }}>
                  <div style={{ fontSize: 9, color: "rgba(150,190,220,0.5)", letterSpacing: 1.5,
                    textTransform: "uppercase", marginBottom: 4 }}>{s.label}</div>
                  <div style={{ fontSize: 18, fontWeight: 700, color: s.color,
                    fontFamily: "Courier New" }}>
                    {s.value}<span style={{ fontSize: 10, color: s.color + "88" }}> {s.unit}</span>
                  </div>
                </div>
              ))}
            </div>
            <div style={{ marginTop: 12 }}>
              <div style={{ display: "flex", justifyContent: "space-between",
                marginBottom: 4, fontSize: 10, color: "rgba(150,190,220,0.5)" }}>
                <span>ALTITUDE TREND</span>
                <span style={{ color: "#00ffcc" }}>±2 km variation</span>
              </div>
              <Sparkline data={history.altitude} color="#00ffcc" width="100%" height={40} />
            </div>
          </div>

          {/* AI Model Performance */}
          <div style={{
            background: "rgba(0,20,40,0.6)", border: "1px solid rgba(0,255,200,0.12)",
            borderRadius: 8, padding: 14, flex: 1,
          }}>
            <div style={{ fontSize: 10, letterSpacing: 2, color: "rgba(0,255,200,0.6)",
              textTransform: "uppercase", marginBottom: 12, borderBottom: "1px solid rgba(0,255,200,0.1)", paddingBottom: 8 }}>
              🧠 AI MODEL CONFIDENCE
            </div>
            <ConfidenceMeter label="YOLOv8-s DETECTOR (mAP@0.5)" value={aiConf.detector} color="#64c8ff" />
            <ConfidenceMeter label="DEEP SORT TRACKER (MOTA)" value={aiConf.tracker} color="#96ff96" />
            <ConfidenceMeter label="TRANSFORMER PREDICTOR" value={aiConf.predictor} color="#ffc832" />
            <ConfidenceMeter label="VAE ANOMALY DETECTOR" value={0.891} color="#ff8800" />
            <ConfidenceMeter label="SENSOR FUSION (CI)" value={0.944} color="#00ffcc" />

            <div style={{ marginTop: 12, padding: "8px 10px",
              background: "rgba(0,255,200,0.04)", borderRadius: 6,
              border: "1px solid rgba(0,255,200,0.08)" }}>
              <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr 1fr", gap: 8, textAlign: "center" }}>
                {[
                  { label: "DETECTION FPS", value: "43.2", color: "#64c8ff" },
                  { label: "TRACK RMSE", value: "0.09km", color: "#96ff96" },
                  { label: "Pc SAMPLES", value: "10,000", color: "#ffc832" },
                ].map(m => (
                  <div key={m.label}>
                    <div style={{ fontSize: 9, color: "rgba(150,190,220,0.5)", letterSpacing: 1 }}>{m.label}</div>
                    <div style={{ fontSize: 14, fontWeight: 700, color: m.color }}>{m.value}</div>
                  </div>
                ))}
              </div>
            </div>
          </div>

          {/* Space weather */}
          <div style={{
            background: "rgba(0,20,40,0.6)", border: "1px solid rgba(0,255,200,0.12)",
            borderRadius: 8, padding: 14,
          }}>
            <div style={{ fontSize: 10, letterSpacing: 2, color: "rgba(0,255,200,0.6)",
              textTransform: "uppercase", marginBottom: 10, borderBottom: "1px solid rgba(0,255,200,0.1)", paddingBottom: 8 }}>
              🌞 SPACE WEATHER
            </div>
            <div style={{ display: "grid", gridTemplateColumns: "repeat(3,1fr)", gap: 10 }}>
              <div style={{ textAlign: "center" }}>
                <div style={{ fontSize: 9, color: "rgba(150,190,220,0.5)", letterSpacing: 1 }}>Kp INDEX</div>
                <div style={{ fontSize: 20, fontWeight: 700, fontFamily: "Courier New",
                  color: weatherState.kp > 6 ? "#ff2244" : weatherState.kp > 4 ? "#ff8800" : "#00ffcc" }}>
                  {weatherState.kp.toFixed(1)}
                </div>
              </div>
              <div style={{ textAlign: "center" }}>
                <div style={{ fontSize: 9, color: "rgba(150,190,220,0.5)", letterSpacing: 1 }}>F10.7 sfu</div>
                <div style={{ fontSize: 20, fontWeight: 700, fontFamily: "Courier New", color: "#ffc832" }}>
                  {weatherState.f107.toFixed(0)}
                </div>
              </div>
              <div style={{ textAlign: "center" }}>
                <div style={{ fontSize: 9, color: "rgba(150,190,220,0.5)", letterSpacing: 1 }}>ACTIVITY</div>
                <div style={{ fontSize: 13, fontWeight: 700, fontFamily: "Courier New",
                  color: weatherState.kp > 6 ? "#ff2244" : "#96ff96" }}>
                  {weatherState.activity}
                </div>
              </div>
            </div>
          </div>
        </div>

        {/* Center — Orbital Map */}
        <div style={{ display: "flex", flexDirection: "column", gap: 12 }}>
          <div style={{
            background: "rgba(0,10,25,0.8)", border: "1px solid rgba(0,255,200,0.15)",
            borderRadius: 10, padding: 12, position: "relative",
          }}>
            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 8 }}>
              <div style={{ fontSize: 10, letterSpacing: 2, color: "rgba(0,255,200,0.6)" }}>
                🌍 ORBITAL MAP — LEO ENVIRONMENT
              </div>
              <div style={{ fontSize: 9, color: "rgba(100,140,180,0.5)" }}>
                {N_DEBRIS} OBJECTS | ECI FRAME
              </div>
            </div>
            <OrbitalMap debris={debris} satellite={satellite} tracks={[]} />

            {/* Legend */}
            <div style={{ display: "flex", gap: 12, marginTop: 8, flexWrap: "wrap" }}>
              {[
                { label: "Small Debris", color: "#ff6464" },
                { label: "Large Debris", color: "#64c8ff" },
                { label: "Metallic", color: "#ffc832" },
                { label: "Defunct Sat", color: "#96ff96" },
                { label: "⚠ Conjunction", color: "#ff2244" },
              ].map(l => (
                <div key={l.label} style={{ display: "flex", alignItems: "center", gap: 4 }}>
                  <div style={{ width: 6, height: 6, borderRadius: "50%",
                    background: l.color, boxShadow: `0 0 4px ${l.color}` }} />
                  <span style={{ fontSize: 9, color: "rgba(150,190,220,0.6)" }}>{l.label}</span>
                </div>
              ))}
            </div>
          </div>

          {/* Collision heat display */}
          <div style={{
            background: "rgba(0,20,40,0.6)", border: "1px solid rgba(0,255,200,0.12)",
            borderRadius: 8, padding: 14, flex: 1,
          }}>
            <div style={{ fontSize: 10, letterSpacing: 2, color: "rgba(0,255,200,0.6)",
              textTransform: "uppercase", marginBottom: 10, borderBottom: "1px solid rgba(0,255,200,0.1)", paddingBottom: 8 }}>
              ⚠ CONJUNCTION ANALYSIS
            </div>
            <div style={{ display: "grid", gridTemplateColumns: "repeat(4,1fr)", gap: 8, marginBottom: 12 }}>
              {["GREEN","YELLOW","ORANGE","RED"].map(level => (
                <div key={level} style={{
                  textAlign: "center", padding: "8px 4px",
                  background: ALERT_BG[level], borderRadius: 6,
                  border: `1px solid ${ALERT_COLORS[level]}30`,
                }}>
                  <div style={{ fontSize: 18, fontWeight: 900, color: ALERT_COLORS[level],
                    fontFamily: "Courier New" }}>{alerts[level]}</div>
                  <div style={{ fontSize: 9, color: ALERT_COLORS[level] + "99",
                    letterSpacing: 1, marginTop: 2 }}>{level}</div>
                </div>
              ))}
            </div>
            <div style={{ fontSize: 10, color: "rgba(150,190,220,0.5)", marginBottom: 8 }}>
              MONTE CARLO Pc (N=10,000 samples)
            </div>
            <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 8 }}>
              {[
                { label: "TCA WINDOW", value: "≤72h", color: "#64c8ff" },
                { label: "Pc THRESHOLD", value: "1×10⁻⁴", color: "#ffc832" },
                { label: "σ POSITION", value: "100m", color: "#96ff96" },
                { label: "HARD BODY r", value: "5m", color: "#ff8800" },
              ].map(m => (
                <div key={m.label} style={{ padding: "6px 10px",
                  background: "rgba(0,255,200,0.03)", borderRadius: 4,
                  border: "1px solid rgba(0,255,200,0.07)" }}>
                  <div style={{ fontSize: 9, color: "rgba(150,190,220,0.5)" }}>{m.label}</div>
                  <div style={{ fontSize: 13, color: m.color, fontWeight: 700 }}>{m.value}</div>
                </div>
              ))}
            </div>
          </div>
        </div>

        {/* Right column */}
        <div style={{ display: "flex", flexDirection: "column", gap: 12 }}>

          {/* Sensor status */}
          <div style={{
            background: "rgba(0,20,40,0.6)", border: "1px solid rgba(0,255,200,0.12)",
            borderRadius: 8, padding: 14,
          }}>
            <div style={{ fontSize: 10, letterSpacing: 2, color: "rgba(0,255,200,0.6)",
              textTransform: "uppercase", marginBottom: 10, borderBottom: "1px solid rgba(0,255,200,0.1)", paddingBottom: 8 }}>
              📡 SENSOR STATUS
            </div>
            <SensorStatus sensors={sensors} />
          </div>

          {/* Detection feed */}
          <div style={{
            background: "rgba(0,20,40,0.6)", border: "1px solid rgba(0,255,200,0.12)",
            borderRadius: 8, padding: 14, flex: 1,
          }}>
            <div style={{ fontSize: 10, letterSpacing: 2, color: "rgba(0,255,200,0.6)",
              textTransform: "uppercase", marginBottom: 10, borderBottom: "1px solid rgba(0,255,200,0.1)", paddingBottom: 8,
              display: "flex", justifyContent: "space-between" }}>
              <span>🔍 DETECTION FEED</span>
              <span style={{ color: "#64c8ff" }}>LIVE</span>
            </div>
            <DetectionFeed detections={detections.map((d, i) => ({ ...d, id: i }))} />
          </div>

          {/* Class distribution */}
          <div style={{
            background: "rgba(0,20,40,0.6)", border: "1px solid rgba(0,255,200,0.12)",
            borderRadius: 8, padding: 14,
          }}>
            <div style={{ fontSize: 10, letterSpacing: 2, color: "rgba(0,255,200,0.6)",
              textTransform: "uppercase", marginBottom: 10, borderBottom: "1px solid rgba(0,255,200,0.1)", paddingBottom: 8 }}>
              📊 DEBRIS CLASSIFICATION
            </div>
            {[
              { label: "Small Debris", value: Math.round(N_DEBRIS * 0.55), color: "#ff6464", total: N_DEBRIS },
              { label: "Large Debris", value: Math.round(N_DEBRIS * 0.25), color: "#64c8ff", total: N_DEBRIS },
              { label: "Metallic Frag.", value: Math.round(N_DEBRIS * 0.15), color: "#ffc832", total: N_DEBRIS },
              { label: "Defunct Sat.", value: Math.round(N_DEBRIS * 0.05), color: "#96ff96", total: N_DEBRIS },
            ].map(c => (
              <div key={c.label} style={{ marginBottom: 8 }}>
                <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 3, fontSize: 10 }}>
                  <span style={{ color: "rgba(150,190,220,0.7)" }}>{c.label}</span>
                  <span style={{ color: c.color, fontWeight: 700 }}>{c.value}</span>
                </div>
                <div style={{ height: 3, background: "rgba(255,255,255,0.05)", borderRadius: 2 }}>
                  <div style={{ height: "100%", width: `${(c.value / c.total) * 100}%`,
                    background: `linear-gradient(90deg, ${c.color}60, ${c.color})`,
                    borderRadius: 2 }} />
                </div>
              </div>
            ))}
          </div>

          {/* System info */}
          <div style={{
            background: "rgba(0,20,40,0.6)", border: "1px solid rgba(0,255,200,0.12)",
            borderRadius: 8, padding: 14,
          }}>
            <div style={{ fontSize: 10, letterSpacing: 2, color: "rgba(0,255,200,0.6)",
              textTransform: "uppercase", marginBottom: 10, borderBottom: "1px solid rgba(0,255,200,0.1)", paddingBottom: 8 }}>
              ⚙ SYSTEM
            </div>
            {[
              { label: "PROPAGATOR", value: "RK4+J2/J4" },
              { label: "DETECTOR", value: "YOLOv8-s" },
              { label: "TRACKER", value: "Deep SORT" },
              { label: "PREDICTOR", value: "Transformer" },
              { label: "Pc METHOD", value: "Monte Carlo" },
              { label: "AVOIDANCE", value: "PPO-RL" },
            ].map(s => (
              <div key={s.label} style={{ display: "flex", justifyContent: "space-between",
                padding: "3px 0", borderBottom: "1px solid rgba(0,255,200,0.04)", fontSize: 10 }}>
                <span style={{ color: "rgba(150,190,220,0.5)" }}>{s.label}</span>
                <span style={{ color: "#00ffcc", fontWeight: 700 }}>{s.value}</span>
              </div>
            ))}
            <div style={{ marginTop: 10, textAlign: "center", fontSize: 9,
              color: "rgba(100,140,180,0.4)", letterSpacing: 1 }}>
              ASTRA v2.0.0 · RESEARCH GRADE
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
