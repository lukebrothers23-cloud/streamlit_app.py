import React, { useMemo, useState, useEffect, useRef } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { Canvas } from "@react-three/fiber";
import { OrbitControls } from "@react-three/drei";

/**
 * Junkyard Wars – Playable React Prototype (single file)
 * v7 – 3D Viewport + Error Fix + More Tests
 *
 * Changes
 * - FIX: removed the unterminated string culprit from earlier revs (no CSV export string concat).
 * - NEW: 3D rig viewport using react-three-fiber (@react-three/fiber + drei) with OrbitControls.
 * - Loot "spark" animation in 3D when new parts drop.
 * - Kept compact core loop: campaign ladder, push-your-luck scavenging, pity timer, forging w/ stabilizers,
 *   DnD equip, bank-or-bust, boss battles, reward modal, autosave.
 * - Tests: kept original self-tests and ADDED extra checks (reforge outcomes + pity path sanity).
 *
 * Install deps (in your Vite React project):
 *   npm i @react-three/fiber three @react-three/drei framer-motion
 */

// ---------- Utils ----------
const clamp = (v, a, b) => Math.max(a, Math.min(b, v));
const rand = (min, max) => Math.random() * (max - min) + min;
const choice = (arr) => arr[Math.floor(Math.random() * arr.length)];

// ---------- Data ----------
const RARITIES = [
  { key: "common", name: "Common", color: "text-zinc-300", base: 0.5, mult: [0.8, 1.0], hex: "#9ca3af" },
  { key: "uncommon", name: "Uncommon", color: "text-green-300", base: 0.3, mult: [1.0, 1.2], hex: "#86efac" },
  { key: "rare", name: "Rare", color: "text-blue-300", base: 0.15, mult: [1.2, 1.5], hex: "#93c5fd" },
  { key: "epic", name: "Epic", color: "text-purple-300", base: 0.04, mult: [1.5, 1.8], hex: "#d8b4fe" },
  { key: "legendary", name: "Legendary", color: "text-amber-300", base: 0.01, mult: [1.8, 2.4], hex: "#fcd34d" },
];
const rarityIndex = (k) => RARITIES.findIndex((r) => r.key === k);
const rarityHex = (k) => (RARITIES[rarityIndex(k)]?.hex || "#e5e7eb");

const PART_SLOTS = ["Chassis", "Engine", "Weapon", "Aux"];
const EFFECTS = ["Shock", "Ricochet", "Turbo", "Pierce", "Vamp", "Coolant"];
const YARDS = [
  { key: 0, name: "Outskirts", threshold: 0, bias: 0.0 },
  { key: 1, name: "Acid Flats", threshold: 200, bias: 0.08 },
  { key: 2, name: "Smog City", threshold: 300, bias: 0.12 },
  { key: 3, name: "Glass Desert", threshold: 400, bias: 0.18 },
  { key: 4, name: "Warlord's Pit", threshold: 500, bias: 0.24 },
];

// ---------- Loot & Craft ----------
function chances(runStats, bias = 0) {
  let p = RARITIES.map((r) => r.base);
  if (runStats.noRareStreak > 0) p[2] += 0.005 * runStats.noRareStreak;
  p[3] += 0.01 * Math.floor(runStats.noEpicStreak / 4);
  if (runStats.last3Low) p[2] += 0.05;
  p[2] += 0.05 * bias; p[3] += 0.03 * bias; p[4] += 0.01 * bias;
  if (runStats.runsSinceLegendary >= 12) p = [0, 0, 0, 0, 1];
  const s = p.reduce((a, b) => a + b, 0); return p.map((x) => x / s);
}
function rollRarity(runStats, bias) {
  const ps = chances(runStats, bias), r = Math.random();
  let a = 0; for (let i = 0; i < ps.length; i++) { a += ps[i]; if (r <= a) return RARITIES[i]; }
  return RARITIES[0];
}
function rollPerks(rarityKey) {
  const idx = rarityIndex(rarityKey);
  const count = [1, 1 + (Math.random() < 0.5), 2, 2 + (Math.random() < 0.5), 3][idx];
  const pool = [...EFFECTS]; const out = [];
  for (let i = 0; i < count; i++) { const j = Math.floor(Math.random() * pool.length); out.push(pool.splice(j, 1)[0]); }
  return out;
}
function rollMult(r) { const [a, b] = r.mult; return rand(a, b); }
function genPart(runStats, bias) {
  const rarity = rollRarity(runStats, bias); const slot = choice(PART_SLOTS); const m = rollMult(rarity);
  const base = { Chassis: { power: 40, eff: 60 }, Engine: { power: 55, eff: 45 }, Weapon: { power: 70, eff: 30 }, Aux: { power: 25, eff: 75 } }[slot];
  const power = Math.round(base.power * m + rand(-3, 3)); const efficiency = Math.round(base.eff * (2 - m) + rand(-3, 3));
  return { id: `${slot}-${Math.random().toString(36).slice(2, 7)}`, name: `${rarity.name} ${slot}`, slot, rarity: rarity.key, power, efficiency, perks: rollPerks(rarity.key), forged: 0 };
}
function genPartAtLeast(minKey, rs, bias) {
  const min = rarityIndex(minKey); let p;
  for (let i = 0; i < 30; i++) { p = genPart(rs, bias); if (rarityIndex(p.rarity) >= min) break; }
  return p;
}
function reforge(part, stabs = 0) {
  let breakC = clamp(0.15 - 0.07 * stabs, 0.01, 0.15);
  const r = Math.random();
  if (r < breakC) return { outcome: "break", part: { ...part, broken: true } };
  if (r < breakC + 0.25) return { outcome: "boost", part: { ...part, power: Math.round(part.power * rand(1.07, 1.12)), efficiency: Math.round(part.efficiency * rand(1.07, 1.12)), forged: part.forged + 1 } };
  if (r < breakC + 0.25 + 0.6) {
    const rarity = RARITIES[rarityIndex(part.rarity)], m = rollMult(rarity);
    return { outcome: "success", part: { ...part, power: Math.round((40 + part.power) * 0.5 * m), efficiency: Math.round((40 + part.efficiency) * 0.5 * (2 - m)), forged: part.forged + 1 } };
  }
  return { outcome: "fail", part: { ...part, forged: part.forged + 1 } };
}

// ---------- Battle ----------
function score(parts) { const p = parts.reduce((s, x) => s + (x?.power || 0), 0); const e = parts.reduce((s, x) => s + (x?.efficiency || 0), 0); return Math.round(p * (1 + e / 400)); }
function simulate(parts, diff = 1) { const ps = score(parts); const ai = Math.round(ps * rand(0.85, 1.15) * diff); return { result: ps >= ai ? "win" : "loss", ps, ai }; }

// ---------- Small UI bits ----------
const Badge = ({ children }) => (<span className="inline-flex items-center rounded-full px-2 py-0.5 text-xs border border-white/10">{children}</span>);
const PartCard = ({ part, selected, onClick, draggable, onDragStart, onDragEnd, actions }) => part ? (
  <motion.div onClick={onClick} draggable={draggable} onDragStart={onDragStart} onDragEnd={onDragEnd}
    initial={{ opacity: 0, scale: .95 }} animate={{ opacity: 1, scale: 1 }} className={`rounded-xl p-3 bg-zinc-800/60 border border-zinc-700/50 ${selected ? 'ring-2 ring-amber-400' : ''}`}>
    <div className="flex justify-between text-sm"><div><span className={
      part.rarity === 'legendary' ? 'text-amber-300' : part.rarity === 'epic' ? 'text-purple-300' : part.rarity === 'rare' ? 'text-blue-300' : part.rarity === 'uncommon' ? 'text-green-300' : 'text-zinc-300'
    }>{part.rarity}</span> • {part.slot}</div><Badge>{part.forged > 0 ? `✦x${part.forged}` : 'New'}</Badge></div>
    <div className="text-sm text-zinc-300 mt-1">PWR <b>{part.power}</b> • EF <b>{part.efficiency}</b></div>
    {part.perks?.length > 0 && (<div className="mt-1 text-xs text-zinc-400">{part.perks.join(', ')}</div>)}
    {actions}
    {part.broken && (<div className="absolute inset-0 grid place-items-center text-rose-300 text-sm">Broken</div>)}
  </motion.div>
) : null;

// ---------- 3D Components ----------
function Rig3D({ build, reveals = [] }) {
  const parts = useMemo(() => PART_SLOTS.map((s) => build[s]).filter(Boolean), [build]);
  const groupRef = useRef();
  // Simple idle rotation
  useFrameSafe(() => { if (groupRef.current) groupRef.current.rotation.y += 0.003; });

  return (
    <group ref={groupRef} position={[0, 0.25, 0]}>
      {/* Chassis */}
      {build.Chassis && (
        <mesh position={[0, 0.6, 0]} castShadow receiveShadow>
          <boxGeometry args={[1.8, 0.4, 1]} />
          <meshStandardMaterial metalness={0.6} roughness={0.4} color={rarityHex(build.Chassis.rarity)} />
        </mesh>
      )}
      {/* Engine */}
      {build.Engine && (
        <mesh position={[0, 0.9, -0.1]} rotation={[Math.PI / 2, 0, 0]} castShadow>
          <cylinderGeometry args={[0.3, 0.3, 0.8, 16]} />
          <meshStandardMaterial metalness={0.7} roughness={0.3} color={rarityHex(build.Engine.rarity)} />
        </mesh>
      )}
      {/* Weapon */}
      {build.Weapon && (
        <mesh position={[0.7, 0.85, 0]} castShadow>
          <boxGeometry args={[0.9, 0.2, 0.2]} />
          <meshStandardMaterial color={rarityHex(build.Weapon.rarity)} />
        </mesh>
      )}
      {/* Aux */}
      {build.Aux && (
        <mesh position={[-0.9, 0.75, 0.1]} castShadow>
          <sphereGeometry args={[0.2, 20, 16]} />
          <meshStandardMaterial color={rarityHex(build.Aux.rarity)} emissive={rarityHex(build.Aux.rarity)} emissiveIntensity={0.2} />
        </mesh>
      )}

      {/* Loot sparks */}
      {reveals.map((r, i) => (
        <LootSpark key={r.id} color={rarityHex(r.rarity)} index={i} />
      ))}
    </group>
  );
}

function LootSpark({ color = "#fff", index = 0 }) {
  const ref = useRef();
  const t0 = useRef(Math.random() * 1000);
  useFrameSafe((state) => {
    const t = (state.clock.getElapsedTime() * 1000 + t0.current) / 1000;
    if (!ref.current) return;
    ref.current.position.set(Math.sin(t + index) * 0.8, 0.6 + Math.abs(Math.sin(t * 2)) * 0.8, Math.cos(t + index) * 0.8);
    ref.current.rotation.y += 0.1;
    const s = 0.4 + 0.3 * Math.sin(t * 3 + index);
    ref.current.scale.setScalar(s);
  });
  return (
    <mesh ref={ref} castShadow>
      <icosahedronGeometry args={[0.15, 0]} />
      <meshStandardMaterial color={color} emissive={color} emissiveIntensity={0.6} metalness={0.3} roughness={0.3} />
    </mesh>
  );
}

function useFrameSafe(fn) {
  // Lazy import guard: only call useFrame if Canvas has mounted
  const frameRef = useRef(null);
  const r3f = requireSafe("@react-three/fiber");
  if (r3f?.useFrame) {
    const { useFrame } = r3f;
    if (!frameRef.current) frameRef.current = useFrame(fn);
  }
}
function requireSafe(mod) { try { return require(mod); } catch { return {}; } }

// ---------- Persistence ----------
const SAVE_KEY = "jw-save-compact-3d";
const saveNow = (state) => { try { localStorage.setItem(SAVE_KEY, JSON.stringify(state)); } catch {} };
const loadNow = () => { try { const r = localStorage.getItem(SAVE_KEY); return r ? JSON.parse(r) : null; } catch { return null; } };

// ---------- App ----------
export default function App() {
  // meta
  const [runs, setRuns] = useState(0), [runsSinceLegendary, setRSL] = useState(0), [tokens, setTokens] = useState(0), [scrap, setScrap] = useState(100), [stabs, setStabs] = useState(0);
  const [yard, setYard] = useState(0), [unlocked, setUnlocked] = useState([0]), [fights, setFights] = useState(0);
  // run
  const [inRun, setInRun] = useState(false), [depth, setDepth] = useState(0), [cart, setCart] = useState([]), [message, setMsg] = useState("");
  // collection
  const [inv, setInv] = useState([]), [build, setBuild] = useState({ Chassis: null, Engine: null, Weapon: null, Aux: null }), [selId, setSel] = useState(null);
  const [dragId, setDrag] = useState(null); const dragging = useMemo(() => inv.find(p => p.id === dragId) || null, [dragId, inv]);
  const [reveals, setReveals] = useState([]);
  // battle
  const [reward, setReward] = useState(null);

  // derived
  const buildArr = useMemo(() => PART_SLOTS.map(s => build[s]).filter(Boolean), [build]);
  const buildScore = useMemo(() => score(buildArr), [buildArr]);
  const nextIsBoss = (fights + 1) % 5 === 0;
  const bias = YARDS[yard].bias;

  // unlock
  useEffect(() => { const newly = [...new Set([...unlocked, ...YARDS.filter(y => buildScore >= y.threshold).map(y => y.key)])]; if (newly.length !== unlocked.length) setUnlocked(newly); }, [buildScore]);

  // autosave
  useEffect(() => { saveNow({ runs, runsSinceLegendary, tokens, scrap, stabs, yard, unlocked, fights, inv, build }); }, [runs, runsSinceLegendary, tokens, scrap, stabs, yard, unlocked, fights, inv, build]);
  useEffect(() => {
    const s = loadNow();
    if (s) {
      setRuns(s.runs || 0); setRSL(s.runsSinceLegendary || 0); setTokens(s.tokens || 0); setScrap(s.scrap ?? 100); setStabs(s.stabs || 0); setYard(s.yard || 0); setUnlocked(s.unlocked || [0]); setFights(s.fights || 0); setInv(s.inv || []); setBuild(s.build || { Chassis: null, Engine: null, Weapon: null, Aux: null });
    } else {
      // starter kit
      setInv(PART_SLOTS.map(slot => ({ id: `${slot}-starter`, name: `Starter ${slot}`, slot, rarity: 'common', power: 30 + Math.floor(Math.random() * 10), efficiency: 30 + Math.floor(Math.random() * 10), perks: [] })));
      setMsg('Starter kit granted.');
    }
  }, []);

  // run stats
  const runStats = useMemo(() => {
    const last3Low = cart.slice(-3).every(p => p && (p.rarity === 'common' || p.rarity === 'uncommon'));
    let noRare = 0, noEpic = 0; for (let i = cart.length - 1; i >= 0; i--) { if (['rare', 'epic', 'legendary'].includes(cart[i].rarity)) break; noRare++; }
    for (let i = cart.length - 1; i >= 0; i--) { if (['epic', 'legendary'].includes(cart[i].rarity)) break; noEpic++; }
    return { noRareStreak: noRare, noEpicStreak: noEpic, last3Low, runsSinceLegendary };
  }, [cart, runsSinceLegendary]);

  // actions
  const start = () => { if (inRun) return; setInRun(true); setDepth(0); setCart([]); setReveals([]); setMsg('Entered the scrapyard. Push deeper for better loot.'); };
  const bank = () => { if (!inRun) return; setInRun(false); setRuns(r => r + 1); setInv(v => [...cart, ...v]); if (cart.some(p => p.rarity === 'legendary')) setRSL(0); else setRSL(x => x + 1); setCart([]); setDepth(0); setMsg(`You escaped with ${cart.length} parts!`); };
  const bust = () => { if (!inRun) return; setInRun(false); setRuns(r => r + 1); setRSL(x => x + 1); setCart([]); setDepth(0); setMsg('Defeated! Loot lost.'); };
  const push = () => {
    if (!inRun) return; setDepth(d => d + 1);
    const drops = Math.floor(rand(1, 3)); const arr = []; for (let i = 0; i < drops; i++) arr.push(genPart(runStats, bias));
    if (depth === 0 && !arr.some(p => ['rare', 'epic', 'legendary'].includes(p.rarity))) arr[0] = genPartAtLeast('rare', runStats, bias); // first push guarantee
    const space = Math.max(0, (6) - cart.length);
    if (space === 0) { setMsg('Cart full! Bank or scrap.'); return; }
    const toAdd = arr.slice(0, space);
    setCart(c => [...c, ...toAdd]);
    setReveals(toAdd.map(p => ({ id: p.id, rarity: p.rarity })));
  };
  const equip = (part) => { setBuild(b => ({ ...b, [part.slot]: part })); setInv(v => v.filter(p => p.id !== part.id)); setSel(null); };
  const unequip = (slot) => { const p = build[slot]; if (!p) return; setInv(v => [p, ...v]); setBuild(b => ({ ...b, [slot]: null })); };
  const scrapPart = (part, from = 'inventory') => { const base = [3, 6, 12, 24, 48][rarityIndex(part.rarity)]; setScrap(s => s + base + part.forged * 4); if (from === 'inventory') setInv(v => v.filter(p => p.id !== part.id)); else setCart(c => c.filter(p => p.id !== part.id)); if (selId === part.id) setSel(null); };
  const doReforge = (part) => { const cost = 18 + part.forged * 6; if (scrap < cost) { setMsg('Not enough scrap.'); return; } setScrap(s => s - cost); const r = reforge(part, stabs); if (stabs > 0) setStabs(s => s - 1); if (r.outcome === 'break') { scrapPart({ ...part }, 'inventory'); setMsg('Forge broke the part.'); } else { setInv(v => v.map(p => p.id === part.id ? r.part : p)); setBuild(b => { const s = part.slot; return b[s]?.id === part.id ? { ...b, [s]: r.part } : b; }); setMsg(r.outcome === 'boost' ? 'Boosted stats!' : r.outcome === 'success' ? 'Reforged!' : 'Forge failed.'); } };
  const fight = () => {
    if (buildArr.length < 4) { setMsg('Fill all slots first.'); return; }
    const diff = 1 + yard * 0.15 + Math.min(0.5, runs * 0.03) + (nextIsBoss ? 0.25 : 0);
    const out = simulate(buildArr, diff); setFights(f => f + 1);
    if (out.result === 'win') {
      setTokens(t => t + 1); setScrap(s => s + 20);
      const drop = nextIsBoss ? genPartAtLeast('rare', runStats, bias) : (Math.random() < 0.4 ? genPart(runStats, bias) : null);
      if (drop) setInv(v => [drop, ...v]);
      setReward({ win: true, boss: nextIsBoss, ps: out.ps, ai: out.ai, drop });
      setMsg(`Victory! ${nextIsBoss ? 'Boss down. ' : ''}+1 token, +20 scrap.`);
    } else {
      setReward({ win: false, boss: nextIsBoss, ps: out.ps, ai: out.ai, drop: null });
      setMsg('Defeat. Try reforging or scavenging.');
    }
  };

  // clear reveal sparks
  useEffect(() => { if (reveals.length === 0) return; const t = setTimeout(() => setReveals([]), 1200); return () => clearTimeout(t); }, [reveals]);

  // -------- Tests (kept originals + added) --------
  const runTests = () => {
    // Original checks
    const rs = { noRareStreak: 0, noEpicStreak: 0, last3Low: false, runsSinceLegendary: 0 };
    const sum = chances(rs, 0).reduce((a, b) => a + b, 0); if (Math.abs(sum - 1) > 1e-6) return alert('Test fail: chance sum');
    const p1 = genPart(rs, 0); if (!p1 || !p1.id) return alert('Test fail: genPart');
    const s1 = score([{ power: 10, efficiency: 0 }]); const s2 = score([{ power: 10, efficiency: 100 }]); if (!(s2 > s1)) return alert('Test fail: score monotonic');
    // Added checks
    const rf = reforge({ id: 't', slot: 'Weapon', rarity: 'common', power: 10, efficiency: 10, forged: 0 }, 1);
    if (!['break', 'boost', 'success', 'fail'].includes(rf.outcome)) return alert('Test fail: reforge outcome');
    // Pity path sanity: if we force runsSinceLegendary >= 12, chances should be 100% legendary
    const pity = chances({ ...rs, runsSinceLegendary: 12, noEpicStreak: 0, noRareStreak: 0 }, 0);
    if (pity[4] !== 1) return alert('Test fail: pity 100% legendary');
    alert('All tests passed.');
  };

  return (
    <div className="min-h-screen w-full bg-gradient-to-b from-zinc-950 via-zinc-900 to-zinc-950 text-zinc-100 p-4 md:p-8">
      <header className="mb-4 flex flex-col md:flex-row md:items-end md:justify-between gap-4">
        <div>
          <h1 className="text-2xl md:text-3xl font-extrabold tracking-tight">Junkyard Wars</h1>
          <p className="text-zinc-300 text-sm">Scavenge • Forge • Dominate (3D)</p>
        </div>
        <div className="flex flex-col md:flex-row gap-2 md:items-center">
          <div className="flex flex-wrap items-center gap-2 text-sm">
            <Badge>Yard: {YARDS[yard].name}</Badge>
            <Badge>Runs: {runs}</Badge>
            <Badge>Since Leg: {runsSinceLegendary}</Badge>
            <Badge>Tokens: {tokens}</Badge>
            <Badge>Scrap: {scrap}</Badge>
            <Badge>Stabs: {stabs}</Badge>
          </div>
          <div className="flex flex-wrap items-center gap-2 text-xs">
            <button onClick={() => { const s = loadNow(); if (s) { setRuns(s.runs || 0); setRSL(s.runsSinceLegendary || 0); setTokens(s.tokens || 0); setScrap(s.scrap ?? 100); setStabs(s.stabs || 0); setYard(s.yard || 0); setUnlocked(s.unlocked || [0]); setFights(s.fights || 0); setInv(s.inv || []); setBuild(s.build || { Chassis: null, Engine: null, Weapon: null, Aux: null }); setMsg('Loaded.'); } else setMsg('No save.'); }} className="px-2 py-1 rounded-lg bg-zinc-700 hover:bg-zinc-600">Load</button>
            <button onClick={() => { saveNow({ runs, runsSinceLegendary, tokens, scrap, stabs, yard, unlocked, fights, inv, build }); setMsg('Saved.'); }} className="px-2 py-1 rounded-lg bg-indigo-700 hover:bg-indigo-600">Save</button>
            <button onClick={runTests} className="px-2 py-1 rounded-lg bg-zinc-700 hover:bg-zinc-600">Run Tests</button>
          </div>
        </div>
      </header>

      {/* 3D Viewport */}
      <div className="mb-6 rounded-2xl border border-zinc-700/50 bg-zinc-900/60 overflow-hidden">
        <div className="flex items-center justify-between px-3 py-2 text-xs text-zinc-400 border-b border-zinc-700/50">
          <div>3D Rig View (drag to orbit, scroll to zoom)</div>
          <div>Build Score: <b className="text-zinc-200">{buildScore}</b></div>
        </div>
        <div style={{ height: 360 }}>
          <Canvas shadows camera={{ position: [3.2, 2.2, 3.2], fov: 50 }}>
            <ambientLight intensity={0.6} />
            <directionalLight position={[3, 5, 3]} intensity={0.9} castShadow shadow-mapSize-width={1024} shadow-mapSize-height={1024} />
            <Rig3D build={build} reveals={reveals} />
            {/* Ground */}
            <mesh rotation={[-Math.PI / 2, 0, 0]} position={[0, 0, 0]} receiveShadow>
              <planeGeometry args={[20, 20]} />
              <shadowMaterial opacity={0.25} />
            </mesh>
            <OrbitControls enablePan={false} maxPolarAngle={Math.PI / 2 - 0.05} minDistance={2} maxDistance={8} />
          </Canvas>
        </div>
      </div>

      {/* Top row */}
      <div className="grid md:grid-cols-3 gap-4 mb-6">
        {/* Run */}
        <div className="bg-zinc-900/60 rounded-2xl p-4 border border-zinc-700/50">
          <h2 className="font-semibold mb-2">Run</h2>
          <div className="flex flex-wrap gap-2 mb-2">
            {!inRun ? (<button onClick={start} className="px-3 py-2 rounded-xl bg-emerald-600 hover:bg-emerald-500">Enter</button>) : (<>
              <button onClick={push} className="px-3 py-2 rounded-xl bg-amber-600 hover:bg-amber-500">Push</button>
              <button onClick={bank} className="px-3 py-2 rounded-xl bg-zinc-700 hover:bg-zinc-600">Bank</button>
              <button onClick={bust} className="px-3 py-2 rounded-xl bg-rose-700 hover:bg-rose-600">Give Up</button>
            </>)}
          </div>
          <div className="text-sm text-zinc-300 space-y-1">
            <div>Depth: <b>{depth}</b></div>
            <div>Cart: <b>{cart.length}</b> / 6</div>
            <div className="text-xs text-zinc-400">Leave to bank loot. Defeat loses it.</div>
          </div>
          <div className="mt-3 grid grid-cols-2 gap-2">
            {cart.map((p) => <PartCard key={p.id} part={p} actions={<div className="mt-2"><button onClick={() => scrapPart(p, 'cart')} className="px-2 py-1 rounded bg-zinc-700 text-xs">Scrap</button></div>} />)}
          </div>
        </div>

        {/* Build */}
        <div className="bg-zinc-900/60 rounded-2xl p-4 border border-zinc-700/50">
          <h2 className="font-semibold mb-2">Build</h2>
          <div className="grid grid-cols-2 gap-2">
            {PART_SLOTS.map((slot) => (
              <div key={slot}
                   onDragOver={(e) => { if (dragging && dragging.slot === slot) { e.preventDefault(); e.dataTransfer.dropEffect = 'move'; } }}
                   onDrop={(e) => { e.preventDefault(); if (dragging && dragging.slot === slot) { equip(dragging); setDrag(null); } }}
                   className="rounded-xl p-3 bg-zinc-800/50 border border-zinc-700/50">
                <div className="flex justify-between items-center mb-1 text-sm">
                  <div>{slot}</div>
                  {build[slot] && (<button onClick={() => unequip(slot)} className="text-xs px-2 py-1 rounded bg-zinc-700">Unequip</button>)}
                </div>
                <PartCard part={build[slot]} />
              </div>
            ))}
          </div>
          <div className="mt-3"><div className="text-xs text-zinc-400 mb-1">Build Score</div><div className="h-2 bg-zinc-800 rounded-full"><div className="h-full bg-amber-400" style={{ width: `${clamp((buildScore / 600) * 100, 0, 100)}%` }} /></div></div>
          <div className="mt-3 flex items-center gap-2 flex-wrap">
            <button onClick={fight} className="px-3 py-2 rounded-xl bg-indigo-600 hover:bg-indigo-500">Battle</button>
            <Badge>Next: {nextIsBoss ? 'Boss' : 'Normal'}</Badge>
            <div className="flex gap-2 text-xs">
              {YARDS.map((y) => {
                const unlockedY = unlocked.includes(y.key);
                return (
                  <button key={y.key} disabled={!unlockedY} onClick={() => setYard(y.key)} className={`px-2 py-1 rounded ${yard === y.key ? 'bg-amber-700' : unlockedY ? 'bg-zinc-700 hover:bg-zinc-600' : 'bg-zinc-800 text-zinc-500 cursor-not-allowed'}`}>{y.name}</button>
                );
              })}
            </div>
          </div>
        </div>

        {/* Inventory / Forge */}
        <div className="bg-zinc-900/60 rounded-2xl p-4 border border-zinc-700/50">
          <h2 className="font-semibold mb-2">Forge & Inventory</h2>
          <div className="flex items-center gap-2 mb-2 text-sm">
            <button onClick={() => { const c = 25; if (scrap < c) return setMsg('Need 25 scrap'); setScrap(s => s - c); setStabs(s => s + 1); }} className="px-2 py-1 rounded bg-zinc-700">Buy Stabilizer (25)</button>
            <div className="text-xs text-zinc-400">Use: reduces break chance for next reforge</div>
          </div>
          <div className="grid grid-cols-2 gap-2 max-h-64 overflow-auto pr-1">
            {inv.map((p) => (
              <PartCard key={p.id}
                part={p}
                selected={selId === p.id}
                onClick={() => setSel((id) => id === p.id ? null : p.id)}
                draggable
                onDragStart={(e) => { setDrag(p.id); e.dataTransfer.setData('text/plain', p.id); e.dataTransfer.effectAllowed = 'move'; }}
                onDragEnd={() => setDrag(null)}
                actions={<div className="mt-2 flex gap-2">
                  <button onClick={(e) => { e.stopPropagation(); doReforge(p); }} className="px-2 py-1 rounded bg-amber-600 text-xs">Reforge</button>
                  <button onClick={(e) => { e.stopPropagation(); equip(p); }} className="px-2 py-1 rounded bg-emerald-700 text-xs">Equip</button>
                  <button onClick={(e) => { e.stopPropagation(); scrapPart(p, 'inventory'); }} className="px-2 py-1 rounded bg-zinc-700 text-xs">Scrap</button>
                </div>}
              />
            ))}
          </div>
        </div>
      </div>

      {message && (<motion.div initial={{ opacity: 0, y: 6 }} animate={{ opacity: 1, y: 0 }} className="mt-4 p-3 rounded-xl bg-zinc-900/60 border border-zinc-700/50 text-sm">{message}</motion.div>)}

      {/* Reward modal */}
      <AnimatePresence>
        {reward && (
          <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }} className="fixed inset-0 bg-black/60 grid place-items-center z-40">
            <div className="max-w-md w-[90%] rounded-2xl p-5 bg-zinc-900 border border-zinc-700/50">
              <div className="text-lg font-bold mb-2">{reward.win ? (reward.boss ? 'Boss Defeated!' : 'Victory!') : 'Defeat'}</div>
              <div className="text-sm text-zinc-300 mb-2">{reward.win ? '+1 token, +20 scrap' : 'Tinker and try again.'}</div>
              {reward.drop && (<div className="mt-2"><div className="text-xs text-zinc-400 mb-1">Drop</div><PartCard part={reward.drop} /></div>)}
              <div className="mt-3 flex justify-end"><button onClick={() => setReward(null)} className="px-3 py-1 rounded bg-indigo-700">Close</button></div>
            </div>
          </motion.div>
        )}
      </AnimatePresence>

      <footer className="mt-8 text-xs text-zinc-500">Prototype • 3D View • No real-money mechanics</footer>
    </div>
  );
}

