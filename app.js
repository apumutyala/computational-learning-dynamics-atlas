import * as THREE from 'three';
import { OrbitControls } from './vendor/OrbitControls.js';

// AI chat and connection summaries need the backend in backend/ (a Cloudflare Worker).
// Set this to its URL, e.g. 'https://atlas-ai.<account>.workers.dev'. Empty string = AI off:
// all AI UI stays hidden and the page makes no AI network calls.
const AI_API = '';
const API = AI_API.replace(/\/+$/, '');
const AI_ENABLED = !!AI_API;
const $ = s => document.querySelector(s);
const esc = s => String(s ?? '').replace(/[&<>"']/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
const inlineMd = s => esc(s)
  .replace(/\[([^\]]+)\]\((https?:\/\/[^)\s]+)\)/g, (_, t, u) => `<a href="${u}" target="_blank" rel="noopener noreferrer">${t}</a>`)
  .replace(/\*\*([^*]+)\*\*/g, '<b>$1</b>');
const EV = {
  exact: { label: 'Exact, conditional', long: 'Exact identity under stated assumptions', color: '#7fd1a8' },
  established: { label: 'Established', long: 'Established connection', color: '#8ab8e6' },
  empirical: { label: 'Empirical', long: 'Empirical association, not a guarantee', color: '#e6b27f' },
  hypothesis: { label: 'Hypothesis', long: 'Untested hypothesis', color: '#b9a0e0' },
};
const EV_ORDER = ['exact', 'established', 'empirical', 'hypothesis'];
const reduceMotion = matchMedia('(prefers-reduced-motion: reduce)').matches;
const isMobile = () => innerWidth <= 760;

const data = await (await fetch('atlas.json?v=65559209')).json();
const nodes = data.nodes, edges = data.edges;
const byId = new Map(nodes.map((n, i) => (n.i = i, [n.id, n])));
const clusterOf = new Map(data.clusters.map(c => [c.name, c]));
const adj = new Map(nodes.map(n => [n.id, []]));
edges.forEach((e, i) => { e.i = i; adj.get(e.a).push(e); adj.get(e.b).push(e); });
const other = (e, id) => e.a === id ? e.b : e.a;

const S = { current: null, hover: null, hoverEdge: null, focusEdge: null, showHyp: false, drift: !reduceMotion, trail: [], facet: null, chat: [], chatOpen: false, mode: 'idle', cluster: null, query: '' };
const visible = e => S.showHyp || e.type !== 'hypothesis';

/* ---------------- three.js scene ---------------- */
const canvas = $('#scene');
let renderer;
try { renderer = new THREE.WebGLRenderer({ canvas, antialias: true, alpha: true, powerPreference: 'high-performance' }); }
catch { $('#nogl').style.display = 'grid'; $('#loading').classList.add('done'); throw new Error('WebGL unavailable'); }
renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
// iOS only applies :active styles (tap feedback) once a touchstart listener exists
document.addEventListener('touchstart', () => {}, { passive: true });
renderer.setClearColor(0x000000, 0);
const scene = new THREE.Scene();
const camera = new THREE.PerspectiveCamera(50, innerWidth / innerHeight, 1, 5000);
const HOME = new THREE.Vector3(80, 120, 760);
camera.position.copy(HOME);
const controls = new OrbitControls(camera, canvas);
controls.enableDamping = true; controls.dampingFactor = 0.07; controls.rotateSpeed = 0.6; controls.zoomSpeed = 0.8;
controls.minDistance = 40; controls.maxDistance = 1500;
controls.autoRotate = S.drift; controls.autoRotateSpeed = 0.22;

const pointVert = `attribute float size;attribute float alpha;attribute vec3 color;varying vec3 vColor;varying float vAlpha;uniform float uScale;uniform float uMax;
void main(){vec4 mv=modelViewMatrix*vec4(position,1.0);gl_Position=projectionMatrix*mv;float d=-mv.z;gl_PointSize=clamp(size*uScale/d,1.5,uMax*max(size,8.0)/22.0);vColor=color;vAlpha=alpha*clamp(1.35-d/1700.0,0.3,1.0);}`;
const pointFrag = `varying vec3 vColor;varying float vAlpha;uniform float uCore;
void main(){vec2 c=gl_PointCoord-0.5;float d=length(c);if(d>0.5)discard;float core=smoothstep(uCore,uCore*0.6,d);float halo=pow(smoothstep(0.5,0.0,d),2.4)*0.6;gl_FragColor=vec4(vColor*(0.7+core*0.55),(core+halo)*vAlpha);}`;
const uScale = { value: 900 }, uMax = { value: 34 * Math.min(devicePixelRatio, 2) }; // cap ≈ 34 CSS px so close-ups don't balloon
const pointMat = (core = 0.17) => new THREE.ShaderMaterial({ vertexShader: pointVert, fragmentShader: pointFrag, uniforms: { uScale, uMax, uCore: { value: core } }, transparent: true, depthWrite: false, blending: THREE.AdditiveBlending });

// dust
{
  const n = 700, pos = new Float32Array(n * 3), col = new Float32Array(n * 3), size = new Float32Array(n), alpha = new Float32Array(n);
  let s = 7; const r = () => ((s = (s * 16807) % 2147483647) / 2147483647);
  for (let i = 0; i < n; i++) {
    const rad = 380 + r() * 900, th = r() * Math.PI * 2, ph = Math.acos(2 * r() - 1);
    pos.set([rad * Math.sin(ph) * Math.cos(th), rad * Math.cos(ph) * .7, rad * Math.sin(ph) * Math.sin(th)], i * 3);
    col.set([.62, .72, .8], i * 3); size[i] = 2 + r() * 3; alpha[i] = .08 + r() * .14;
  }
  const g = new THREE.BufferGeometry();
  g.setAttribute('position', new THREE.BufferAttribute(pos, 3)); g.setAttribute('color', new THREE.BufferAttribute(col, 3));
  g.setAttribute('size', new THREE.BufferAttribute(size, 1)); g.setAttribute('alpha', new THREE.BufferAttribute(alpha, 1));
  var dust = new THREE.Points(g, pointMat()); scene.add(dust);
}

// nodes
const N = nodes.length;
const nPos = new Float32Array(N * 3), nCol = new Float32Array(N * 3), nSize = new Float32Array(N), nAlpha = new Float32Array(N);
const tSize = new Float32Array(N), tAlpha = new Float32Array(N);
nodes.forEach((n, i) => {
  nPos.set(n.p, i * 3); n.v = new THREE.Vector3(...n.p);
  const c = new THREE.Color(clusterOf.get(n.group).color); n.color = c;
  nCol.set([c.r, c.g, c.b], i * 3); nSize[i] = 0; nAlpha[i] = 0;
});
const nodeGeo = new THREE.BufferGeometry();
nodeGeo.setAttribute('position', new THREE.BufferAttribute(nPos, 3));
nodeGeo.setAttribute('color', new THREE.BufferAttribute(nCol, 3));
nodeGeo.setAttribute('size', new THREE.BufferAttribute(nSize, 1));
nodeGeo.setAttribute('alpha', new THREE.BufferAttribute(nAlpha, 1));
scene.add(new THREE.Points(nodeGeo, pointMat(0.3)));

// edges
const E = edges.length;
const ePos = new Float32Array(E * 6), eCol = new Float32Array(E * 6), eAlpha = new Float32Array(E * 2), eT = new Float32Array(E * 2), eDash = new Float32Array(E * 2);
const tEAlpha = new Float32Array(E), curEAlpha = new Float32Array(E);
edges.forEach((e, i) => {
  const a = byId.get(e.a), b = byId.get(e.b), len = a.v.distanceTo(b.v);
  ePos.set([...a.p, ...b.p], i * 6);
  const c = new THREE.Color(EV[e.type].color);
  eCol.set([c.r, c.g, c.b, c.r, c.g, c.b], i * 6);
  eT.set([0, len], i * 2); eDash.set(e.type === 'hypothesis' ? [1, 1] : [0, 0], i * 2);
});
const edgeGeo = new THREE.BufferGeometry();
edgeGeo.setAttribute('position', new THREE.BufferAttribute(ePos, 3));
edgeGeo.setAttribute('color', new THREE.BufferAttribute(eCol, 3));
edgeGeo.setAttribute('alpha', new THREE.BufferAttribute(eAlpha, 1));
edgeGeo.setAttribute('t', new THREE.BufferAttribute(eT, 1));
edgeGeo.setAttribute('dashed', new THREE.BufferAttribute(eDash, 1));
const edgeMat = new THREE.ShaderMaterial({
  vertexShader: `attribute vec3 color;attribute float alpha;attribute float t;attribute float dashed;varying vec3 vC;varying float vA;varying float vT;varying float vD;
void main(){vec4 mv=modelViewMatrix*vec4(position,1.0);gl_Position=projectionMatrix*mv;vC=color;vA=alpha*clamp(1.35-(-mv.z)/1700.0,0.3,1.0);vT=t;vD=dashed;}`,
  fragmentShader: `varying vec3 vC;varying float vA;varying float vT;varying float vD;void main(){if(vD>0.5&&fract(vT/7.0)>0.55)discard;gl_FragColor=vec4(vC,vA);}`,
  transparent: true, depthWrite: false, blending: THREE.AdditiveBlending,
});
scene.add(new THREE.LineSegments(edgeGeo, edgeMat));

// pulse along the focused connection
const PULSES = 3;
const pPos = new Float32Array(PULSES * 3), pCol = new Float32Array(PULSES * 3).fill(1), pSize = new Float32Array(PULSES).fill(9), pAlpha = new Float32Array(PULSES);
const pulseGeo = new THREE.BufferGeometry();
pulseGeo.setAttribute('position', new THREE.BufferAttribute(pPos, 3)); pulseGeo.setAttribute('color', new THREE.BufferAttribute(pCol, 3));
pulseGeo.setAttribute('size', new THREE.BufferAttribute(pSize, 1)); pulseGeo.setAttribute('alpha', new THREE.BufferAttribute(pAlpha, 1));
scene.add(new THREE.Points(pulseGeo, pointMat()));

/* ---------------- visual state ---------------- */
let searchMatch = null;
function computeTargets() {
  const cur = S.current && byId.get(S.current);
  const neigh = new Set();
  if (cur) adj.get(cur.id).filter(visible).forEach(e => neigh.add(other(e, cur.id)));
  const fe = S.focusEdge;
  nodes.forEach((n, i) => {
    let size = n.kind === 'proposal' ? 26 : 22, alpha = .9;
    if (cur) {
      if (n.id === cur.id) { size = 46; alpha = 1; }
      else if (neigh.has(n.id)) { size = Math.max(size, 25); alpha = 1; }
      else if (n.group === cur.group) { alpha = .42; }
      else alpha = .22;
    }
    if (S.cluster && !cur) alpha = n.group === S.cluster ? 1 : .2;
    if (searchMatch) alpha = searchMatch.has(n.id) ? 1 : .14;
    if (fe && (fe.a === n.id || fe.b === n.id)) { size = Math.max(size, 32); alpha = 1; }
    if (S.route && S.route.includes(n.id)) { size = Math.max(size, 28); alpha = 1; }
    if (n.id === S.hover) { size += 8; alpha = 1; }
    tSize[i] = size; tAlpha[i] = alpha;
  });
  edges.forEach((e, i) => {
    let a = 0;
    if (visible(e)) {
      a = e.type === 'hypothesis' ? .05 : .075;
      if (cur) a = (e.a === cur.id || e.b === cur.id) ? .62 : .02;
      if (S.cluster && !cur) a = (byId.get(e.a).group === S.cluster || byId.get(e.b).group === S.cluster) ? .3 : .02;
      if (searchMatch) a = .02;
    }
    if (S.routeEdges && S.routeEdges.has(e.id)) a = .9;
    if (S.hoverEdge === e.id) a = 1;
    if (fe && fe.id === e.id) a = 1;
    tEAlpha[i] = a;
  });
}

function step(dt) {
  const k = Math.min(1, dt * 9);
  for (let i = 0; i < N; i++) { nSize[i] += (tSize[i] - nSize[i]) * k; nAlpha[i] += (tAlpha[i] - nAlpha[i]) * k; }
  nodeGeo.attributes.size.needsUpdate = nodeGeo.attributes.alpha.needsUpdate = true;
  for (let i = 0; i < E; i++) { curEAlpha[i] += (tEAlpha[i] - curEAlpha[i]) * k; eAlpha[i * 2] = eAlpha[i * 2 + 1] = curEAlpha[i]; }
  edgeGeo.attributes.alpha.needsUpdate = true;
  const fe = S.focusEdge;
  if (fe) {
    const from = byId.get(S.focusFrom || fe.a).v, to = byId.get(other(fe, S.focusFrom || fe.a)).v, now = performance.now() / 1800;
    for (let j = 0; j < PULSES; j++) {
      const t = (now + j / PULSES) % 1;
      pPos.set([from.x + (to.x - from.x) * t, from.y + (to.y - from.y) * t, from.z + (to.z - from.z) * t], j * 3);
      pAlpha[j] = Math.sin(t * Math.PI) * .95;
    }
  } else pAlpha.fill(0);
  pulseGeo.attributes.position.needsUpdate = pulseGeo.attributes.alpha.needsUpdate = true;
}

/* ---------------- camera flights & view offset ---------------- */
let flight = null, viewOffset = { x: 0, y: 0 }, targetOffset = { x: 0, y: 0 };
const ease = t => t < .5 ? 4 * t * t * t : 1 - Math.pow(-2 * t + 2, 3) / 2;
function flyTo(target, dist, ms = 1300) {
  const dir = camera.position.clone().sub(controls.target).normalize();
  if (dir.lengthSq() < .1) dir.set(0, .2, 1).normalize();
  const endPos = target.clone().add(dir.multiplyScalar(dist));
  flight = { t0: performance.now(), ms: reduceMotion ? 1 : ms, p0: camera.position.clone(), q0: controls.target.clone(), p1: endPos, q1: target.clone() };
}
const flyNode = id => flyTo(byId.get(id).v, isMobile() ? 230 : 180);
function flyEdge(e) {
  const a = byId.get(e.a).v, b = byId.get(e.b).v;
  flyTo(a.clone().add(b).multiplyScalar(.5), Math.max(170, a.distanceTo(b) * 1.15 + 60));
}
function flyHome() { flyTo(new THREE.Vector3(0, 0, 0), HOME.length()); }
function updateOffset() {
  const panelOpen = !$('#panel').hidden, chatOpen = S.chatOpen, side = $('#learn-side'), sideOpen = !!side && !side.hidden;
  if (isMobile()) {
    // Bottom sheet: centre the scene in the visible strip between the top bar and the sheet's top edge.
    const welcomeEl = $('#welcome'), sheet = chatOpen ? $('#chat') : panelOpen ? $('#panel') : sideOpen ? side : !welcomeEl.classList.contains('hide') ? welcomeEl : null;
    const top = $('#topbar').getBoundingClientRect().bottom;
    const sheetTop = sheet ? Math.max(top + 120, sheet.getBoundingClientRect().top) : innerHeight;
    targetOffset = { x: 0, y: sheet ? Math.max(0, innerHeight / 2 - (top + sheetTop) / 2) : 0 };
  }
  else {
    const welcome = !$('#welcome').classList.contains('hide') && !panelOpen;
    const pw = panelOpen ? $('#panel').offsetWidth + 16 : 0, cw = chatOpen ? $('#chat').offsetWidth + 16 : sideOpen ? side.offsetWidth + 16 : welcome ? $('#welcome').offsetWidth + 24 : 0;
    targetOffset = { x: (pw - cw) / 2, y: 0 };
  }
  document.body.classList.toggle('panel-open', panelOpen);
}

/* ---------------- labels ---------------- */
const labelLayer = $('#labels');
const clusterLabels = data.clusters.map(c => {
  const el = document.createElement('div'); el.className = 'lbl cluster'; el.textContent = c.name; labelLayer.append(el);
  return { el, v: new THREE.Vector3(...c.center).add(new THREE.Vector3(0, 52, 0)), c };
});
const nodeLabelPool = new Map();
function nodeLabel(id) {
  if (!nodeLabelPool.has(id)) { const el = document.createElement('div'); el.className = 'lbl'; el.textContent = byId.get(id).name; el.style.opacity = 0; labelLayer.append(el); nodeLabelPool.set(id, { el, w: 0 }); }
  return nodeLabelPool.get(id);
}
const tmp = new THREE.Vector3();
function project(v) {
  tmp.copy(v).project(camera);
  return { x: (tmp.x + 1) / 2 * innerWidth, y: (1 - tmp.y) / 2 * innerHeight, behind: tmp.z > 1 };
}
function drawLabels() {
  const boxes = [];
  const fits = (x, y, w, h) => { for (const b of boxes) if (x < b.x + b.w && x + w > b.x && y < b.y + b.h && y + h > b.y) return false; boxes.push({ x, y, w, h }); return true; };
  const want = [];
  if (S.hover) want.push([S.hover, 'hover']);
  if (S.current) want.push([S.current, 'current']);
  if (S.focusEdge) want.push([S.focusEdge.a, 'current'], [S.focusEdge.b, 'current']);
  (S.route || []).forEach(id => want.push([id, 'neighbor']));
  if (S.current) adj.get(S.current).filter(visible).forEach(e => want.push([other(e, S.current), 'neighbor']));
  if (searchMatch) [...searchMatch].slice(0, 12).forEach(id => want.push([id, 'neighbor']));
  if (S.cluster && !S.current) nodes.filter(n => n.group === S.cluster).forEach(n => want.push([n.id, 'neighbor']));
  const shown = new Set();
  for (const [id, kind] of want) {
    if (shown.has(id)) continue;
    const L = nodeLabel(id), p = project(byId.get(id).v);
    if (!L.w) L.w = L.el.offsetWidth || byId.get(id).name.length * 6.5;
    const x = p.x + 10, y = p.y - 8;
    if (p.behind || x < -50 || x > innerWidth || y < 60 || y > innerHeight - 20 || (shown.size > 2 && !fits(x, y, L.w, 16))) continue;
    if (shown.size <= 2) boxes.push({ x, y, w: L.w, h: 16 });
    shown.add(id);
    L.el.className = 'lbl ' + kind;
    L.el.style.transform = `translate(${x}px,${y}px)`; L.el.style.opacity = 1;
  }
  nodeLabelPool.forEach((L, id) => { if (!shown.has(id)) L.el.style.opacity = 0; });
  const dim = S.current || searchMatch ? .28 : .7;
  clusterLabels.forEach(({ el, v, c }) => {
    const p = project(v), w = el.offsetWidth;
    const ok = !p.behind && p.y > 60 && p.y < innerHeight - 30;
    el.style.transform = `translate(${p.x - w / 2}px,${p.y}px)`;
    el.style.opacity = ok ? (S.cluster === c.name ? 0 : dim) : 0;
  });
}

/* ---------------- picking ---------------- */
const coarse = matchMedia('(pointer: coarse)');
function pick(x, y, touch) {
  const minR = touch || coarse.matches ? 24 : 13; // forgiving tap targets on touch screens
  let best = null, bd = Infinity;
  for (const n of nodes) {
    if (nAlpha[n.i] < .18) continue;
    const p = project(n.v); if (p.behind) continue;
    const d = Math.hypot(p.x - x, p.y - y), r = Math.max(minR, nSize[n.i] * 1.1);
    if (d < r && d < bd) { bd = d; best = n.id; }
  }
  return best;
}
let down = null, interacted = false;
const activePointers = new Set();
canvas.addEventListener('pointerdown', e => {
  activePointers.add(e.pointerId);
  // a second finger turns the gesture into pinch/pan, never a tap
  down = activePointers.size > 1 ? null : { x: e.clientX, y: e.clientY, t: performance.now() };
  stopDrift(); hideHint(); closeToolsMenu();
});
['pointercancel', 'lostpointercapture'].forEach(t => canvas.addEventListener(t, e => { activePointers.delete(e.pointerId); if (t === 'pointercancel') down = null; }));
canvas.addEventListener('pointermove', e => {
  if (e.pointerType !== 'mouse') return;
  const id = pick(e.clientX, e.clientY);
  if (id !== S.hover) { S.hover = id; canvas.style.cursor = id ? 'pointer' : 'grab'; computeTargets(); }
});
canvas.addEventListener('pointerleave', () => { S.hover = null; computeTargets(); });
canvas.addEventListener('pointerup', e => {
  activePointers.delete(e.pointerId);
  if (!down) return;
  const moved = Math.hypot(e.clientX - down.x, e.clientY - down.y), touch = e.pointerType !== 'mouse';
  const held = performance.now() - down.t;
  down = null;
  if (moved > (touch ? 12 : 6) || (touch && held > 600)) return;
  const id = pick(e.clientX, e.clientY, touch);
  if (id) travel(id, 'scene'); else if (!S.current && S.cluster) { S.cluster = null; renderDock(); computeTargets(); } else if (!S.current && !$('#welcome').classList.contains('hide')) hideWelcome();
});
canvas.addEventListener('wheel', () => { stopDrift(); hideHint(); }, { passive: true });
let driftTimer;
function stopDrift() { controls.autoRotate = false; clearTimeout(driftTimer); if (S.drift) driftTimer = setTimeout(() => { if (!S.current && S.mode === 'idle') controls.autoRotate = true; }, 9000); }
function hideHint() { if (!interacted) { interacted = true; $('#hint').classList.add('hide'); } }

/* ---------------- navigation logic ---------------- */
function edgeBetween(a, b) {
  const es = adj.get(a).filter(e => other(e, a) === b);
  return es.find(visible) || es[0] || null;
}
function shortestPath(from, to, allowHyp) {
  const prev = new Map([[from, null]]), q = [from];
  while (q.length) {
    const u = q.shift(); if (u === to) break;
    for (const e of adj.get(u)) { if (!allowHyp && e.type === 'hypothesis') continue; const v = other(e, u); if (!prev.has(v)) { prev.set(v, e); q.push(v); } }
  }
  if (!prev.has(to)) return null;
  const path = []; let cur = to;
  while (cur !== from) { const e = prev.get(cur); path.unshift(e); cur = other(e, cur); }
  return path;
}

// Entry point for every move. source: 'scene' | 'search' | 'chat' | 'list' | 'start' | 'trail'
function travel(id, source) {
  if (!byId.has(id)) return;
  S.cluster = null; clearSearch();
  if (!S.current || id === S.current || source === 'search' || source === 'start' || source === 'trail') return enter(id);
  const e = edgeBetween(S.current, id);
  if (e) return crossing(S.current, e);
  return routeCard(S.current, id);
}

function enter(id, opts = {}) {
  abortExplain();
  S.current = id; S.focusEdge = null; S.focusFrom = null; S.route = null; S.routeEdges = null; S.mode = 'node'; S.facet = null;
  if (S.trail[S.trail.length - 1] !== id) S.trail.push(id);
  if (S.trail.length > 24) S.trail.shift();
  controls.autoRotate = false;
  $('#welcome').classList.add('hide');
  renderNode(id, opts.originId ?? id);
  flyNode(id);
  renderDock(); renderChatContext(); computeTargets();
  history.replaceState(null, '', '#' + id);
}

/* ---------------- panel rendering ---------------- */
const panel = $('#panel'), sat = $('#satellite');
const X = '<svg viewBox="0 0 16 16" fill="none"><path d="m4 4 8 8M12 4l-8 8" stroke="currentColor" stroke-width="1.4" stroke-linecap="round"/></svg>';
function clusterChip(n) { const c = clusterOf.get(n.group); return `<span class="dot" style="color:${c.color}"></span>${esc(n.group)}`; }

function openPanel(html, originId) {
  const wasHidden = panel.hidden;
  panel.hidden = false; panel.innerHTML = html; panel.querySelector('.panel-body').scrollTop = 0;
  updateOffset();
  if (reduceMotion) return;
  const o = originId && byId.get(originId);
  const r = panel.getBoundingClientRect();
  if (o && wasHidden) {
    const p = project(o.v);
    const dx = p.x - (r.left + r.width / 2), dy = p.y - (r.top + r.height / 2);
    panel.animate([{ transform: `translate(${dx}px,${dy}px) scale(.12)`, opacity: 0, filter: 'blur(6px)' }, { transform: 'none', opacity: 1, filter: 'blur(0)' }], { duration: 520, easing: 'cubic-bezier(.2,.8,.2,1)' });
  } else {
    panel.querySelector('.panel-body').animate([{ opacity: 0, transform: 'translateY(6px)' }, { opacity: 1, transform: 'none' }], { duration: 300, easing: 'ease-out' });
  }
}
function closePanel() {
  abortExplain(); closeSatellite();
  panel.hidden = true; S.current = null; S.focusEdge = null; S.route = null; S.routeEdges = null; S.mode = 'idle';
  updateOffset(); computeTargets(); renderChatContext(); renderDock();
  history.replaceState(null, '', location.pathname);
  if (S.drift) stopDrift();
}

function renderNode(id, originId) {
  const n = byId.get(id);
  const es = adj.get(id), vis = es.filter(visible), hiddenHyp = es.length - vis.length;
  const groups = EV_ORDER.map(t => [t, vis.filter(e => e.type === t)]).filter(([, l]) => l.length);
  const facets = [
    ['handle', 'Design handle', 'What you can shape'],
    ['boundary', 'Where it stops', 'Scope & limits'],
    ['test', 'Test worth running', 'Atlas proposal'],
  ].filter(([k]) => n[k]);
  const conn = groups.map(([t, l]) => `<div class="conn-group"><h4 class="c-${t}"><span class="dot" style="color:${EV[t].color};width:6px;height:6px"></span>${EV[t].label} · ${l.length}</h4>${l.map(e => {
    const m = byId.get(other(e, id));
    return `<button class="conn" data-go="${m.id}" data-edge="${e.id}"><span class="dot" style="color:${clusterOf.get(m.group).color}"></span><span><b>${esc(m.name)}</b><small>${esc(e.relation)} · ${esc(m.group)}</small></span><span class="arrow">→</span></button>`;
  }).join('')}</div>`).join('');
  const html = `<div class="panel-head"><span class="eyebrow">${clusterChip(n)}${n.kind === 'proposal' ? '<span class="badge c-hypothesis">Proposal</span>' : ''}</span><button class="icon-btn" data-close aria-label="Close topic window">${X}</button></div>
  <div class="panel-body">
    <h2>${esc(n.name)}</h2>
    <p class="meta">${esc(n.authors)} · ${esc(n.year)}</p>
    <p class="summary">${esc(n.text)}</p>
    ${n.url ? `<a class="src" href="${n.url}" target="_blank" rel="noopener noreferrer">${esc(n.sourceLabel || 'Primary reference')} ↗</a>` : ''}
    ${facets.length ? `<div class="facets">${facets.map(([k, t, s]) => `<button class="facet" data-facet="${k}" aria-expanded="false"><span>${t}</span><small>${s}</small></button><div class="facet-inline" data-inline="${k}">${inlineMd(n[k])}</div>`).join('')}</div>` : ''}
    <div class="section-title">Connections <span>${vis.length} shown${hiddenHyp ? ` · ${hiddenHyp} hidden` : ''}</span></div>
    ${conn || '<p class="meta" style="margin-top:8px">No connections at this evidence setting.</p>'}
    ${hiddenHyp ? `<p class="hidden-note">${hiddenHyp} untested hypothesis connection${hiddenHyp > 1 ? 's are' : ' is'} hidden. <button class="text-btn" data-showhyp>Show hypotheses</button></p>` : ''}
    <div class="panel-actions">${AI_ENABLED ? '<button class="btn primary" data-ask>Ask about this topic</button>' : ''}${n.guide ? `<a class="btn${AI_ENABLED ? '' : ' primary'}" href="docs/${n.guide}" download>Read the guide ↓</a>` : ''}</div>
    <p class="foot-note">Proposed tests are atlas proposals, not reported results.</p>
  </div>`;
  openPanel(html, originId);
}

function crossing(fromId, e, routeInfo) {
  abortExplain(); closeSatellite();
  const toId = other(e, fromId), a = byId.get(fromId), b = byId.get(toId);
  S.mode = 'crossing'; S.focusEdge = e; S.focusFrom = fromId; S.current = fromId;
  if (!routeInfo) { S.route = null; S.routeEdges = null; }
  controls.autoRotate = false; flyEdge(e); computeTargets();
  const ca = clusterOf.get(a.group).color, cb = clusterOf.get(b.group).color;
  const stepTxt = routeInfo ? `<span class="badge" style="color:var(--muted)">Step ${routeInfo.i + 1} of ${routeInfo.path.length}</span>` : '';
  const last = !routeInfo || routeInfo.i === routeInfo.path.length - 1;
  const html = `<div class="panel-head"><span class="eyebrow">Crossing a connection ${stepTxt}</span><button class="icon-btn" data-close aria-label="Close">${X}</button></div>
  <div class="panel-body">
    <div class="cross"><div class="end"><small>From</small><b>${esc(a.name)}</b></div><div class="line" style="background:linear-gradient(90deg,${ca},${EV[e.type].color},${cb})"></div><div class="end to"><small>To</small><b>${esc(b.name)}</b></div></div>
    <div class="rel"><span class="badge c-${e.type}">${EV[e.type].label}</span><span class="relation">${esc(e.relation)}</span></div>
    ${AI_ENABLED ? `<div class="ai"><span class="eyebrow"><svg width="12" height="12" viewBox="0 0 16 16" fill="none" aria-hidden="true"><path d="M8 1.5 9.6 6.4 14.5 8 9.6 9.6 8 14.5 6.4 9.6 1.5 8 6.4 6.4Z" stroke="currentColor" stroke-width="1.2" stroke-linejoin="round"/></svg>Why this connects · AI summary</span><p id="explain"><span class="shimmer" style="width:96%"></span><span class="shimmer" style="width:88%"></span><span class="shimmer" style="width:62%"></span></p></div>` : ''}
    <p class="record"><b>Recorded connection:</b> ${esc(e.text)}</p>
    ${e.url ? `<a class="src" href="${e.url}" target="_blank" rel="noopener noreferrer">Evidence for this connection ↗</a>` : '<p class="meta">Atlas proposal with no external source; treat it as a hypothesis to test.</p>'}
    <div class="panel-actions">
      ${last ? `<button class="btn primary" data-enter="${b.id}">Enter ${esc(shortName(b.name))} →</button>` : `<button class="btn primary" data-next>Next step →</button><button class="btn" data-enter="${routeInfo.target}">Skip to ${esc(shortName(byId.get(routeInfo.target).name))}</button>`}
      <button class="btn" data-enter="${a.id}" data-back>Stay on ${esc(shortName(a.name))}</button>
    </div>
    ${AI_ENABLED ? '<p class="foot-note">The AI summary is generated from the recorded connection and both topic records. The evidence label comes from the atlas, not the model.</p>' : ''}
  </div>`;
  S.routeInfo = routeInfo || null;
  openPanel(html, toId);
  if (AI_ENABLED) explainEdge(e, fromId);
}

function routeCard(fromId, toId) {
  abortExplain(); closeSatellite();
  let path = shortestPath(fromId, toId, S.showHyp), usedHyp = S.showHyp;
  if (!path && !S.showHyp) { path = shortestPath(fromId, toId, true); usedHyp = !!path; }
  const a = byId.get(fromId), b = byId.get(toId);
  S.mode = 'route'; S.focusEdge = null;
  S.route = [fromId, toId];
  if (path) { const ids = [fromId]; path.forEach(e => ids.push(other(e, ids[ids.length - 1]))); S.route = ids; }
  S.routeEdges = new Set((path || []).map(e => e.id));
  computeTargets();
  if (path) {
    const all = S.route.map(id => byId.get(id).v), c = all.reduce((s, v) => s.add(v), new THREE.Vector3()).multiplyScalar(1 / all.length);
    const r = Math.max(...all.map(v => v.distanceTo(c)));
    flyTo(c, Math.max(220, r * 2.4));
  }
  const html = `<div class="panel-head"><span class="eyebrow">No direct connection</span><button class="icon-btn" data-close aria-label="Close">${X}</button></div>
  <div class="panel-body">
    <h2 style="font-size:21px">${esc(a.name)} → ${esc(b.name)}</h2>
    ${path ? `<p class="summary">These topics have no direct recorded link. The shortest recorded route takes ${path.length} step${path.length > 1 ? 's' : ''}${usedHyp && !S.showHyp ? ' and passes through hypothesis connections' : ''}.</p>
    <ol class="route">${S.route.map((id, i) => { const n = byId.get(id), e = path[i - 1]; return `<li><span class="dot" style="color:${clusterOf.get(n.group).color}"></span><span>${esc(n.name)}${e ? `<small><span class="c-${e.type}">${EV[e.type].label}</span> · ${esc(e.relation)}</small>` : '<small>You are here</small>'}</span></li>`; }).join('')}</ol>
    <div class="panel-actions"><button class="btn primary" data-walk>Walk the route →</button><button class="btn" data-enter="${b.id}">Jump straight there</button></div>
    <p class="foot-note">A route is a chain of recorded connections. It does not mean the endpoints are directly related.</p>`
    : `<p class="summary">No recorded route connects these topics in the atlas. That means the link has not been mapped, not that the fields are independent.</p><div class="panel-actions"><button class="btn primary" data-enter="${b.id}">Jump to ${esc(shortName(b.name))}</button><button class="btn" data-enter="${a.id}">Stay here</button></div>`}
  </div>`;
  S.pendingRoute = path ? { path, from: fromId, target: toId } : null;
  openPanel(html, toId);
}
// Learning paths: highlight a whole route (node ids + edge ids) and optionally focus/fly to one node on it.
function highlightRoute(ids, edgeIds) { S.route = ids ? ids.slice() : null; S.routeEdges = ids ? new Set(edgeIds || []) : null; computeTargets(); }
function showRoute(ids, edgeIds, focusId) {
  abortExplain(); closeSatellite(); clearSearch();
  if (!panel.hidden) { panel.hidden = true; updateOffset(); }
  S.cluster = null; S.focusEdge = null; S.focusFrom = null; S.mode = 'path'; S.current = focusId || null;
  controls.autoRotate = false; $('#welcome').classList.add('hide');
  highlightRoute(ids, edgeIds);
  if (focusId) { flyNode(focusId); if (S.trail[S.trail.length - 1] !== focusId) S.trail.push(focusId); history.replaceState(null, '', '#' + focusId); }
  else if (ids && ids.length) {
    const all = ids.map(id => byId.get(id).v), c = all.reduce((s, v) => s.add(v), new THREE.Vector3()).multiplyScalar(1 / all.length);
    flyTo(c, Math.max(240, Math.max(...all.map(v => v.distanceTo(c))) * 2.4));
  }
  renderDock(); renderChatContext(); updateOffset();
}
function clearRoute() { if (S.mode === 'path') { S.mode = 'idle'; S.current = null; } highlightRoute(null); renderDock(); }
const shortName = s => s.length > 26 ? s.split(/[,:(&—–]/)[0].trim().slice(0, 26) : s;

function clusterCard(name) {
  abortExplain(); closeSatellite();
  const c = clusterOf.get(name), ms = nodes.filter(n => n.group === name);
  S.current = null; S.cluster = name; S.mode = 'cluster'; S.focusEdge = null; S.route = null; S.routeEdges = null;
  controls.autoRotate = false;
  flyTo(new THREE.Vector3(...c.center), isMobile() ? 380 : 320);
  $('#welcome').classList.add('hide');
  const internal = edges.filter(e => visible(e) && byId.get(e.a).group === name && byId.get(e.b).group === name).length;
  const bridges = edges.filter(e => visible(e) && (byId.get(e.a).group === name) !== (byId.get(e.b).group === name));
  const outward = {}; bridges.forEach(e => { const g = byId.get(byId.get(e.a).group === name ? e.b : e.a).group; outward[g] = (outward[g] || 0) + 1; });
  const html = `<div class="panel-head"><span class="eyebrow"><span class="dot" style="color:${c.color}"></span>Cluster</span><button class="icon-btn" data-close aria-label="Close">${X}</button></div>
  <div class="panel-body"><h2>${esc(name)}</h2><p class="meta">${ms.length} topics · ${internal} internal connections · ${bridges.length} bridges to other clusters</p>
  <div class="section-title">Topics</div>${ms.map(n => `<button class="conn" data-go="${n.id}"><span class="dot" style="color:${c.color}"></span><span><b>${esc(n.name)}</b><small>${esc(n.authors)}</small></span><span class="arrow">→</span></button>`).join('')}
  <div class="section-title">Bridges out <span>by destination cluster</span></div>
  <div class="chips" style="margin-top:6px">${Object.entries(outward).sort((x, y) => y[1] - x[1]).map(([g, k]) => `<button class="chip" data-cluster="${esc(g)}"><span class="dot" style="color:${clusterOf.get(g).color}"></span>${esc(g)} · ${k}</button>`).join('')}</div></div>`;
  openPanel(html, null);
  computeTargets(); renderDock(); renderChatContext();
}

panel.addEventListener('click', ev => {
  const t = ev.target.closest('button,a'); if (!t) return;
  if (t.matches('[data-close]')) return closePanel();
  if (t.dataset.go) return travel(t.dataset.go, S.mode === 'cluster' ? 'search' : 'list');
  if (t.dataset.enter) return enter(t.dataset.enter, { originId: t.dataset.enter });
  if (t.dataset.facet) return toggleFacet(t);
  if (t.matches('[data-showhyp]')) return setHyp(true);
  if (t.matches('[data-ask]')) return openChat(true);
  if (t.dataset.cluster) return clusterCard(t.dataset.cluster);
  if (t.matches('[data-walk]') && S.pendingRoute) { const r = S.pendingRoute; return crossing(r.from, r.path[0], { path: r.path, i: 0, target: r.target, from: r.from }); }
  if (t.matches('[data-next]') && S.routeInfo) {
    const r = S.routeInfo, at = other(r.path[r.i], S.focusFrom);
    if (S.trail[S.trail.length - 1] !== at) S.trail.push(at);
    renderDock();
    return crossing(at, r.path[r.i + 1], { ...r, i: r.i + 1 });
  }
});
panel.addEventListener('pointerover', ev => {
  const t = ev.target.closest('.conn[data-edge]');
  const id = t ? t.dataset.edge : null;
  if (id !== S.hoverEdge) { S.hoverEdge = id; S.hover = t ? t.dataset.go : null; computeTargets(); }
});
panel.addEventListener('pointerleave', () => { S.hoverEdge = null; S.hover = null; computeTargets(); });

/* satellite windows that extend outward from the topic window */
function toggleFacet(btn) {
  const k = btn.dataset.facet, n = byId.get(S.current);
  const open = btn.getAttribute('aria-expanded') !== 'true';
  panel.querySelectorAll('.facet').forEach(b => b.setAttribute('aria-expanded', 'false'));
  panel.querySelectorAll('.facet-inline').forEach(d => d.classList.remove('open'));
  if (!open) return closeSatellite();
  btn.setAttribute('aria-expanded', 'true');
  if (isMobile()) { panel.querySelector(`[data-inline="${k}"]`).classList.add('open'); return; }
  const titles = { handle: 'Design handle', boundary: 'Where it stops', test: 'A test worth running' };
  const notes = { handle: 'What the cited work lets you build, set or adjust.', boundary: 'Claims this topic does not support on its own.', test: 'An original atlas proposal, not a reported experiment.' };
  sat.innerHTML = `<span class="eyebrow">${esc(n.name)}</span><h3>${titles[k]}</h3><p>${inlineMd(n[k])}</p><p class="foot-note" style="margin:0">${notes[k]}</p>`;
  sat.hidden = false;
  const pr = panel.getBoundingClientRect(), br = btn.getBoundingClientRect();
  const top = Math.max(72, Math.min(br.top - 20, innerHeight - sat.offsetHeight - 16));
  sat.style.top = top + 'px'; sat.style.left = (pr.left - sat.offsetWidth - 12) + 'px';
  if (!reduceMotion) sat.animate([{ transform: `translateX(${sat.offsetWidth * .35}px) scale(.85)`, opacity: 0, transformOrigin: 'right center' }, { transform: 'none', opacity: 1, transformOrigin: 'right center' }], { duration: 360, easing: 'cubic-bezier(.2,.8,.2,1)' });
}
function closeSatellite() { sat.hidden = true; }

/* ---------------- AI: connection explanations ---------------- */
let explainCtrl = null;
const explainCache = new Map();
function abortExplain() { if (explainCtrl) { explainCtrl.abort(); explainCtrl = null; } }
async function streamSSE(path, body, onText, signal) {
  const res = await fetch(`${API}${path}`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body), signal, credentials: 'same-origin' });
  if (!res.ok || !res.body) throw new Error('unavailable');
  const reader = res.body.getReader(), dec = new TextDecoder(); let buf = '';
  for (;;) {
    const { value, done } = await reader.read(); if (done) break;
    buf += dec.decode(value, { stream: true });
    let i;
    while ((i = buf.indexOf('\n\n')) >= 0) {
      const line = buf.slice(0, i).trim(); buf = buf.slice(i + 2);
      if (!line.startsWith('data:')) continue;
      const ev = JSON.parse(line.slice(5));
      if (ev.error) throw new Error(ev.error);
      if (ev.t) onText(ev.t);
      if (ev.done) return;
    }
  }
}
async function explainEdge(e, fromId) {
  const key = e.id + ':' + fromId, box = () => document.getElementById('explain');
  if (explainCache.has(key)) { box().textContent = explainCache.get(key); return; }
  explainCtrl = new AbortController(); const ctrl = explainCtrl;
  let text = '';
  try {
    await streamSSE('/api/explain', { edge: e.id, from: fromId }, t => {
      text += t; const b = box(); if (b && !ctrl.signal.aborted) b.innerHTML = esc(text) + '<span class="cursor"></span>';
    }, ctrl.signal);
    explainCache.set(key, text);
    const b = box(); if (b) b.textContent = text;
  } catch (err) {
    if (ctrl.signal.aborted) return;
    const b = box(); if (b) b.innerHTML = `<span style="color:var(--muted)">${esc(err.message && err.message !== 'unavailable' && !/fetch/i.test(err.message) ? err.message : 'The AI summary is unavailable right now. The recorded connection below is the source of truth.')}</span>`;
  }
}

/* ---------------- chat ---------------- */
const chatEl = $('#chat'), msgsEl = $('#messages'), input = $('#chat-input'), sendBtn = $('#send');
function openChat(focusTopic) {
  S.chatOpen = true; chatEl.hidden = false; document.body.classList.add('chat-open');
  $('#ask-btn').setAttribute('aria-expanded', 'true');
  renderChatContext(); renderMessages(); updateOffset(); closeSatellite();
  if (!reduceMotion) chatEl.animate([{ transform: isMobile() ? 'translateY(40px)' : 'translateX(-24px)', opacity: 0 }, { transform: 'none', opacity: 1 }], { duration: 320, easing: 'cubic-bezier(.2,.8,.2,1)' });
  if (focusTopic && S.current && !S.chat.length) input.value = '';
  setTimeout(() => input.focus(), 50);
}
function closeChat() { S.chatOpen = false; chatEl.hidden = true; document.body.classList.remove('chat-open'); $('#ask-btn').setAttribute('aria-expanded', 'false'); updateOffset(); }
if (!AI_ENABLED) $('#ask-btn').style.display = 'none';
$('#ask-btn').onclick = () => S.chatOpen ? closeChat() : openChat();
$('#chat-close').onclick = closeChat;
$('#chat-clear').onclick = () => { S.chat = []; renderMessages(); };
function renderChatContext() {
  const n = S.current && byId.get(S.current);
  $('#chat-ctx').innerHTML = n ? `<span class="dot" style="color:${clusterOf.get(n.group).color};width:6px;height:6px"></span>Viewing ${esc(n.name)}` : S.cluster ? `Browsing ${esc(S.cluster)}` : 'No topic selected · ask about anything in the atlas';
  const sug = n ? ['Explain this simply', 'Why does this matter for building correctable AI?', 'What should I explore next?']
    : ['What is this atlas about?', 'Where should a newcomer start?', 'What does the alignment thesis propose?'];
  $('#suggest').innerHTML = S.chat.length > 4 ? '' : sug.map(s => `<button class="chip" data-q="${esc(s)}">${esc(s)}</button>`).join('');
}
$('#suggest').addEventListener('click', e => { const b = e.target.closest('[data-q]'); if (b) send(b.dataset.q); });
function mdChat(src) {
  const blocks = src.trim().split(/\n{2,}/);
  return blocks.map(b => {
    const lines = b.split('\n');
    if (lines.every(l => /^\s*([-*]|\d+\.)\s+/.test(l))) {
      const ordered = /^\s*\d+\./.test(lines[0]);
      return `<${ordered ? 'ol' : 'ul'}>${lines.map(l => `<li>${inlineChat(l.replace(/^\s*([-*]|\d+\.)\s+/, ''))}</li>`).join('')}</${ordered ? 'ol' : 'ul'}>`;
    }
    return `<p>${lines.map(inlineChat).join('<br>')}</p>`;
  }).join('');
}
function inlineChat(s) {
  return inlineMd(s.replace(/^#+\s*/, ''))
    .replace(/`([^`]+)`/g, '<code>$1</code>')
    .replace(/\[\[([a-z0-9_-]+)\]\]/gi, (m, id) => { const n = byId.get(id); return n ? `<button class="topic-link" data-topic="${id}"><span class="dot" style="color:${clusterOf.get(n.group).color}"></span>${esc(n.name)}</button>` : esc(id); });
}
function renderMessages(streamingText) {
  if (!S.chat.length && streamingText == null) {
    msgsEl.innerHTML = `<div class="chat-empty"><p>Ask about the topic you're viewing, how two ideas connect, or where to go next. Answers link to topics you can travel to.</p></div>`;
    return;
  }
  msgsEl.innerHTML = S.chat.map(m => m.role === 'user' ? `<div class="msg user">${esc(m.content)}</div>` : m.error ? `<div class="msg err">${esc(m.content)}</div>` : `<div class="msg bot">${mdChat(m.content)}</div>`).join('')
    + (streamingText != null ? `<div class="msg bot">${streamingText ? mdChat(streamingText) : '<span class="shimmer" style="width:70%"></span><span class="shimmer" style="width:50%"></span>'}</div>` : '');
  msgsEl.scrollTop = msgsEl.scrollHeight;
}
msgsEl.addEventListener('click', e => { const b = e.target.closest('[data-topic]'); if (b) { travel(b.dataset.topic, 'chat'); if (isMobile()) closeChat(); } });
input.addEventListener('input', () => { sendBtn.disabled = !input.value.trim(); input.style.height = 'auto'; input.style.height = Math.min(120, input.scrollHeight) + 'px'; });
input.addEventListener('keydown', e => { if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); $('#chat-form').requestSubmit(); } });
$('#chat-form').addEventListener('submit', e => { e.preventDefault(); send(input.value); });
let chatBusy = false;
async function send(q) {
  q = q.trim(); if (!q || chatBusy) return;
  chatBusy = true; input.value = ''; sendBtn.disabled = true; input.style.height = 'auto';
  S.chat.push({ role: 'user', content: q }); renderChatContext(); renderMessages('');
  let text = '';
  try {
    await streamSSE('/api/chat', { messages: S.chat.filter(m => !m.error).slice(-12), current: S.current, path: S.trail.slice(-10) }, t => { text += t; renderMessages(text); });
    S.chat.push({ role: 'assistant', content: text || '(No answer returned.)' });
  } catch (err) {
    S.chat.push({ role: 'assistant', error: true, content: err.message && err.message !== 'unavailable' && !/fetch/i.test(err.message) ? err.message : 'The chat is unavailable right now. Topic windows and recorded connections still work.' });
  }
  chatBusy = false; renderMessages(); renderChatContext();
}

/* ---------------- search ---------------- */
const q = $('#q'), results = $('#results');
let resultIds = [], activeResult = 0;
function clearSearch() { q.value = ''; results.hidden = true; q.setAttribute('aria-expanded', 'false'); if (searchMatch) { searchMatch = null; computeTargets(); } }
q.addEventListener('input', () => {
  const s = q.value.trim().toLowerCase();
  if (!s) return clearSearch();
  const scored = nodes.map(n => {
    const name = n.name.toLowerCase(), hay = [n.name, n.authors, n.group, n.text].join(' ').toLowerCase();
    const score = name.startsWith(s) ? 3 : name.includes(s) ? 2 : hay.includes(s) ? 1 : 0;
    return [score, n];
  }).filter(([sc]) => sc).sort((a, b) => b[0] - a[0]);
  resultIds = scored.slice(0, 9).map(([, n]) => n.id); activeResult = 0;
  searchMatch = new Set(scored.map(([, n]) => n.id)); computeTargets();
  results.hidden = false; q.setAttribute('aria-expanded', 'true');
  results.innerHTML = resultIds.length ? resultIds.map((id, i) => { const n = byId.get(id); return `<button role="option" data-id="${id}" class="${i === 0 ? 'active' : ''}"><span class="dot" style="color:${clusterOf.get(n.group).color}"></span><span>${esc(n.name)}<small>${esc(n.authors)}</small></span></button>`; }).join('')
    : `<p class="meta" style="padding:8px 10px;margin:0">No topic or researcher matches “${esc(q.value)}”. Try a broader term such as “filter” or “control”.</p>`;
});
q.addEventListener('keydown', e => {
  if (e.key === 'ArrowDown' || e.key === 'ArrowUp') { e.preventDefault(); activeResult = (activeResult + (e.key === 'ArrowDown' ? 1 : -1) + resultIds.length) % Math.max(1, resultIds.length); results.querySelectorAll('button').forEach((b, i) => b.classList.toggle('active', i === activeResult)); }
  if (e.key === 'Enter' && resultIds[activeResult]) { travel(resultIds[activeResult], 'search'); q.blur(); }
  if (e.key === 'Escape') { clearSearch(); q.blur(); }
});
results.addEventListener('click', e => { const b = e.target.closest('[data-id]'); if (b) travel(b.dataset.id, 'search'); });
document.addEventListener('pointerdown', e => { if (!e.target.closest('.search')) { results.hidden = true; } });

/* ---------------- dock, toggles, keys ---------------- */
function renderDock() {
  const tr = $('#trail');
  tr.hidden = S.trail.length < 1;
  tr.innerHTML = `<span class="label">Path</span>` + S.trail.slice(-7).map((id, i, arr) => `${i ? '<span class="sep">›</span>' : ''}<button data-trail="${id}" class="${id === S.current ? 'here' : ''}">${esc(shortName(byId.get(id).name))}</button>`).join('');
  $('#clusters').innerHTML = data.clusters.map(c => `<button data-cluster="${esc(c.name)}" class="${S.cluster === c.name ? 'on' : ''}" title="${c.count} topics"><span class="dot" style="color:${c.color}"></span>${esc(c.name)}</button>`).join('');
}
$('#trail').addEventListener('click', e => { const b = e.target.closest('[data-trail]'); if (b) travel(b.dataset.trail, 'trail'); });
$('#clusters').addEventListener('click', e => { const b = e.target.closest('[data-cluster]'); if (b) clusterCard(b.dataset.cluster); });
function setHyp(on) {
  S.showHyp = on; $('#t-hyp').setAttribute('aria-pressed', String(on));
  computeTargets();
  if (S.mode === 'node' && S.current) renderNode(S.current, null);
  toast(on ? 'Showing untested hypothesis connections (dashed).' : 'Hypothesis connections hidden.');
}
$('#t-hyp').onclick = () => setHyp(!S.showHyp);
$('#t-drift').onclick = () => { S.drift = !S.drift; $('#t-drift').setAttribute('aria-pressed', String(S.drift)); controls.autoRotate = S.drift && !S.current; };
function hideWelcome() { $('#welcome').classList.add('hide'); updateOffset(); rememberWelcome(); }
const wc = $('#welcome-close'); if (wc) wc.onclick = hideWelcome;
// Overview resets the view; it reopens the welcome window only if it is closed, and closes it if it is open.
$('#t-overview').onclick = e => { e.currentTarget.blur(); closeToolsMenu(); const w = $('#welcome'), wasOpen = !w.classList.contains('hide') && panel.hidden && !S.cluster; if (!panel.hidden) closePanel(); S.cluster = null; renderDock(); computeTargets(); flyHome(); w.classList.toggle('hide', wasOpen); updateOffset(); rememberWelcome(); };

/* ---------------- phone helpers: welcome card, overflow menu, sheet swipe ---------------- */
const WELCOME_KEY = 'atlas.welcomeDismissed';
function welcomeDismissed() { try { return sessionStorage.getItem(WELCOME_KEY) === '1'; } catch { return false; } }
function rememberWelcome() { const closed = $('#welcome').classList.contains('hide'); try { closed ? sessionStorage.setItem(WELCOME_KEY, '1') : sessionStorage.removeItem(WELCOME_KEY); } catch { /* storage unavailable */ } }
document.querySelectorAll('[data-welcome-dismiss]').forEach(b => b.addEventListener('click', hideWelcome));
// The logo doubles as an info button: reopen (or close) the welcome card.
const brand = $('.brand');
brand.setAttribute('role', 'button'); brand.tabIndex = 0; brand.title = 'About this atlas';
brand.addEventListener('click', () => { const w = $('#welcome'); if (w.classList.contains('hide')) { if (S.chatOpen) closeChat(); if (!panel.hidden) closePanel(); w.classList.remove('hide'); updateOffset(); rememberWelcome(); } else hideWelcome(); });
brand.addEventListener('keydown', e => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); brand.click(); } });
if (isMobile() && welcomeDismissed()) $('#welcome').classList.add('hide');

const moreBtn = $('#t-more');
function closeToolsMenu() { document.body.classList.remove('tools-open'); moreBtn?.setAttribute('aria-expanded', 'false'); }
moreBtn?.addEventListener('click', () => { const open = !document.body.classList.contains('tools-open'); document.body.classList.toggle('tools-open', open); moreBtn.setAttribute('aria-expanded', String(open)); });
document.addEventListener('pointerdown', e => { if (!e.target.closest('#t-more,.tools,#track-menu')) closeToolsMenu(); });
$('#t-hyp').addEventListener('click', () => setTimeout(closeToolsMenu, 150));
$('#t-drift').addEventListener('click', () => setTimeout(closeToolsMenu, 150));

// Swipe the sheet's header down to close it (phones).
let sheetDrag = null;
panel.addEventListener('pointerdown', e => { if (isMobile() && e.pointerType !== 'mouse' && e.target.closest('.panel-head') && !e.target.closest('button')) sheetDrag = { y: e.clientY }; });
panel.addEventListener('pointermove', e => { if (sheetDrag) { const dy = Math.max(0, e.clientY - sheetDrag.y); panel.style.transform = dy ? `translateY(${dy}px)` : ''; } });
const endSheetDrag = e => {
  if (!sheetDrag) return;
  const dy = e.clientY - sheetDrag.y; sheetDrag = null; panel.style.transform = '';
  if (e.type === 'pointerup' && dy > 70) closePanel();
};
panel.addEventListener('pointerup', endSheetDrag); panel.addEventListener('pointercancel', endSheetDrag);
$('#t-drift').setAttribute('aria-pressed', String(S.drift));
let toastT;
function toast(msg) { const t = $('#toast'); t.textContent = msg; t.hidden = false; t.style.opacity = 1; clearTimeout(toastT); toastT = setTimeout(() => { t.style.opacity = 0; setTimeout(() => t.hidden = true, 300); }, 2200); }
document.addEventListener('keydown', e => {
  const typing = e.target.matches('input,textarea');
  if (e.key === '/' && !typing) { e.preventDefault(); q.focus(); }
  if (e.key === 'Escape' && !typing) { if (!sat.hidden) closeSatellite(); else if (S.mode === 'crossing' || S.mode === 'route') enter(S.focusFrom || S.current); else if (!panel.hidden) closePanel(); else if (S.chatOpen) closeChat(); }
});
// Welcome "start a track" buttons: each opens a track's entry path (learn.js handles [data-track-path] clicks).
const TRACK_STARTS = [
  ['intro-ai-safety-to-thesis', 'Intro to AI safety', 'cs120', 'New to AI safety: rewards, preferences, robustness, oversight'],
  ['bridge-dynamics-to-deep-learning', 'Dynamics → deep learning', 'kalman', 'From state-space models and Kalman filters to neural networks'],
  ['bridge-interp-to-devinterp', 'Interpretability → SLT → J-lens', 'circuits', 'Mechanistic interpretability, singular learning theory, developmental interpretability, the LLC and the Jacobian lens'],
  ['fde-harness-to-thesis', 'Forward-deployed engineer', 'mcp', 'Shipping LLM agents into high-stakes products: harnesses, evals, monitoring, escalation'],
  ['thesis-supporting-case', 'Alignment thesis', 'alignmentthesis', "The atlas's alignment thesis: the case for it, then the case against"],
  ['inference-to-thesis', 'Inference engineer', 'roofline', 'LLM serving and performance engineering, then what it means for alignment'],
].filter(t => byId.has(t[2]));
$('#starts').innerHTML = TRACK_STARTS.map(([path, label, node, sub]) => `<button class="chip" data-track-path="${path}" data-track-node="${node}" title="${esc(sub)}" aria-label="${esc(label + ': ' + sub)}"><span class="dot" style="color:${clusterOf.get(byId.get(node).group).color}"></span>${esc(label)}</button>`).join('');
// learn.js opens the path; if the learning layer failed to load, fall back to the track's first topic.
$('#starts').addEventListener('click', e => { const b = e.target.closest('[data-track-path]'); if (!b) return; hideWelcome(); setTimeout(() => { if (!document.querySelector('#learn-side:not([hidden])')) travel(b.dataset.trackNode, 'start'); }, 1500); });

/* ---------------- resize & loop ---------------- */
function resize() {
  renderer.setSize(innerWidth, innerHeight, false);
  camera.aspect = innerWidth / innerHeight;
  uScale.value = innerHeight / (2 * Math.tan(THREE.MathUtils.degToRad(camera.fov / 2)));
  updateOffset(); applyOffset(true);
}
function applyOffset(force) {
  const k = force ? 1 : .12;
  viewOffset.x += (targetOffset.x - viewOffset.x) * k; viewOffset.y += (targetOffset.y - viewOffset.y) * k;
  if (Math.abs(viewOffset.x) < .5 && Math.abs(viewOffset.y) < .5 && !targetOffset.x && !targetOffset.y) camera.clearViewOffset();
  else camera.setViewOffset(innerWidth, innerHeight, viewOffset.x, viewOffset.y, innerWidth, innerHeight);
  camera.updateProjectionMatrix();
}
addEventListener('resize', resize);
resize(); computeTargets(); renderDock(); renderChatContext(); renderMessages();
let last = performance.now();
function loop(now) {
  const dt = Math.min(.05, (now - last) / 1000); last = now;
  if (flight) {
    const t = Math.min(1, (now - flight.t0) / flight.ms), e = ease(t);
    camera.position.lerpVectors(flight.p0, flight.p1, e); controls.target.lerpVectors(flight.q0, flight.q1, e);
    if (t >= 1) flight = null;
  }
  controls.update();
  if (!reduceMotion) dust.rotation.y += dt * .004;
  applyOffset(); step(dt); renderer.render(scene, camera); drawLabels();
  requestAnimationFrame(loop);
}
requestAnimationFrame(loop);
$('#loading').classList.add('done');
const hashId = decodeURIComponent(location.hash.slice(1));
if (byId.has(hashId)) setTimeout(() => travel(hashId, 'start'), 300);
addEventListener('hashchange', () => { const id = decodeURIComponent(location.hash.slice(1)); if (byId.has(id) && id !== S.current) travel(id, 'start'); });
window.__atlas = { S, travel, byId, project, enter };
// Learning harness (learn.js) gets only this surface; the atlas keeps working if it fails to load.
const learnAPI = { S, byId, edges, nodes, EV, enter, travel, showRoute, highlightRoute, clearRoute, closePanel, updateOffset, isMobile, esc, clusterColor: id => clusterOf.get(byId.get(id).group).color, edgeById: id => edges.find(e => e.id === id) || null, edgeBetween: (a, b) => adj.get(a)?.find(e => other(e, a) === b) || null };
import('./learn.js?v=20261012b').then(m => m.init(learnAPI)).catch(err => console.warn('Learning paths unavailable:', err));
