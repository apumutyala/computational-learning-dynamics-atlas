// Personalized learning paths: background questionnaire → adaptive probe → recommended path,
// then a step-by-step path view on the 3D graph. Static and client-side only; no AI calls.
const $ = s => document.querySelector(s);
const LS_KEY = 'atlas.learn.v1';
const SKILLS = { 'linear-algebra': 'Linear algebra', probability: 'Probability', optimization: 'Optimization', 'calculus-odes': 'Calculus & ODEs', 'stats-ml': 'Statistical ML', 'deep-learning': 'Deep learning', 'signals-estimation': 'Signals & estimation', control: 'Control', 'dynamical-systems': 'Dynamical systems', 'generative-models': 'Generative models', rl: 'Reinforcement learning', theory: 'Learning theory' };
const PRIOR = .25, MIN_Q = 8, MAX_Q = 12, SLOPE = 2.2;
const B = [0, -1.4, 0, 1.4];               // logit ability at which an item of difficulty d is a coin flip (before guessing)
const V0 = 8, VSELF = 1.5;                 // prior variance of logit ability: untested vs self-reported
// Skill families: the probe opens with one item per family. KIN: close pairs, where an answer is weak evidence (weight SPILL).
const FAMILY = { probability: 0, 'stats-ml': 0, 'linear-algebra': 0, optimization: 0, 'signals-estimation': 1, control: 1, rl: 1, 'deep-learning': 2, 'generative-models': 2, theory: 3, 'dynamical-systems': 3, 'calculus-odes': 3 };
const KIN = ['probability|stats-ml', 'probability|theory', 'signals-estimation|control', 'deep-learning|generative-models', 'dynamical-systems|calculus-odes', 'linear-algebra|optimization'], SPILL = .5;
const kin = (a, b) => KIN.includes(a + '|' + b) || KIN.includes(b + '|' + a);
const LEVEL_TARGET = { intro: .3, intermediate: .55, advanced: .72 };
const LEVEL_NEED = [0, .6, .72, .82, .9];  // skill mastery that counts as having "demonstrated" a node of level L
const LEVEL_NAME = ['', 'Intro', 'Intermediate', 'Advanced', 'Frontier'];
const X = '<svg viewBox="0 0 16 16" fill="none"><path d="m4 4 8 8M12 4l-8 8" stroke="currentColor" stroke-width="1.4" stroke-linecap="round"/></svg>';
const clamp = (x, a = 0, b = 1) => Math.max(a, Math.min(b, x));
const sig = x => 1 / (1 + Math.exp(-x));
const pct = x => Math.round(x * 100);

let A, esc, D = null, dataP = null, st = {};
// Corrupt or hand-edited storage must not crash the atlas: keep only well-formed parts.
const load = () => { try { const v = JSON.parse(localStorage.getItem(LS_KEY)); if (!v || typeof v !== 'object' || Array.isArray(v)) return {};
  if (!v.model || typeof v.model.m !== 'object' || typeof v.model.n !== 'object' || typeof v.model.seen !== 'object' || !v.model.m || !v.model.n || !v.model.seen) { delete v.model; delete v.probe; delete v.rec; }
  if (!Array.isArray(v.probe?.answers)) delete v.probe;
  v.progress = Object.fromEntries(Object.entries(v.progress && typeof v.progress === 'object' ? v.progress : {}).filter(([, x]) => x && Array.isArray(x.done)));
  return v; } catch { return {}; } };
const save = () => { try { localStorage.setItem(LS_KEY, JSON.stringify(st)); } catch { /* storage unavailable: progress lives for this visit only */ } };

/* ---------------- data ---------------- */
// Real files in learning/ win; learning/fixtures/ is used only for a file that is missing.
async function getJSON(name) {
  try { const r = await fetch(name); if (r.ok) return { json: await r.json(), fixture: name.includes('/fixtures/') }; } catch { /* missing or malformed */ }
  return { json: null, fixture: false };
}
const BASE = ['nodes.json', 'paths.json', 'background.json'];
// learning/index.json lists the question banks, so new banks load without a code change.
const DEFAULT_QFILES = ['questions-1.json', 'questions-2.json'];
function loadData() {
  return dataP ||= getJSON('learning/index.json').then(ix => {
    const qf = Array.isArray(ix.json?.questions) && ix.json.questions.length ? ix.json.questions : DEFAULT_QFILES;
    const FILES = [...BASE, ...qf];
    return Promise.all(FILES.map(f => getJSON('learning/' + f))).then(async real => {
      // Fixtures stand in for a missing real file. Question files are a set: once any real bank exists,
      // a missing bank is just empty (no sample items mixed into real content).
      const anyRealQ = real.slice(3).some(r => r.json);
      return Promise.all(real.map((r, i) => r.json || (i >= 3 && anyRealQ) ? r : getJSON('learning/fixtures/' + (i >= 3 ? DEFAULT_QFILES[(i - 3) % 2] : FILES[i]))));
    });
  }).then(([n, p, b, ...qs]) => {
    const seen = new Set(), items = qs.flatMap(q => q.json?.items || []).filter(it =>
      it && !seen.has(it.id) && seen.add(it.id) && A.byId.has(it.node) && SKILLS[it.skill] && Array.isArray(it.choices) && it.answer >= 0 && it.answer < it.choices.length);
    const meta = n.json?.nodes || {};
    const paths = (p.json?.paths || []).map(x => ({ ...x, steps: (x.steps || []).filter(s => A.byId.has(s.node)) })).filter(x => x.steps.length >= 2);
    D = { meta, paths, bg: b.json?.questions || [], items, pathById: new Map(paths.map(x => [x.id, x])),
      fixtures: [n, p, b, ...qs].some(r => r.fixture) };
    return D;
  });
}
function nodeMeta(id) {
  const m = D.meta[id];
  if (m) return { level: m.level || 2, skills: (m.skills || []).filter(s => SKILLS[s]), prereqs: m.prereqs || [] };
  return { level: 2, skills: [...new Set(D.items.filter(i => i.node === id).map(i => i.skill))], prereqs: [] };
}

/* ---------------- learner model ---------------- */
// IRT with uncertainty: logit ability θ_s ~ N(μ, v) per skill, mastery m[s] = σ(μ) ∈ [0,1], variance v[s].
// P(correct) = g + (1-g)·σ(SLOPE·(θ − B[d])), g = 1/#choices (0 for "I don't know"). Each answer is an exact grid
// posterior moment-matched back to a Gaussian; kin skills (KIN) get the same evidence down-weighted by SPILL.
const newModel = priors => ({ m: { ...priors }, n: {}, seen: {}, self: { ...priors }, v: Object.fromEntries(Object.keys(priors).map(s => [s, VSELF])) });
const mOf = (M, s) => M.m[s] ?? PRIOR;
const lg = m => { m = clamp(+m || PRIOR, .02, .98); return Math.log(m / (1 - m)); };
const vOf = (M, s) => { const v = +M.v?.[s]; return v > 0 ? v : s in M.m ? VSELF / (1 + (M.n[s] || 0)) : V0; };
const GRID = Array.from({ length: 121 }, (_, i) => i / 10 - 6);
function posterior(mu, v, it, correct, g, w) {
  const b = B[it.difficulty] ?? 0; let W = 0, S = 0, S2 = 0;
  for (const t of GRID) { const p = g + (1 - g) * sig(SLOPE * (t - b)), x = Math.exp(-((t - mu) ** 2) / (2 * v)) * (correct ? p : 1 - p) ** w; W += x; S += x * t; S2 += x * t * t; }
  const m = S / W; return [m, Math.max(.05, S2 / W - m * m)];
}
function observe(M, it, correct, idk) {
  const g = idk ? 0 : 1 / it.choices.length; if (!M.v || typeof M.v !== 'object') M.v = {};
  Object.keys(SKILLS).forEach(s => { const own = s === it.skill; if (!own && !kin(s, it.skill)) return;
    const [mu, v] = posterior(lg(mOf(M, s)), vOf(M, s), it, correct, own ? g : 1 / it.choices.length, own ? 1 : SPILL);
    M.m[s] = clamp(sig(mu), .02, .98); M.v[s] = v; });
  M.n[it.skill] = (M.n[it.skill] || 0) + 1;
  const ev = M.seen[it.node] ||= { c: 0, t: 0 };
  ev.t++; if (correct) ev.c++;
}
const conf = (M, s) => { const n = (M.n[s] || 0) + (M.self?.[s] != null ? .6 : 0); return n / (n + 1.5); }; // weight of direct evidence (self-report = 0.6 answer)
// Expected share of a skill's mastery variance (in σ(θ) space) that one answer at difficulty d removes; exact on the grid.
function gainOf(M, s, d, g) {
  const mu = lg(mOf(M, s)), v = vOf(M, s), b = B[d] ?? 0, a = [0, 0, 0, 0, 0, 0];
  for (const t of GRID) { const w = Math.exp(-((t - mu) ** 2) / (2 * v)), p = g + (1 - g) * sig(SLOPE * (t - b)), m = sig(t); a[0] += w; a[1] += w * m; a[2] += w * m * m; a[3] += w * p; a[4] += w * p * m; a[5] += w * p * m * m; }
  const vr = (W, S, S2) => W > 1e-12 ? S2 / W - (S / W) ** 2 : 0, V = vr(a[0], a[1], a[2]), P = a[3] / a[0];
  return V > 1e-9 ? (V - P * vr(a[3], a[4], a[5]) - (1 - P) * vr(a[0] - a[3], a[1] - a[4], a[2] - a[5])) / V : 0;
}
function goalVotes(bgAns) {
  const v = {};
  D.bg.forEach(q => (bgAns[q.id] || []).forEach(oid => { const o = q.options.find(o => o.id === oid); if (o?.goal) v[o.goal] = (v[o.goal] || 0) + 1; }));
  return v;
}
function priorsFrom(bgAns) {
  const pr = {};
  D.bg.forEach(q => (bgAns[q.id] || []).forEach(oid => { const o = q.options.find(o => o.id === oid); Object.entries(o?.sets || {}).forEach(([s, x]) => { if (SKILLS[s]) pr[s] = Math.max(pr[s] ?? 0, clamp(+x)); }); }));
  return pr;
}
// Core skills of a path: share of its steps that exercise each skill, plus a bonus for the path's declared skills.
const coreW = new WeakMap();
function core(p) {
  if (!coreW.has(p)) { const w = {}; p.steps.forEach(st => nodeMeta(st.node).skills.forEach(s => w[s] = (w[s] || 0) + 1 / p.steps.length)); (p.skills || []).forEach(s => { if (SKILLS[s]) w[s] = (w[s] || 0) + .5; });
    const t = Object.values(w).reduce((a, x) => a + x, 0) || 1; coreW.set(p, Object.entries(w).map(([s, x]) => [s, x / t])); }
  return coreW.get(p);
}
const overall = M => { let a = .5 * PRIOR, b = .5; Object.keys(SKILLS).forEach(s => { const c = conf(M, s); a += c * mOf(M, s); b += c; }); return a / b; };
// score = goal match + level fit + strength in the core skills + room to grow. lvl = core-weighted mastery where each skill
// counts by its evidence: untested skills are uncertain (low weight, leaning to the learner's overall level), not weak.
// str = evidence-weighted signed strength (+ measured strong, − measured weak, 0 untested). opt > 0: optimistic view.
function scorePath(p, M, votes, opt = 0) {
  const maxV = Math.max(1, ...Object.values(votes)), goal = (votes[p.id] || 0) / maxV, G = overall(M);
  let a = 0, b = 0, str = 0, rel = 0;
  core(p).forEach(([s, x]) => { const m = mOf(M, s), c = conf(M, s), u = .15 * (1 - c);
    a += x * (c * m + u * ((m + G) / 2 + opt)); b += x * (c + u); str += x * c * (2 * m - 1); rel += x * 4 * m * (1 - m); });
  const lvl = a / b, fit = clamp(1 - Math.abs(lvl - (LEVEL_TARGET[p.level] ?? .55)) * 2.2);
  return { p, score: 3 * goal + 1.2 * fit + 1.5 * str + .3 * rel, goal, fit, rel, str, lvl };
}
const rankPaths = (M, votes) => D.paths.map(p => scorePath(p, M, votes)).sort((a, b) => b.score - a.score);
// A step is skipped only with evidence: 2+ correct on its node, or every skill it uses answered 2+ times and measured at its level.
function demonstrated(M, id) {
  const ev = M.seen[id], nm = nodeMeta(id);
  if (ev?.t && ev.c / ev.t < .67) return false;
  if (ev?.c >= 2) return true;
  return nm.skills.length > 0 && nm.skills.every(s => (M.n[s] || 0) >= 2 && mOf(M, s) >= LEVEL_NEED[nm.level]);
}
// start = first step whose node the learner has not demonstrated (steps are skipped only as a leading run)
function startStep(p, M) { let i = 0; while (i < p.steps.length - 1 && demonstrated(M, p.steps[i].node)) i++; return i; }

// Item selection. Explore: the first items cover distinct skill families (background-weighted). Exploit: maximize
// (core-skill relevance to the top candidate paths) × (expected share of the skill's variance the answer removes).
// Candidates = the current top 3 plus the top 3 if every untested core skill turned out strong (so specialists get asked about their field).
function skillRelevance(M, votes) {
  const r = {}, w = [1, .7, .5], top = [...rankPaths(M, votes).slice(0, 3), ...D.paths.map(p => scorePath(p, M, votes, .35)).sort((a, b) => b.score - a.score).slice(0, 3)];
  top.forEach(({ p }, k) => core(p).forEach(([s, x]) => r[s] = (r[s] || 0) + w[k % 3] * x));
  const mx = Math.max(.001, ...Object.values(r)); Object.keys(r).forEach(s => r[s] /= mx);
  return { r, onPath: new Set(top.flatMap(({ p }) => p.steps.map(s => s.node))) };
}
const NFAM = new Set(Object.values(FAMILY)).size;
let IMP = null; // how central a skill is across all paths (opening items favour central skills: theory over calculus)
const imp = s => (IMP ||= D.paths.reduce((r, p) => (core(p).forEach(([k, x]) => r[k] = (r[k] || 0) + x / D.paths.length * 4), r), {}))[s] || 0;
function nextItem(ob) {
  const M = ob.model, asked = new Set(ob.asked.map(a => a.id)), sk = ob.asked.map(a => D.items.find(i => i.id === a.id)?.skill);
  const fams = new Set(sk.map(s => FAMILY[s])), explore = ob.asked.length < NFAM && D.items.some(it => !fams.has(FAMILY[it.skill]));
  const { r, onPath } = skillRelevance(M, ob.votes), pri = ob.priors || {}, fc = {}, ft = {}; // share correct per family: dig into strong families
  ob.asked.forEach((a, i) => { const f = FAMILY[sk[i]]; ft[f] = (ft[f] || 0) + 1; fc[f] = (fc[f] || 0) + (a.correct ? 1 : 0); });
  const famStr = f => ft[f] ? fc[f] / ft[f] : 0;
  let best = null, bs = -1, gains = new Map();
  for (const it of D.items) {
    if (asked.has(it.id) || (explore && fams.has(FAMILY[it.skill]))) continue;
    const key = it.skill + it.difficulty + it.choices.length, n = M.n[it.skill] || 0;
    const gain = gains.get(key) ?? gains.set(key, gainOf(explore && !n ? { m: { [it.skill]: Math.max(mOf(M, it.skill), .5) }, v: { [it.skill]: vOf(M, it.skill) }, n: {} } : M, it.skill, it.difficulty, 1 / it.choices.length)).get(key);
    // (opening items are pitched at mid-scale: a first answer should be able to show strength, not just weakness)
    let sc = explore ? gain * (1 + 2 * (pri[it.skill] || 0)) * (.3 + imp(it.skill))
      : ((r[it.skill] || 0) + .12) * gain * (.35 + 1 / (1 + n)) * (onPath.has(it.node) ? 1.3 : 1) * (n > 1 ? 1 : 1 + 3 * famStr(FAMILY[it.skill]) * (n ? .5 + imp(it.skill) : 1))
        * (1 + clamp(2 * mOf(M, it.skill) - 1) * ((it.difficulty || 2) - 2) * .7); // a skill that looks strong: probe its ceiling
    if (sk.length && sk[sk.length - 1] === it.skill) sc *= .55;
    if (sc > bs) { bs = sc; best = it; }
  }
  return best;
}
function confident(ob) {
  if (ob.asked.length >= MAX_Q) return true;
  if (ob.asked.length < MIN_Q) return false;
  // Confident = the top recommendation has held for the last 3 answers and each of its skills is pinned down
  // (2+ answers, or 1 answer that left mastery far from the 0.5 boundary, or nothing left to ask about it).
  const M = ob.model, h = ob.tops.slice(-3), top = D.pathById.get(h[h.length - 1]);
  if (h.length < 3 || h.some(x => x !== h[0]) || !top) return false;
  return (top.skills || []).every(s => (M.n[s] || 0) >= 2 || ((M.n[s] || 0) >= 1 && Math.abs(mOf(M, s) - .5) >= .25) || !D.items.some(i => i.skill === s && !ob.asked.some(a => a.id === i.id)));
}

/* ---------------- onboarding modal ---------------- */
const modal = () => $('#learn-modal'), dlg = () => $('#learn-dialog');
let ob = null, opener = null;
async function openOnboarding() {
  if (!dlg().contains(document.activeElement)) opener = document.activeElement; // retake from results keeps the original opener
  await loadData();
  ob = { step: 1, bg: { ...(st.bg || {}) }, asked: [], tops: [], cur: null, fb: null };
  modal().hidden = false; document.body.classList.add('learn-modal-open');
  renderModal();
}
function closeModal() {
  modal().hidden = true; document.body.classList.remove('learn-modal-open'); ob = null;
  if (opener && document.contains(opener)) opener.focus();
}
const stepper = n => `<ol class="l-stepper" aria-label="Progress">${['Background', 'Quick check', 'Your path'].map((t, i) => `<li class="${i + 1 === n ? 'on' : i + 1 < n ? 'done' : ''}"${i + 1 === n ? ' aria-current="step"' : ''}><span>${i + 1}</span>${t}</li>`).join('')}</ol>`;
const head = (n, title) => `<div class="l-head">${stepper(n)}<button class="icon-btn" data-l="close" aria-label="Close personalization">${X}</button></div><h2 id="learn-title" tabindex="-1">${title}</h2>`;
function renderModal(focusSel) {
  const d = dlg();
  if (ob.step === 1) {
    d.innerHTML = head(1, 'Tell us where you are starting') + `<p class="l-sub">Self-report sets your starting estimate. A short quick check then refines it. Nothing leaves this browser.</p>
    ${D.bg.map(q => `<fieldset class="l-q"><legend>${esc(q.prompt)}</legend><div class="l-opts">${q.options.map(o => {
      const on = (ob.bg[q.id] || []).includes(o.id);
      return `<label class="l-opt${on ? ' on' : ''}"><input type="${q.type === 'multi' ? 'checkbox' : 'radio'}" name="${esc(q.id)}" value="${esc(o.id)}"${on ? ' checked' : ''}><span>${esc(o.label)}</span></label>`;
    }).join('')}</div></fieldset>`).join('')}
    <div class="l-actions"><button class="btn primary" data-l="probe">Continue to quick check →</button><span class="l-note">${D.items.length ? `About ${MIN_Q}–${MAX_Q} questions` : 'No questions available'}</span></div>`;
  } else if (ob.step === 2) {
    const it = ob.cur, k = ob.asked.length + (ob.fb ? 0 : 1);
    const fb = ob.fb;
    d.innerHTML = head(2, `Question ${k}`) + `<div class="l-probe-meta"><span class="l-chip">${esc(SKILLS[it.skill])}</span><span class="l-diff" title="Difficulty ${it.difficulty} of 3" aria-label="Difficulty ${it.difficulty} of 3">${[1, 2, 3].map(i => `<i class="${i <= it.difficulty ? 'on' : ''}"></i>`).join('')}</span><span class="l-note">Topic: ${esc(A.byId.get(it.node).name)}</span></div>
    <div class="l-bar" role="progressbar" aria-label="Quick check progress" aria-valuemin="0" aria-valuemax="${MAX_Q}" aria-valuenow="${ob.asked.length}"><i style="width:${pct(ob.asked.length / MAX_Q)}%"></i></div>
    ${quizItem(it, fb)}
    <div class="l-actions">${fb ? `<button class="btn primary" data-l="next">${confident(ob) || !nextItem(ob) ? 'See my results →' : 'Next question →'}</button>` : `<button class="btn" data-l="idk">I don't know</button>`}
    ${ob.asked.length >= MIN_Q && !fb ? '<button class="text-btn" data-l="finish">Finish early</button>' : ''}</div>`;
  } else renderResults(d);
  const f = focusSel && d.querySelector(focusSel);
  (f || d.querySelector('#learn-title')).focus();
}
function quizItem(it, fb) {
  return `<p class="l-prompt" id="l-prompt">${esc(it.prompt)}</p><div class="l-choices" role="group" aria-labelledby="l-prompt">${it.choices.map((c, i) => {
    const cls = fb ? (i === it.answer ? ' right' : i === fb.choice ? ' wrong' : ' dim') : '';
    return `<button class="l-choice${cls}" data-choice="${i}"${fb ? ' disabled' : ''}><kbd>${i + 1}</kbd><span>${esc(c)}</span>${fb && i === it.answer ? '<b class="sr"> (correct answer)</b>' : ''}${fb && i === fb.choice && i !== it.answer ? '<b class="sr"> (your answer)</b>' : ''}</button>`;
  }).join('')}</div>
  <div class="l-feedback" aria-live="polite">${fb ? `<div class="l-fb ${fb.correct ? 'ok' : fb.idk ? 'idk' : 'no'}"><b>${fb.correct ? 'Correct.' : fb.idk ? 'No problem — here is the idea.' : 'Not quite.'}</b> ${esc(it.explain)}${it.source ? ` <a href="${esc(it.source)}" target="_blank" rel="noopener noreferrer">Source ↗</a>` : ''}</div>` : ''}</div>`;
}
function startProbe() {
  const bg = {};
  D.bg.forEach(q => { const v = [...dlg().querySelectorAll(`input[name="${CSS.escape(q.id)}"]:checked`)].map(i => i.value); if (v.length) bg[q.id] = v; });
  ob.bg = bg; ob.votes = goalVotes(bg); ob.priors = priorsFrom(bg); ob.model = newModel(ob.priors);
  ob.step = 2; ob.cur = nextItem(ob); ob.fb = null;
  if (!ob.cur) return finishProbe();
  renderModal();
}
function answer(choice, idk) {
  if (!ob?.cur || ob.fb) return;
  const it = ob.cur, correct = !idk && choice === it.answer;
  observe(ob.model, it, correct, idk);
  ob.asked.push({ id: it.id, correct, idk: !!idk }); ob.tops.push(rankPaths(ob.model, ob.votes)[0]?.p.id);
  ob.fb = { choice: idk ? -1 : choice, correct, idk: !!idk };
  renderModal('[data-l="next"]');
}
function nextQuestion() {
  if (confident(ob)) return finishProbe();
  ob.cur = nextItem(ob); ob.fb = null;
  if (!ob.cur) return finishProbe();
  renderModal();
}
function finishProbe() {
  const ranked = rankPaths(ob.model, ob.votes);
  st.bg = ob.bg; st.priors = ob.priors; st.model = ob.model; st.votes = ob.votes;
  st.probe = { answers: ob.asked, at: Date.now() };
  st.rec = ranked.length ? { path: ranked[0].p.id, alts: ranked.slice(1, 3).map(r => r.p.id), why: { goal: ranked[0].goal, fit: ranked[0].fit, rel: ranked[0].rel } } : null;
  st.progress ||= {};
  ranked.forEach(({ p }) => { const pr = st.progress[p.id] ||= { done: [], at: 0 }; if (!pr.done.length) pr.at = startStep(p, st.model); });
  save(); ob.step = 3; renderModal(); renderWelcome(); if (view) renderSide();
}
function masteryBars(M, priors) {
  const skills = Object.keys(SKILLS).filter(s => (M.n[s] || 0) > 0 || s in (priors || {})).sort((a, b) => (M.n[b] || 0) - (M.n[a] || 0) || mOf(M, b) - mOf(M, a));
  if (!skills.length) return '<p class="l-note">No skills measured yet.</p>';
  return `<ul class="l-mastery">${skills.map(s => { const m = mOf(M, s), n = M.n[s] || 0; return `<li><span class="l-sk">${esc(SKILLS[s])}</span><span class="l-meter" role="meter" aria-label="${esc(SKILLS[s])} mastery" aria-valuemin="0" aria-valuemax="100" aria-valuenow="${pct(m)}"><i style="width:${pct(m)}%"></i></span><span class="l-val">${pct(m)}%</span><small>${n ? `${n} question${n > 1 ? 's' : ''}${n < 2 ? ' · low confidence' : ''}` : 'self-reported'}</small></li>`; }).join('')}</ul>`;
}
function renderResults(d) {
  const M = st.model, rec = st.rec && D.pathById.get(st.rec.path);
  const right = st.probe.answers.filter(a => a.correct).length;
  d.innerHTML = head(3, 'Your starting point') + `<p class="l-sub">${right} of ${st.probe.answers.length} correct. Bars show estimated mastery; questions you answered count for more than self-report.</p>
  ${masteryBars(M, st.priors)}
  ${rec ? `<div class="l-section">Recommended path</div>${pathCard(rec, true)}
  <div class="l-section">Also a good fit</div>${st.rec.alts.map(id => D.pathById.get(id)).filter(Boolean).map(p => pathCard(p, false)).join('')}`
    : '<p class="l-note">No learning paths are available yet.</p>'}
  <div class="l-actions"><button class="btn" data-l="retake">Retake assessment</button><button class="btn" data-l="browse">Browse all paths</button></div>`;
}
function whyText(p) {
  const s = scorePath(p, st.model, st.votes || {}), bits = [];
  if (s.goal > 0) bits.push('matches your goal');
  const above = s.lvl > (LEVEL_TARGET[p.level] ?? .55);
  bits.push(s.fit > .7 ? 'pitched at your level' : above ? (s.fit > .4 ? 'you will move quickly' : 'mostly review for you') : (s.fit > .4 ? 'a reasonable stretch' : 'a big stretch from your level'));
  if (s.rel > .7) bits.push('exercises skills you are ready to grow');
  return bits.join(' · ');
}
function pathCard(p, primary) {
  const pr = st.progress?.[p.id], at = pr ? pr.at : 0, s = p.steps[at];
  return `<div class="l-card${primary ? ' rec' : ''}"><div class="l-card-top"><b>${esc(p.title)}</b><span class="l-tag">${esc(p.level)} · ${p.steps.length} steps${p.estHours ? ` · ~${p.estHours} h` : ''}</span></div>
  <p>${esc(p.summary)}</p>${st.model ? `<small class="l-why">${esc(whyText(p))}</small>` : ''}
  ${st.model && at > 0 && !pr?.done.length ? `<small class="l-start">Start at step ${at + 1}: <b>${esc(A.byId.get(s.node).name)}</b> — you already showed the earlier ${at > 1 ? `${at} steps` : 'step'}.</small>` : ''}
  <div class="l-actions"><button class="btn${primary ? ' primary' : ''}" data-path="${esc(p.id)}">${pr?.done.length ? 'Continue path' : primary ? `Start ${at > 0 ? `at step ${at + 1}` : 'path'} →` : 'Open path'}</button></div></div>`;
}

/* ---------------- side panel: path browser + path view + checkpoints ---------------- */
const side = () => $('#learn-side');
let view = null; // { kind: 'browse' } | { kind: 'path', id, active, quiz }
const routeOf = p => ({ ids: p.steps.map(s => s.node), edges: p.steps.map((s, i) => s.via || (i ? A.edgeBetween(p.steps[i - 1].node, s.node)?.id : null)).filter(Boolean) });
let welcomeBack = false; // re-show the welcome window if browsing paths was the only thing that hid it
function openSide(v) {
  if (side().hidden) welcomeBack = !$('#welcome').classList.contains('hide');
  if (v.kind === 'path') welcomeBack = false;
  view = v; side().hidden = false; document.body.classList.add('learn-open');
  $('#welcome').classList.add('hide'); $('#paths-btn')?.setAttribute('aria-expanded', 'true');
  renderSide(); A.updateOffset();
}
function closeSide() {
  if (side().hidden) return;
  if (side().contains(document.activeElement)) $('#paths-btn')?.focus(); // don't strand keyboard focus on <body>
  side().hidden = true; document.body.classList.remove('learn-open'); $('#paths-btn')?.setAttribute('aria-expanded', 'false');
  if (view?.kind === 'path') A.clearRoute();
  if (welcomeBack && $('#panel').hidden && !A.S.current) $('#welcome').classList.remove('hide');
  welcomeBack = false; view = null; if ($('#learn-pill')) $('#learn-pill').hidden = true; A.updateOffset();
}
const sideHead = (eyebrow) => `<div class="panel-head"><span class="eyebrow">${eyebrow}</span><button class="icon-btn" data-l="side-close" aria-label="Close learning paths">${X}</button></div>`;
function renderSide(focusSel) {
  const el = side();
  if (view.kind === 'browse') {
    const recId = st.rec?.path;
    el.innerHTML = sideHead('Learning paths') + `<div class="panel-body"><h2 tabindex="-1" class="l-h">Curated paths</h2>
    <p class="meta">Each path is a chain of recorded connections. ${st.model ? 'Ordered by fit to your assessment.' : 'Personalize to get a recommendation and a starting step.'}</p>
    ${st.model ? '' : '<div class="l-actions" style="margin:0 0 6px"><button class="btn primary" data-l="personalize">Personalize my path</button></div>'}
    ${(st.model ? rankPaths(st.model, st.votes || {}).map(r => r.p) : D.paths).map(p => {
      const pr = st.progress?.[p.id], done = pr?.done.length || 0;
      return `<button class="l-pathrow" data-path="${esc(p.id)}"><span class="l-pr-top"><b>${esc(p.title)}</b>${p.id === recId ? '<span class="badge c-exact">For you</span>' : ''}</span>
      <small>${esc(p.level)} · ${p.steps.length} steps${p.estHours ? ` · ~${p.estHours} h` : ''} · ${(p.skills || []).map(s => esc(SKILLS[s] || s)).join(', ')}</small>
      ${done ? `<span class="l-bar sm"><i style="width:${pct(done / p.steps.length)}%"></i></span>` : ''}</button>`;
    }).join('') || '<p class="meta">No learning paths are available yet.</p>'}
    ${st.model ? '<div class="l-actions"><button class="btn" data-l="results">My results</button><button class="btn" data-l="retake">Retake assessment</button><button class="btn" data-l="reset">Reset progress</button></div>' : ''}
    ${D.fixtures ? '<p class="foot-note">Some learning data is still sample content.</p>' : ''}</div>`;
  } else {
    const p = D.pathById.get(view.id), pr = st.progress[view.id] ||= { done: [], at: 0 };
    const done = new Set(pr.done), a = view.active;
    el.innerHTML = sideHead(`<button class="l-back" data-l="browse" aria-label="All paths">‹</button>Learning path · ${esc(p.level)}`) + `<div class="panel-body">
    <h2 tabindex="-1" class="l-h">${esc(p.title)}</h2><p class="meta">${esc(p.audience || p.summary)}</p>
    <div class="l-prog"><span class="l-bar"><i style="width:${pct(done.size / p.steps.length)}%"></i></span><small>${done.size} of ${p.steps.length} done</small></div>
    <ol class="l-steps">${p.steps.map((s, i) => {
      const n = A.byId.get(s.node), e = s.via && A.edgeById(s.via), on = i === a, nm = nodeMeta(s.node);
      const ev = e ? `<span class="c-${e.type}">${A.EV[e.type].label}</span> · ${esc(e.relation)}` : i ? 'No direct recorded link' : 'Starting point';
      return `<li class="${on ? 'on' : ''}${done.has(s.node) ? ' done' : ''}">
      <button class="l-step" data-step="${i}" aria-expanded="${on}"><span class="l-num" style="--c:${A.clusterColor(s.node)}">${done.has(s.node) ? '✓' : i + 1}</span><span><b>${esc(n.name)}</b><small>${ev}</small></span></button>
      ${on ? `<div class="l-detail">${view.quiz ? renderQuiz() : `<p><span class="l-k">Objective</span>${esc(s.objective)}</p><p><span class="l-k">Why here</span>${esc(s.why)}</p>
        ${e ? `<p class="record"><b>${esc(A.EV[e.type].long)}:</b> ${esc(e.text)}</p>` : ''}
        <p class="l-note">${LEVEL_NAME[nm.level] || ''}${nm.prereqs.length ? ` · builds on ${nm.prereqs.filter(id => A.byId.has(id)).map(id => esc(A.byId.get(id).name)).join(', ')}` : ''}</p>
        <div class="l-actions"><button class="btn" data-l="topic">Open topic window</button><button class="btn" data-l="quiz">Checkpoint · ${checkpointItems(s.node).length} q</button>
        <button class="btn${done.has(s.node) ? '' : ' primary'}" data-l="done" aria-pressed="${done.has(s.node)}">${done.has(s.node) ? 'Done ✓' : 'Mark done'}</button>
        ${i < p.steps.length - 1 ? '<button class="btn" data-l="nextstep">Next step →</button>' : ''}</div>`}</div>` : ''}</li>`;
    }).join('')}</ol>
    <div class="l-actions"><button class="btn" data-l="${st.model ? 'retake' : 'personalize'}">${st.model ? 'Retake assessment' : 'Personalize'}</button><button class="btn" data-l="reset">Reset progress</button></div>
    <p class="foot-note">A path is a chain of recorded connections; evidence labels come from the atlas.</p></div>`;
  }
  const f = focusSel && el.querySelector(focusSel);
  if (f) f.focus();
}
function openPath(id, step) {
  const p = D.pathById.get(id); if (!p) return;
  st.progress ||= {}; const pr = st.progress[id] ||= { done: [], at: 0 };
  st.active = id; save();
  goStep(p, step ?? pr.at ?? 0, true);
}
function goStep(p, i, focusHead) {
  i = clamp(i, 0, p.steps.length - 1);
  st.progress[p.id].at = i; save();
  const r = routeOf(p);
  A.showRoute(r.ids, r.edges, p.steps[i].node);
  openSide({ kind: 'path', id: p.id, active: i, quiz: null });
  renderSide(focusHead ? '.l-h' : `[data-step="${i}"]`);
}
function checkpointItems(node) {
  const own = D.items.filter(it => it.node === node);
  if (own.length) return own.sort((a, b) => a.difficulty - b.difficulty).slice(0, 4);
  const sk = new Set(nodeMeta(node).skills);
  return D.items.filter(it => sk.has(it.skill)).sort((a, b) => a.difficulty - b.difficulty).slice(0, 3);
}
function renderQuiz() {
  const q = view.quiz, items = q.items;
  if (!items.length) return `<p class="l-note">No checkpoint questions for this topic yet.</p><div class="l-actions"><button class="btn" data-l="quiz-exit">Back to step</button></div>`;
  if (q.i >= items.length) {
    const c = q.res.filter(Boolean).length, pass = c / items.length >= .67;
    return `<div class="l-fb ${pass ? 'ok' : 'no'}"><b>${c} of ${items.length} correct.</b> ${pass ? 'Checkpoint passed — step marked done.' : 'Review the topic window and try again when ready.'}</div>
    <div class="l-actions">${pass ? '<button class="btn primary" data-l="nextstep">Next step →</button>' : '<button class="btn primary" data-l="quiz">Try again</button>'}<button class="btn" data-l="quiz-exit">Back to step</button></div>`;
  }
  const it = items[q.i];
  return `<div class="l-probe-meta"><span class="l-chip">Checkpoint ${q.i + 1} / ${items.length}</span>${q.related ? '<span class="l-note">Related practice</span>' : ''}</div>${quizItem(it, q.fb)}
  <div class="l-actions">${q.fb ? `<button class="btn primary" data-l="quiz-next">${q.i + 1 < items.length ? 'Next question →' : 'See result'}</button>` : '<button class="btn" data-l="quiz-idk">I don\'t know</button>'}<button class="text-btn" data-l="quiz-exit">Exit checkpoint</button></div>`;
}
function quizAnswer(choice, idk) {
  const q = view.quiz; if (!q || q.fb || q.i >= q.items.length) return;
  const it = q.items[q.i], correct = !idk && choice === it.answer;
  q.fb = { choice: idk ? -1 : choice, correct, idk: !!idk }; q.res.push(correct);
  st.model ||= newModel({}); observe(st.model, it, correct, idk); save();
  renderSide('[data-l="quiz-next"]');
}
function quizNext() {
  const q = view.quiz; q.i++; q.fb = null;
  if (q.i >= q.items.length) {
    const p = D.pathById.get(view.id), node = p.steps[view.active].node, c = q.res.filter(Boolean).length;
    st.quiz ||= {}; st.quiz[node] = { c, t: q.items.length, at: Date.now() };
    if (c / q.items.length >= .67) { const pr = st.progress[p.id]; if (!pr.done.includes(node)) pr.done.push(node); }
    save();
  }
  renderSide('.l-detail button');
}
function resetProgress() {
  if (!confirm('Reset your learning progress and assessment on this device?')) return;
  st = {}; save(); try { localStorage.removeItem(LS_KEY); } catch { /* ignore */ }
  renderWelcome();
  if (view?.kind === 'path') { A.clearRoute(); }
  openSide({ kind: 'browse' }); A.updateOffset();
}

/* ---------------- events ---------------- */
function onClick(ev) {
  const t = ev.target.closest('button,[data-path]'); if (!t) return;
  const act = t.dataset.l;
  if (t.dataset.path) { if (ob) closeModal(); return openPath(t.dataset.path); }
  if (t.dataset.choice != null) return ob && !side().contains(t) ? answer(+t.dataset.choice) : quizAnswer(+t.dataset.choice);
  if (t.dataset.step != null) {
    const p = D.pathById.get(view.id), i = +t.dataset.step;
    if (i === view.active && !view.quiz) { view.active = -1; return renderSide(`[data-step="${i}"]`); }
    return goStep(p, i);
  }
  switch (act) {
    case 'close': return closeModal();
    case 'probe': return startProbe();
    case 'idk': return answer(-1, true);
    case 'next': return nextQuestion();
    case 'finish': return finishProbe();
    case 'retake': return openOnboarding(); // the side stays open behind the modal, so Esc returns to it
    case 'browse': if (ob) closeModal(); if (view?.kind === 'path') A.clearRoute(); return openSide({ kind: 'browse' }), renderSide('.l-h');
    case 'personalize': return openOnboarding();
    case 'results': return showResults();
    case 'reset': return resetProgress();
    case 'side-close': return closeSide();
    case 'topic': { const p = D.pathById.get(view.id), r = routeOf(p); A.enter(p.steps[view.active].node); A.highlightRoute(r.ids, r.edges); return syncPill(); }
    case 'done': { const p = D.pathById.get(view.id), node = p.steps[view.active].node, pr = st.progress[p.id];
      pr.done = pr.done.includes(node) ? pr.done.filter(x => x !== node) : [...pr.done, node]; save(); return renderSide('[data-l="done"]'); }
    case 'nextstep': { const p = D.pathById.get(view.id); return goStep(p, view.active + 1); }
    case 'quiz': { const p = D.pathById.get(view.id), node = p.steps[view.active].node; view.quiz = { items: checkpointItems(node), related: !D.items.some(it => it.node === node), i: 0, res: [], fb: null }; return renderSide('.l-detail .l-choice, .l-detail button'); }
    case 'quiz-idk': return quizAnswer(-1, true);
    case 'quiz-next': return quizNext();
    case 'quiz-exit': view.quiz = null; return renderSide('[data-l="quiz"]');
  }
}
async function showResults() {
  await loadData();
  if (!st.model || !st.probe) return openOnboarding();
  if (!dlg().contains(document.activeElement)) opener = document.activeElement; // retake from results keeps the original opener
  ob = { step: 3 }; modal().hidden = false; document.body.classList.add('learn-modal-open'); renderModal();
}
function onKey(e) {
  if (!modal().hidden) {
    if (e.key === 'Escape') { e.preventDefault(); e.stopImmediatePropagation(); return closeModal(); }
    if (e.key === 'Tab') { // keep focus inside the dialog
      const f = [...dlg().querySelectorAll('button:not([disabled]),input,a[href],[tabindex="-1"]')].filter(x => x.offsetParent !== null && x.tabIndex >= 0);
      if (!f.length) return;
      const first = f[0], last = f[f.length - 1];
      if (!dlg().contains(document.activeElement)) { e.preventDefault(); first.focus(); }
      else if (e.shiftKey && document.activeElement === first) { e.preventDefault(); last.focus(); }
      else if (!e.shiftKey && document.activeElement === last) { e.preventDefault(); first.focus(); }
      e.stopImmediatePropagation(); return;
    }
    if (ob?.step === 2 && !ob.fb && /^[1-4]$/.test(e.key) && +e.key <= ob.cur.choices.length && !e.target.matches('input,textarea')) { e.stopImmediatePropagation(); return answer(+e.key - 1); }
    e.stopImmediatePropagation(); return; // the atlas behind the dialog ignores keys
  }
  if (view?.quiz && !view.quiz.fb && /^[1-4]$/.test(e.key) && side().contains(e.target)) { const it = view.quiz.items[view.quiz.i]; if (it && +e.key <= it.choices.length) { e.stopImmediatePropagation(); return quizAnswer(+e.key - 1); } }
  if (e.key === 'Escape' && !side().hidden && !e.target.matches('input,textarea') && $('#panel').hidden && $('#satellite').hidden && !A.S.chatOpen) {
    e.stopImmediatePropagation();
    if (view?.quiz) { view.quiz = null; return renderSide('[data-l="quiz"]'); }
    closeSide();
  }
}
function syncPill() { const pl = $('#learn-pill'); if (pl) pl.hidden = !(view?.kind === 'path' && !$('#panel').hidden); }
function renderWelcome() {
  const c = $('#learn-continue'); if (!c) return;
  const id = st.active || st.rec?.path, p = id && D?.pathById.get(id);
  c.hidden = !p; if (p) c.textContent = `Continue: ${p.title}`;
  c.dataset.path = p ? p.id : '';
}

export async function init(api) {
  A = api; esc = api.esc; st = load();
  // A stale cached index.html may lack the harness markup: do nothing rather than crash the atlas.
  if (!modal() || !dlg() || !side()) return console.warn('Learning paths: page markup missing; skipped.');
  const on = (sel, ev, fn) => $(sel)?.addEventListener(ev, fn);  modal().addEventListener('click', ev => { if (ev.detail > 1) return; /* 2nd click of a double-click lands on the re-rendered dialog (or the backdrop) */ if (ev.target.matches('.l-backdrop')) return closeModal(); onClick(ev); });
  dlg().addEventListener('change', ev => { const l = ev.target.closest('.l-opt'); if (!l) return; l.closest('.l-opts').querySelectorAll('.l-opt').forEach(o => o.classList.toggle('on', o.querySelector('input').checked)); });
  side().addEventListener('click', ev => ev.detail > 1 || onClick(ev));
  document.addEventListener('keydown', onKey, true);
  on('#personalize-btn', 'click', () => openOnboarding());
  on('#paths-btn', 'click', async () => { if (!side().hidden && view?.kind === 'browse') return closeSide(); await loadData(); if (view?.kind === 'path') A.clearRoute(); openSide({ kind: 'browse' }); renderSide('.l-h'); });
  on('#learn-continue', 'click', async e => { await loadData(); openPath(e.currentTarget.dataset.path); });
  on('#learn-pill', 'click', () => { A.closePanel(); syncPill(); side().querySelector('.l-step[aria-expanded="true"]')?.focus(); });
  on('#t-overview', 'click', () => closeSide());
  // Keep the path lit when the topic window closes or re-renders (closePanel/crossings clear S.route).
  new MutationObserver(() => {
    syncPill();
    if (view?.kind === 'path' && !A.S.route) { const r = routeOf(D.pathById.get(view.id)); A.highlightRoute(r.ids, r.edges); }
  }).observe($('#panel'), { attributes: true, attributeFilter: ['hidden'], childList: true });
  if (st.active || st.rec) { await loadData(); renderWelcome(); }
}
