// Personalized learning paths: background questionnaire → adaptive probe → recommended path,
// then a step-by-step path view on the 3D graph. Static and client-side only; no AI calls.
const $ = s => document.querySelector(s);
const LS_KEY = 'atlas.learn.v1';
const SKILLS = { 'linear-algebra': 'Linear algebra', probability: 'Probability', optimization: 'Optimization', 'calculus-odes': 'Calculus & ODEs', 'stats-ml': 'Statistical ML', 'deep-learning': 'Deep learning', 'signals-estimation': 'Signals & estimation', control: 'Control', 'dynamical-systems': 'Dynamical systems', 'generative-models': 'Generative models', rl: 'Reinforcement learning', theory: 'Learning theory' };
const PRIOR = .25, MIN_Q = 8, MAX_Q = 12, SLOPE = 7;
const DIFF_AT = [0, .3, .5, .7];            // mastery at which an item of difficulty d is a coin flip (before guessing)
const LEVEL_TARGET = { intro: .3, intermediate: .5, advanced: .7 };
const LEVEL_NEED = [0, .55, .7, .82, .9];  // skill mastery that counts as having "demonstrated" a node of level L
const LEVEL_NAME = ['', 'Intro', 'Intermediate', 'Advanced', 'Frontier'];
const X = '<svg viewBox="0 0 16 16" fill="none"><path d="m4 4 8 8M12 4l-8 8" stroke="currentColor" stroke-width="1.4" stroke-linecap="round"/></svg>';
const clamp = (x, a = 0, b = 1) => Math.max(a, Math.min(b, x));
const mean = a => a.length ? a.reduce((s, x) => s + x, 0) / a.length : 0;
const sig = x => 1 / (1 + Math.exp(-x));
const pct = x => Math.round(x * 100);

let A, esc, D = null, dataP = null, st = {};
const load = () => { try { return JSON.parse(localStorage.getItem(LS_KEY)) || {}; } catch { return {}; } };
const save = () => { try { localStorage.setItem(LS_KEY, JSON.stringify(st)); } catch { /* storage unavailable: progress lives for this visit only */ } };

/* ---------------- data ---------------- */
// Real files in learning/ win; learning/fixtures/ is used only for a file that is missing.
async function getJSON(name) {
  for (const base of ['learning/', 'learning/fixtures/']) {
    try { const r = await fetch(base + name); if (r.ok) return { json: await r.json(), fixture: base !== 'learning/' }; } catch { /* try next */ }
  }
  return { json: null, fixture: true };
}
function loadData() {
  return dataP ||= Promise.all(['nodes.json', 'paths.json', 'background.json', 'questions-1.json', 'questions-2.json'].map(getJSON)).then(([n, p, b, q1, q2]) => {
    const seen = new Set(), items = [...(q1.json?.items || []), ...(q2.json?.items || [])].filter(it =>
      it && !seen.has(it.id) && seen.add(it.id) && A.byId.has(it.node) && SKILLS[it.skill] && Array.isArray(it.choices) && it.answer >= 0 && it.answer < it.choices.length);
    const meta = n.json?.nodes || {};
    const paths = (p.json?.paths || []).map(x => ({ ...x, steps: (x.steps || []).filter(s => A.byId.has(s.node)) })).filter(x => x.steps.length >= 2);
    D = { meta, paths, bg: b.json?.questions || [], items, pathById: new Map(paths.map(x => [x.id, x])),
      fixtures: [n, p, b, q1, q2].some(r => r.fixture) };
    return D;
  });
}
function nodeMeta(id) {
  const m = D.meta[id];
  if (m) return { level: m.level || 2, skills: (m.skills || []).filter(s => SKILLS[s]), prereqs: m.prereqs || [] };
  return { level: 2, skills: [...new Set(D.items.filter(i => i.node === id).map(i => i.skill))], prereqs: [] };
}

/* ---------------- learner model ---------------- */
// mastery m[s] ∈ [0,1]; P(correct) = g + (1-g)·σ(SLOPE·(m − DIFF_AT[d])), g = 1/#choices.
const newModel = priors => ({ m: { ...priors }, n: {}, seen: {} });
const mOf = (M, s) => M.m[s] ?? PRIOR;
function pCorrect(M, it) { const g = 1 / it.choices.length; return g + (1 - g) * sig(SLOPE * (mOf(M, it.skill) - DIFF_AT[it.difficulty || 2])); }
function observe(M, it, correct) {
  const s = it.skill, n = M.n[s] || 0, lr = .34 / (1 + .45 * n);
  M.m[s] = clamp(mOf(M, s) + lr * ((correct ? 1 : 0) - pCorrect(M, it)), .02, .98);
  M.n[s] = n + 1;
  const ev = M.seen[it.node] ||= { c: 0, t: 0 };
  ev.t++; if (correct) ev.c++;
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
// score = goal match + level fit + skill relevance (room to grow in the path's skills)
function scorePath(p, M, votes) {
  const maxV = Math.max(1, ...Object.values(votes));
  const goal = (votes[p.id] || 0) / maxV;
  const ms = (p.skills?.length ? p.skills : [...new Set(p.steps.flatMap(s => nodeMeta(s.node).skills))]).map(s => mOf(M, s));
  const fit = clamp(1 - Math.abs(mean(ms) - (LEVEL_TARGET[p.level] ?? .5)) * 2.2);
  const rel = mean(ms.map(m => 4 * m * (1 - m)));
  return { p, score: 2 * goal + 1.2 * fit + .8 * rel, goal, fit, rel };
}
const rankPaths = (M, votes) => D.paths.map(p => scorePath(p, M, votes)).sort((a, b) => b.score - a.score);
function demonstrated(M, id) {
  const ev = M.seen[id];
  if (ev?.t) { if (ev.c / ev.t >= .67) return true; if (!ev.c) return false; }
  const nm = nodeMeta(id);
  return nm.skills.length > 0 && nm.skills.every(s => (M.n[s] || 0) > 0 && mOf(M, s) >= LEVEL_NEED[nm.level]);
}
// start = first step whose node the learner has not demonstrated (steps are skipped only as a leading run)
function startStep(p, M) { let i = 0; while (i < p.steps.length - 1 && demonstrated(M, p.steps[i].node)) i++; return i; }

// Item selection: maximize (skill relevance to candidate paths) × (Fisher-style information at current mastery) × (skill uncertainty)
function skillRelevance(M, votes) {
  const r = {}, top = rankPaths(M, votes).slice(0, 3), w = [1, .6, .4];
  top.forEach(({ p }, k) => {
    (p.skills || []).forEach(s => r[s] = (r[s] || 0) + w[k]);
    p.steps.forEach(st => nodeMeta(st.node).skills.forEach(s => r[s] = (r[s] || 0) + w[k] * .5 / p.steps.length * 2));
  });
  const mx = Math.max(.001, ...Object.values(r)); Object.keys(r).forEach(s => r[s] /= mx);
  return { r, onPath: new Set(top.flatMap(({ p }) => p.steps.map(s => s.node))) };
}
function nextItem(ob) {
  const M = ob.model, { r, onPath } = skillRelevance(M, ob.votes), asked = new Set(ob.asked.map(a => a.id)), last = ob.asked[ob.asked.length - 1];
  let best = null, bs = -1;
  for (const it of D.items) {
    if (asked.has(it.id)) continue;
    const s = sig(SLOPE * (mOf(M, it.skill) - DIFF_AT[it.difficulty || 2])), info = s * (1 - s) + .02;
    const unc = 1 / (1 + (M.n[it.skill] || 0));
    let sc = ((r[it.skill] || 0) + .12) * info * (.35 + unc) * (onPath.has(it.node) ? 1.3 : 1);
    if (last && D.items.find(i => i.id === last.id)?.skill === it.skill) sc *= .55;
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
  opener = document.activeElement;
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
    ${ob.asked.length >= 4 && !fb ? '<button class="text-btn" data-l="finish">Finish early</button>' : ''}</div>`;
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
  observe(ob.model, it, correct);
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
  save(); ob.step = 3; renderModal(); renderWelcome();
}
function masteryBars(M, priors) {
  const skills = Object.keys(SKILLS).filter(s => (M.n[s] || 0) > 0 || s in (priors || {})).sort((a, b) => (M.n[b] || 0) - (M.n[a] || 0) || mOf(M, b) - mOf(M, a));
  if (!skills.length) return '<p class="l-note">No skills measured yet.</p>';
  return `<ul class="l-mastery">${skills.map(s => { const m = mOf(M, s), n = M.n[s] || 0; return `<li><span class="l-sk">${esc(SKILLS[s])}</span><span class="l-meter" role="meter" aria-label="${esc(SKILLS[s])} mastery" aria-valuemin="0" aria-valuemax="100" aria-valuenow="${pct(m)}"><i style="width:${pct(m)}%"></i></span><span class="l-val">${pct(m)}%</span><small>${n ? `${n} question${n > 1 ? 's' : ''}` : 'self-reported'}</small></li>`; }).join('')}</ul>`;
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
  const above = mean((p.skills || []).map(k => mOf(st.model, k))) > (LEVEL_TARGET[p.level] ?? .5);
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
  $('#welcome').classList.add('hide'); $('#paths-btn').setAttribute('aria-expanded', 'true');
  renderSide(); A.updateOffset();
}
function closeSide() {
  if (side().hidden) return;
  side().hidden = true; document.body.classList.remove('learn-open'); $('#paths-btn').setAttribute('aria-expanded', 'false');
  if (view?.kind === 'path') A.clearRoute();
  if (welcomeBack && $('#panel').hidden && !A.S.current) $('#welcome').classList.remove('hide');
  welcomeBack = false; view = null; $('#learn-pill').hidden = true; A.updateOffset();
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
  st.model ||= newModel({}); observe(st.model, it, correct); save();
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
    case 'retake': closeSide(); return openOnboarding();
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
  opener = document.activeElement;
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
function syncPill() { $('#learn-pill').hidden = !(view?.kind === 'path' && !$('#panel').hidden); }
function renderWelcome() {
  const c = $('#learn-continue'); if (!c) return;
  const id = st.active || st.rec?.path, p = id && D?.pathById.get(id);
  c.hidden = !p; if (p) c.textContent = `Continue: ${p.title}`;
  c.dataset.path = p ? p.id : '';
}

export async function init(api) {
  A = api; esc = api.esc; st = load();
  modal().addEventListener('click', ev => { if (ev.target.matches('.l-backdrop')) return closeModal(); onClick(ev); });
  dlg().addEventListener('change', ev => { const l = ev.target.closest('.l-opt'); if (!l) return; l.closest('.l-opts').querySelectorAll('.l-opt').forEach(o => o.classList.toggle('on', o.querySelector('input').checked)); });
  side().addEventListener('click', onClick);
  document.addEventListener('keydown', onKey, true);
  $('#personalize-btn').onclick = () => openOnboarding();
  $('#paths-btn').onclick = async () => { if (!side().hidden && view?.kind === 'browse') return closeSide(); await loadData(); if (view?.kind === 'path') A.clearRoute(); openSide({ kind: 'browse' }); renderSide('.l-h'); };
  $('#learn-continue').onclick = async e => { await loadData(); openPath(e.currentTarget.dataset.path); };
  $('#learn-pill').onclick = () => { A.closePanel(); syncPill(); side().querySelector('.l-step[aria-expanded="true"]')?.focus(); };
  $('#t-overview').addEventListener('click', () => { closeSide(); });
  // Keep the path lit when the topic window closes or re-renders (closePanel/crossings clear S.route).
  new MutationObserver(() => {
    syncPill();
    if (view?.kind === 'path' && !A.S.route) { const r = routeOf(D.pathById.get(view.id)); A.highlightRoute(r.ids, r.edges); }
  }).observe($('#panel'), { attributes: true, attributeFilter: ['hidden'], childList: true });
  if (st.active || st.rec) { await loadData(); renderWelcome(); }
}
