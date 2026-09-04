/* Split out of preview.html 2026-08-13. Pure move - byte-identical code.
   Loaded with `defer` so it still runs after the DOM is parsed, which is what
   an inline script at the end of <body> did. Declaration ORDER inside this
   file is unchanged: a const used before its declaration was what blanked the
   page once already, and moving code is exactly when that recurs. */
/* ═══════════════════════════════════════════════════════════════════════════
   Essex Policy Assistant — blueprint UI (/preview).

   Conversation handling, markdown rendering, feedback posting, the mobile
   drawer and swipe-to-delete are carried over from static/index.html, which is
   the shipped and daily-used implementation; this file re-skins them rather
   than re-deriving them.

   Two things the mockup showed are deliberately NOT here:
     - confidence tags ("High confidence" / "Partial match"): the backend has no
       confidence signal and retrieval score does not track answer correctness,
       so any badge would be invented.
     - document section labels and "Updated <date>" lines: no such field exists
       in the index. Only real metadata (title, doc_type, academic_year) shows.
   ═══════════════════════════════════════════════════════════════════════════ */

let activeConversationId = null;
let currentScreen = 'chat';

const messagesEl = document.getElementById('messages');
const conversationList = document.getElementById('conversation-list');
const convEmpty = document.getElementById('conv-empty');

/* ── who is using this browser ──────────────────────────────────────────────
   A name, not a login. It travels as X-User on every request so each trial
   user gets their own history. It is SEPARATION, NOT SECURITY - anyone can
   type any name - and the prompt says so, because a box asking for your name
   invites the assumption that something is being verified. */
const USER_KEY = 'essex-assistant-user';
function loadUser() {
  try { return (localStorage.getItem(USER_KEY) || '').trim(); } catch (e) { return ''; }
}
function saveUser(name) {
  try { localStorage.setItem(USER_KEY, (name || '').trim()); } catch (e) {}
}
let currentUser = loadUser();

async function fetchJSON(url, options) {
  options = options || {};
  options.headers = { ...(options.headers || {}), 'X-User': currentUser || '' };
  const res = await fetch(url, options);
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

/* ── corner registration marks: REMOVED 2026-08-10 ────────────────────────
   The mockup drew a small "+" at each corner of every card, button and
   dialog. essex.ac.uk does not do this - its panels are plain square
   hairline boxes, and its one signature flourish is the 3px rule above and
   below a primary button (see .btn-primary in blueprint.css). The marks read
   as drafting-board decoration rather than Essex.

   Kept as no-ops rather than deleted so the [data-corners] attributes and the
   applyCorners() calls scattered through the render paths stay valid; drop
   both if the marks are never coming back. */
function addCorners() {}
function applyCorners() {}

/* ── screens ──────────────────────────────────────────────────────────────── */
function showScreen(name) {
  currentScreen = name;
  for (const el of document.querySelectorAll('.screen')) {
    el.classList.toggle('active', el.id === 'screen-' + name);
  }
  // The conversation list lives in the chat screen's sidebar, so without this
  // a returning user landing on onboarding would have no route back to their
  // history at all (the mockup has the same dead end).
  document.getElementById('nav-chat').hidden = (name === 'chat');
  setDrawer(false);
  if (name === 'chat') {
    if (!activeConversationId && !messagesEl.children.length) {
      messagesEl.appendChild(renderEmptyState());
    }
    messagesEl.scrollTop = messagesEl.scrollHeight;
  }
}

document.getElementById('brand-home').onclick = () => showScreen('chat');
document.getElementById('nav-settings').onclick = () => showScreen('settings');
document.getElementById('nav-chat').onclick = () => showScreen('chat');
document.getElementById('settings-back').onclick = () => showScreen('chat');

/* ── text helpers (from index.html) ───────────────────────────────────────── */
function escapeHtml(s) {
  return s.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
}
function inlineMd(s) {
  return s
    .replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>')
    // Absolute URLs, and the served paths that locally-ingested documents
    // (ingest_local.py) are cited by - which would otherwise be dead text
    // while a crawled document's citation is clickable.
    //
    // ONE alternation rather than two passes, deliberately: Essex's own URLs
    // contain "/documents/" (.../-/media/documents/...), so a second pass over
    // already-linked text re-matches inside the anchor it just built and
    // mangles every crawled citation. Matching once consumes the whole URL.
    // The lookbehind stops a bare path being matched mid-token.
    .replace(/(https?:\/\/[^\s<)]+|(?<![\w:/])\/documents\/[^\s<)]+)/g,
             '<a href="$1" target="_blank" rel="noopener">$1</a>');
}
function markdownToHtml(text) {
  const lines = escapeHtml(text).split('\n');
  let html = '', list = null, para = [];
  const flushPara = () => { if (para.length) { html += '<p>' + para.map(inlineMd).join('<br>') + '</p>'; para = []; } };
  const closeList = () => { if (list) { html += '</' + list + '>'; list = null; } };
  for (const raw of lines) {
    const line = raw.replace(/\s+$/, '');
    const bullet = line.match(/^\s*[-*•]\s+(.*)/);
    const numbered = line.match(/^\s*\d+[.)]\s+(.*)/);
    if (bullet) {
      flushPara();
      if (list !== 'ul') { closeList(); html += '<ul>'; list = 'ul'; }
      html += '<li>' + inlineMd(bullet[1]) + '</li>';
    } else if (numbered) {
      flushPara();
      if (list !== 'ol') { closeList(); html += '<ol>'; list = 'ol'; }
      html += '<li>' + inlineMd(numbered[1]) + '</li>';
    } else if (line.trim() === '') {
      flushPara(); closeList();
    } else {
      closeList(); para.push(line);
    }
  }
  flushPara(); closeList();
  return html;
}
// The only document name the backend has is the stored filename, so that is
// what is shown - tidied, never invented.
function readableTitle(url) {
  const name = (url.split('/').pop() || url).replace(/\.(pdf|docx?)$/i, '').replace(/[-_]+/g, ' ').trim();
  return name || url;
}
function normaliseQ(s) {
  return (s || '').toLowerCase().replace(/[\s’'".,?!]+/g, ' ').trim();
}
function relativeDate(ts) {
  if (!ts && ts !== 0) return '';
  const ms = Number(ts) < 1e12 ? Number(ts) * 1000 : Number(ts);
  const d = new Date(ms);
  if (isNaN(d.getTime())) return '';
  const startOf = (x) => new Date(x.getFullYear(), x.getMonth(), x.getDate()).getTime();
  const days = Math.round((startOf(new Date()) - startOf(d)) / 86400000);
  if (days <= 0) return 'Today';
  if (days === 1) return 'Yesterday';
  if (days < 7) return d.toLocaleDateString(undefined, { weekday: 'short' });
  return d.toLocaleDateString(undefined, { day: 'numeric', month: 'short' });
}

/* ── composer ─────────────────────────────────────────────────────────────── */
/* ── settings (localStorage) ──────────────────────────────────────────────────
   Declared HERE, above every use. buildComposer() is invoked at top level and
   reads `settings` for the scope selector; with these declarations further down
   the file that was a temporal-dead-zone ReferenceError, which threw during the
   initial render and left the page completely blank. `let`/`const` do not hoist
   like `function` does. */
/* ── settings (localStorage) ──────────────────────────────────────────────── */
const SETTINGS_KEY = 'essex-assistant-settings';
const DEFAULT_SETTINGS = { detail: 'default', partnerMode: 'essex_only', notifyDigest: false, notifyChanges: false };
function loadSettings() {
  try { return { ...DEFAULT_SETTINGS, ...JSON.parse(localStorage.getItem(SETTINGS_KEY) || '{}') }; }
  catch (e) { return { ...DEFAULT_SETTINGS }; }
}
let settings = loadSettings();
function saveSettings() {
  try { localStorage.setItem(SETTINGS_KEY, JSON.stringify(settings)); } catch (e) {}
}

function buildComposer() {
  const wrap = document.createElement('div');
  wrap.className = 'composer';
  wrap.innerHTML = `
    <div class="scope-row">
      <label class="scope" title="Which documents this question is answered from">
        <select id="scope-select">
          <option value="essex_only">University of Essex</option>
          <option value="partner_only">Partner colleges</option>
        </select>
      </label>
      <span class="scope-hint" id="scope-hint"></span>
      <a class="guide-link" href="/guide" target="_blank">How to use this</a>
    </div>
    <div class="composer-row">
      <textarea class="input" rows="1" placeholder="Ask about a policy or rule of assessment…"></textarea>
      <button class="send-btn" aria-label="Send">
        <svg viewBox="0 0 24 24" width="18" height="18" aria-hidden="true">
          <path d="M4 12h14M12 5l7 7-7 7" fill="none" stroke="currentColor"
                stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"/>
        </svg>
      </button>
    </div>
    <span class="text-muted composer-note">Assistant can make mistakes. Verify against the official document before acting.</span>`;
  const sel = wrap.querySelector('#scope-select');
  const hint = wrap.querySelector('#scope-hint');
  const HINTS = {
    essex_only: 'Answers use Essex documents only.',
    partner_only: 'Answers use partner-college documents only.',
  };
  sel.value = settings.partnerMode === 'partner_only' ? 'partner_only' : 'essex_only';
  hint.textContent = HINTS[sel.value];
  sel.addEventListener('change', () => {
    settings.partnerMode = sel.value; saveSettings();
    hint.textContent = HINTS[sel.value];
  });

  const ta = wrap.querySelector('textarea');
  const btn = wrap.querySelector('button');
  const grow = () => { ta.style.height = 'auto'; ta.style.height = Math.min(ta.scrollHeight, 180) + 'px'; };
  ta.addEventListener('input', grow);
  ta.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      const t = ta.value.trim();
      if (t) { ta.value = ''; grow(); send(t); }
    }
  });
  btn.onclick = () => {
    const t = ta.value.trim();
    if (t) { ta.value = ''; grow(); send(t); }
  };
  applyCorners(wrap);
  return { wrap, textarea: ta, button: btn };
}
// One composer now, not two: with the landing screen gone there is a single
// chat view, and a second composer would have nowhere to live.
const chatComposer = buildComposer();
document.getElementById('chat-composer-bar').appendChild(chatComposer.wrap);
function setSending(on) { chatComposer.button.disabled = on; }

/* ── example questions ────────────────────────────────────────────────────
   Real questions from the project's own evaluation sets (eval/questions*.json),
   i.e. questions this corpus is known to contain answers for - not invented
   copy dressed up as capability. */
const EXAMPLE_QUESTIONS = [
  'What constitutes plagiarism according to University of Essex policy?',
  'How long does a research degree viva last, and what happens if it runs long?',
  'Under which circumstances can a concern be raised under the whistleblowing policy?',
  'A student has failed a core taught module with a mark of 45. Can that failure be condoned?',
];
/* Rendered into the empty conversation area rather than a landing screen, so
   they appear whenever there is nothing to show and disappear the moment a
   conversation starts. */
function renderEmptyState() {
  const wrap = document.createElement('div');
  wrap.className = 'empty-wrap';
  const h = document.createElement('h1');
  h.textContent = 'What would you like to know?';
  const sub = document.createElement('p');
  sub.className = 'text-muted empty-sub';
  sub.textContent = 'Answers come from published Essex policy and rules-of-assessment '
    + 'documents, and cite the source they used.';
  const egh = document.createElement('div');
  egh.className = 'text-muted empty-egh';
  egh.textContent = 'Try asking';
  const grid = document.createElement('div');
  grid.className = 'empty-grid';
  for (const q of EXAMPLE_QUESTIONS) {
    const card = document.createElement('button');
    card.className = 'eg-card';
    card.innerHTML = `<div class="q"></div>
      <div class="go">Ask
        <svg viewBox="0 0 24 24" width="12" height="12" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"><line x1="5" y1="12" x2="19" y2="12"/><polyline points="13,6 19,12 13,18"/></svg>
      </div>`;
    card.querySelector('.q').textContent = q;
    card.onclick = () => send(q);
    grid.appendChild(card);
  }
  wrap.append(h, sub, egh, grid);
  return wrap;
}

/* ── follow-up suggestions ────────────────────────────────────────────────
   There is no follow-up endpoint and nothing in the backend generates these.
   They are derived here, in the browser, from terms that actually appear in
   the answer just given (bolded spans and multi-word proper-noun phrases) and
   from the real title of a cited document. Nothing is hardcoded per topic, and
   the row is labelled as built from the answer rather than presented as a
   model suggestion. When no term can be extracted, no chips are shown. */
const FU_BLOCKLIST = new Set([
  'university of essex', 'the university', 'essex policy', 'policy assistant',
]);
const FU_LEAD_STOP = new Set(['The', 'This', 'These', 'Those', 'A', 'An', 'It', 'If', 'You', 'Your',
  'However', 'Where', 'When', 'What', 'For', 'In', 'On', 'At', 'As', 'And', 'But', 'Note', 'Source',
  'Sources', 'Section', 'According']);

function extractTerms(text) {
  const out = [], seen = new Set();
  const push = (raw) => {
    let t = String(raw).replace(/\s+/g, ' ').replace(/^[^\w(]+|[^\w)]+$/g, '').trim();
    t = t.replace(/^(The|A|An)\s+/, '');
    if (t.length < 6 || t.length > 60) return;
    // Multi-word only. Single bolded words are nearly always list headers
    // ("Acknowledgment", "Inclusivity") and make a limp, ambiguous question.
    if (!/\s/.test(t)) return;
    const k = t.toLowerCase();
    if (seen.has(k) || FU_BLOCKLIST.has(k)) return;
    // near-duplicates ("Code of Practice for Postgraduate Research" vs the same
    // phrase + " Degrees") would otherwise fill both chip slots with one idea
    for (const prev of seen) if (prev.includes(k) || k.includes(prev)) return;
    seen.add(k); out.push(t);
  };
  for (const m of text.matchAll(/\*\*([^*\n]{4,60})\*\*/g)) push(m[1]);
  const plain = text.replace(/\*\*/g, '');
  const phrase = /\b([A-Z][a-z]{2,}(?:\s+(?:of|for|and|the|in|to|on)\s+[A-Z][a-z]{2,}|\s+[A-Z][a-z]{2,})+)\b/g;
  for (const m of plain.matchAll(phrase)) {
    const first = m[1].split(/\s+/)[0];
    if (FU_LEAD_STOP.has(first)) continue;
    push(m[1]);
  }
  return out;
}

function deriveFollowUps(answerText, sources) {
  const terms = extractTerms(answerText || '');
  const chips = [];
  if (terms[0]) chips.push(`What does the policy say about ${terms[0]}?`);
  if (terms[1]) chips.push(`Where is ${terms[1]} set out?`);
  if (chips.length && sources && sources.length) {
    const doc = readableTitle(sources[0]);
    if (doc.length <= 70) chips.push(`What else does "${doc}" cover?`);
  }
  return chips.slice(0, 3);
}

/* ── source chips + modal ─────────────────────────────────────────────────── */
const sourceCache = new Map();            // url -> {title, doc_type, academic_year, excerpt}
const modal = document.getElementById('source-modal');
let lastFocused = null;

function docKindLabel(docType) {
  if (!docType) return 'Document';
  const t = String(docType).toLowerCase();
  if (t === 'roa' || t.includes('rules')) return 'Rules of assessment';
  if (t === 'policy') return 'Policy document';
  return docType;
}

let modalUrl = null;

function paintModal(url, info, loading) {
  document.getElementById('dlg-title').textContent = info.title ? readableTitle(info.title) : readableTitle(url);
  document.getElementById('dlg-kind').textContent = docKindLabel(info.doc_type);
  const meta = document.getElementById('dlg-meta');
  meta.textContent = info.academic_year ? 'Academic year ' + info.academic_year : '';
  meta.hidden = !info.academic_year;
  const ex = document.getElementById('dlg-excerpt');
  if (info.excerpt) {
    // Real stored text from this document, the passage nearest the question -
    // see ingest.passages_for_documents for exactly what it is.
    ex.textContent = '“' + info.excerpt.trim() + '”';
  } else if (loading) {
    ex.textContent = 'Finding the matching passage…';
  } else {
    ex.textContent = 'No passage could be retrieved for this document — open the full PDF instead.';
  }
  document.getElementById('dlg-open').href = url;
}

async function openSourceModal(url, question) {
  lastFocused = document.activeElement;
  modalUrl = url;
  const cached = sourceCache.get(url);
  paintModal(url, cached || {}, !cached);
  modal.hidden = false;
  modal.querySelector('.dialog').focus();
  if (cached) return;
  // the per-answer prefetch has not landed (or failed) - fetch just this one
  try {
    const rows = await fetchJSON('/api/sources', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ urls: [url], question: question || '' }),
    });
    if (rows && rows[0]) sourceCache.set(url, rows[0]);
  } catch (e) { /* fall through to the no-passage message */ }
  if (!modal.hidden && modalUrl === url) paintModal(url, sourceCache.get(url) || {}, false);
}
function closeSourceModal() {
  modal.hidden = true;
  if (lastFocused && lastFocused.focus) lastFocused.focus();
}
modal.addEventListener('click', closeSourceModal);                       // backdrop closes
modal.querySelector('.dialog').addEventListener('click', (e) => e.stopPropagation());  // inside does not
document.getElementById('dlg-close').onclick = closeSourceModal;
document.getElementById('dlg-x').onclick = closeSourceModal;

// The mockup has no keyboard affordance; a modal that traps attention must be
// dismissable from the keyboard.
document.addEventListener('keydown', (e) => {
  if (e.key !== 'Escape') return;
  if (!modal.hidden) { closeSourceModal(); return; }
  setDrawer(false);
});

/* Metadata + a matching passage for the cited documents (POST /api/sources).
   Fired once per answer, non-blocking: chips render immediately with the
   URL-derived name and are upgraded in place when the lookup lands. */
async function hydrateSources(urls, question, chipEls) {
  const missing = urls.filter(u => !sourceCache.has(u));
  if (missing.length) {
    try {
      const rows = await fetchJSON('/api/sources', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ urls: missing, question: question || '' }),
      });
      for (const r of rows) sourceCache.set(r.url, r);
    } catch (e) { /* chips still work as links to the PDF */ }
  }
  for (const [url, el] of chipEls) {
    const info = sourceCache.get(url);
    if (info && info.title) el.querySelector('span').textContent = readableTitle(info.title);
  }
}

function buildSourceChips(sources, question) {
  const row = document.createElement('div');
  row.className = 'chip-row';
  const chipEls = [];
  for (const url of sources) {
    const b = document.createElement('button');
    b.className = 'tag tag-outline src-chip';
    b.title = url;
    b.innerHTML = `<svg viewBox="0 0 24 24" width="12" height="12" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"><rect x="5" y="3" width="14" height="18"/><line x1="8" y1="9" x2="16" y2="9"/><line x1="8" y1="13" x2="16" y2="13"/></svg><span></span>`;
    b.querySelector('span').textContent = readableTitle(url);
    b.onclick = () => openSourceModal(url, question);
    row.appendChild(b);
    chipEls.push([url, b]);
  }
  hydrateSources(sources, question, chipEls);
  return row;
}

/* ── feedback ─────────────────────────────────────────────────────────────
   Same payload and the same tag vocabulary as the shipped UI, so ratings from
   both pages land in one log that feedback_report.py can read. */
const FEEDBACK_TAGS = [
  ['wrong_programme', 'Wrong programme / document'],
  ['wrong_figure', 'Wrong or made-up figure'],
  ['outdated', 'Out of date / wrong year'],
  ['no_answer', "Didn't answer / too vague"],
  ['other', 'Other'],
];

function submitFeedback(rating, tags, comment, context, answer) {
  fetch('/api/feedback', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json', 'X-User': currentUser || '' },
    body: JSON.stringify({
      rating,
      question: context.question || '',
      answer,
      conversation_id: activeConversationId,
      retrieval_query: context.retrieval_query || null,
      sources: context.sources || [],
      ranked_top_urls: context.ranked_top_urls || [],
      tags,
      comment: comment || null,
    }),
  }).catch(() => {});
}

function buildFeedback(content, context, bar) {
  const note = document.createElement('span');
  note.className = 'text-muted rate-note';

  const up = document.createElement('button');
  up.className = 'btn btn-icon btn-secondary';
  up.title = 'Helpful'; up.setAttribute('aria-label', 'Helpful');
  up.innerHTML = '<svg viewBox="0 0 24 24" width="15" height="15" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"><path d="M7 10.5v9H4.5a1 1 0 0 1-1-1v-7a1 1 0 0 1 1-1H7z"/><path d="M7 10.5l4-7a2 2 0 0 1 2.9 2.3L13 9h5.2a2 2 0 0 1 2 2.5l-1.6 6a2 2 0 0 1-1.9 1.5H7"/></svg>';
  const down = document.createElement('button');
  down.className = 'btn btn-icon btn-secondary';
  down.title = 'Not helpful'; down.setAttribute('aria-label', 'Not helpful');
  down.innerHTML = '<svg viewBox="0 0 24 24" width="15" height="15" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"><path d="M7 13.5v-9H4.5a1 1 0 0 0-1 1v7a1 1 0 0 0 1 1H7z"/><path d="M7 13.5l4 7a2 2 0 0 0 2.9-2.3L13 15h5.2a2 2 0 0 0 2-2.5l-1.6-6A2 2 0 0 0 16.7 5H7"/></svg>';
  bar.append(note, up, down);

  const panel = document.createElement('div');
  panel.className = 'fb-panel';
  const prompt = document.createElement('div');
  prompt.className = 'fb-prompt text-muted';
  prompt.textContent = 'What went wrong? (optional — pick any that apply)';
  const tagRow = document.createElement('div');
  tagRow.className = 'chip-row';
  const selected = new Set();
  for (const [key, label] of FEEDBACK_TAGS) {
    const t = document.createElement('button');
    t.className = 'btn btn-secondary fb-tag';
    t.textContent = label;
    t.onclick = () => {
      if (selected.has(key)) { selected.delete(key); t.classList.remove('on'); }
      else { selected.add(key); t.classList.add('on'); }
    };
    tagRow.appendChild(t);
  }
  const comment = document.createElement('textarea');
  comment.className = 'input';
  comment.placeholder = 'Anything else? (optional)';
  comment.style.minHeight = '60px';
  const submit = document.createElement('button');
  submit.className = 'fb-submit';
  submit.textContent = 'Send feedback';
  panel.append(prompt, tagRow, comment, submit);

  // Single-select, and clicking the active one clears it. An 'up' is sent the
  // moment it is chosen; a 'down' is sent when the panel is submitted, so the
  // optional detail travels with it in one record rather than two. Clearing
  // resets the control - it cannot retract a rating already written to the
  // append-only log, so the control says so rather than pretending.
  let rating = null, sentUp = false, sentDown = false;
  const paint = () => {
    up.classList.toggle('rate-on', rating === 'up');
    down.classList.toggle('rate-on', rating === 'down');
    panel.classList.toggle('open', rating === 'down' && !sentDown);
  };
  up.onclick = () => {
    if (rating === 'up') { rating = null; note.textContent = sentUp ? 'Rating already recorded.' : ''; paint(); return; }
    rating = 'up'; paint();
    if (!sentUp) { submitFeedback('up', [], '', context, content); sentUp = true; }
    note.textContent = 'Thanks — recorded.';
  };
  down.onclick = () => {
    if (rating === 'down') { rating = null; note.textContent = sentDown ? 'Rating already recorded.' : ''; paint(); return; }
    rating = 'down'; note.textContent = sentDown ? 'Rating already recorded.' : ''; paint();
  };
  submit.onclick = () => {
    submitFeedback('down', [...selected], comment.value.trim(), context, content);
    sentDown = true;
    note.textContent = 'Thanks — logged for improvement.';
    paint();
  };

  return panel;
}

/* ── message rendering ────────────────────────────────────────────────────── */
const COPY_ICON = '<svg viewBox="0 0 24 24" width="13" height="13" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"><rect x="9" y="9" width="12" height="12" rx="2"/><path d="M5 15H4a1 1 0 0 1-1-1V4a1 1 0 0 1 1-1h10a1 1 0 0 1 1 1v1"/></svg>';
const COPIED_ICON = '<svg viewBox="0 0 24 24" width="13" height="13" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"><path d="M20 6L9 17l-5-5"/></svg>';

// Copies the raw text (markdown source for an answer, plain text for a
// question) rather than the rendered HTML - pasting into an email or doc
// should not carry the app's markup.
function buildCopyButton(getText) {
  const btn = document.createElement('button');
  btn.className = 'btn btn-icon btn-secondary copy-btn';
  btn.type = 'button';
  btn.title = 'Copy';
  btn.setAttribute('aria-label', 'Copy');
  btn.innerHTML = COPY_ICON;
  let resetTimer = null;
  btn.onclick = async () => {
    try {
      await navigator.clipboard.writeText(getText());
    } catch (e) {
      // Clipboard API unavailable (non-HTTPS/non-localhost context, or
      // permission denied) - fall back to the legacy copy command.
      const ta = document.createElement('textarea');
      ta.value = getText();
      ta.style.position = 'fixed';
      ta.style.opacity = '0';
      document.body.appendChild(ta);
      ta.select();
      try { document.execCommand('copy'); } catch (e2) { /* give up quietly */ }
      document.body.removeChild(ta);
    }
    btn.innerHTML = COPIED_ICON;
    btn.classList.add('copied');
    btn.title = 'Copied';
    clearTimeout(resetTimer);
    resetTimer = setTimeout(() => {
      btn.innerHTML = COPY_ICON;
      btn.classList.remove('copied');
      btn.title = 'Copy';
    }, 1500);
  };
  return btn;
}

function renderUser(text) {
  const hint = messagesEl.querySelector('.chat-hint');
  if (hint) hint.remove();
  const wrap = document.createElement('div');
  wrap.className = 'msg-user-wrap';
  const el = document.createElement('div');
  el.className = 'msg-user';
  el.textContent = text;
  wrap.appendChild(el);
  wrap.appendChild(buildCopyButton(() => text));
  messagesEl.appendChild(wrap);
  messagesEl.scrollTop = messagesEl.scrollHeight;
  return wrap;
}

function renderAssistant(content, sources, context) {
  const card = document.createElement('div');
  card.className = 'msg-assistant';
  card.dataset.corners = '';

  const body = document.createElement('div');
  body.className = 'msg-body';
  body.innerHTML = markdownToHtml(content);
  card.appendChild(body);

  if (sources && sources.length) {
    card.appendChild(buildSourceChips(sources, (context && context.question) || ''));
  }

  // Provenance: what produced this answer. Deliberately quiet - one small line,
  // full detail on hover - because it matters when someone questions an answer
  // months later, not while they are reading it.
  if (context && context.provenance) {
    const p = context.provenance;
    const el = document.createElement('div');
    el.className = 'provenance';
    const corpus = (p.corpus_version || '').slice(0, 7);
    el.textContent = `${p.generator || 'unknown model'} · corpus ${corpus || '?'} · build ${p.code_revision || '?'}`;
    el.title = `Answer generated by: ${p.generator}\n`
             + `Query rewriter: ${p.contextualizer}\n`
             + `Corpus version: ${p.corpus_version}\n`
             + `Code revision: ${p.code_revision}\n\n`
             + `Recorded so this answer can be explained later: policies change, `
             + `and knowing which corpus produced an answer is the difference between `
             + `"the rule was updated" and "the tool was wrong".`;
    card.appendChild(el);
  }

  // The rewritten retrieval query, shown only when it really differs from what
  // was typed - carried over from index.html, where it exists because a
  // contextualizer that misreads a topic switch is otherwise invisible.
  if (context && context.retrieval_query && context.question
      && normaliseQ(context.retrieval_query) !== normaliseQ(context.question)) {
    const rw = document.createElement('details');
    rw.className = 'rewrite text-muted';
    rw.innerHTML = '<summary>Searched for a rephrased version of your question</summary>'
      + '<div class="rewrite-q"></div>';
    rw.querySelector('.rewrite-q').textContent = context.retrieval_query;
    card.appendChild(rw);
  }

  if (context) {
    const chips = deriveFollowUps(content, sources || []);
    if (chips.length) {
      const box = document.createElement('div');
      box.className = 'followups';
      const label = document.createElement('div');
      label.className = 'fu-label text-muted';
      label.textContent = 'Ask next — built from the wording of this answer';
      const row = document.createElement('div');
      row.className = 'chip-row';
      for (const q of chips) {
        const b = document.createElement('button');
        b.className = 'btn btn-ghost fu-btn';
        b.textContent = q;
        b.onclick = () => send(q);
        row.appendChild(b);
      }
      box.append(label, row);
      card.appendChild(box);
    }
  }

  // Actions bar (rating when this answer has a context to rate against, plus
  // copy). Always present - unlike the question's copy button, this one
  // stays visible rather than hover-revealed, per instruction 2026-09-04.
  // Copy is appended AFTER buildFeedback populates the bar: .rate-note has
  // margin-right: auto, which shoves everything after it flush right as a
  // group - append copy before that and it strands alone on the left instead
  // of sitting next to the thumbs.
  const actions = document.createElement('div');
  actions.className = 'msg-actions';
  card.appendChild(actions);
  if (context) {
    card.appendChild(buildFeedback(content, context, actions));
  }
  actions.appendChild(buildCopyButton(() => content));

  messagesEl.appendChild(card);
  addCorners(card);
  messagesEl.scrollTop = messagesEl.scrollHeight;
  return card;
}

/* ── conversations ────────────────────────────────────────────────────────── */
async function loadConversations() {
  let convs = [];
  try { convs = await fetchJSON('/api/conversations'); } catch (e) { return; }
  conversationList.innerHTML = '';
  convEmpty.hidden = convs.length > 0;
  for (const c of convs) {
    const item = document.createElement('div');
    item.className = 'conv-item' + (c.id === activeConversationId ? ' active' : '');

    const actions = document.createElement('div');
    actions.className = 'conv-actions';
    const swipeDel = document.createElement('button');
    swipeDel.className = 'conv-swipe-del';
    swipeDel.textContent = 'Delete';
    swipeDel.onclick = (e) => { e.stopPropagation(); deleteConversation(c.id); };
    actions.appendChild(swipeDel);

    const content = document.createElement('button');
    content.className = 'conv-content';
    content.onclick = () => {
      if (closeSwipe(item)) return;      // a tap that dismisses a swipe must not also open the row
      setDrawer(false);
      openConversation(c.id);
    };
    const text = document.createElement('div');
    text.className = 'conv-text';
    const title = document.createElement('div');
    title.className = 'conv-title';
    title.textContent = (c.title || '').trim() || 'Untitled conversation';
    title.title = title.textContent;
    const date = document.createElement('div');
    date.className = 'conv-date text-muted';
    date.textContent = relativeDate(c.created_at);
    text.append(title, date);
    const del = document.createElement('button');
    del.className = 'conv-del';
    del.innerHTML = '&times;';
    del.title = 'Delete conversation';
    del.setAttribute('aria-label', 'Delete conversation');
    del.onclick = (e) => { e.stopPropagation(); deleteConversation(c.id); };
    content.append(text, del);

    item.append(actions, content);
    attachSwipe(item, content);
    conversationList.appendChild(item);
  }
}

async function deleteConversation(id) {
  await fetch('/api/conversations/' + id, { method: 'DELETE' }).catch(() => {});
  if (id === activeConversationId) {
    activeConversationId = null;
    messagesEl.innerHTML = '';
    showScreen('chat');
  }
  await loadConversations();
}

/* An answer can be perfectly faithful to a rule that has since been rewritten.
   Conversations already record what was cited (the URLs are printed in the
   answer) and when, so this needs no per-user state and no accounts: for each
   stored answer, ask whether any document it cited changed after that answer
   was given. Failure is silent - a missing marker is a smaller harm than a
   false one, which would teach people to ignore it. */
async function markStaleAnswers(rendered) {
  for (const r of rendered) {
    const urls = [...new Set((r.content.match(/https?:\/\/[^\s)\]]+/g) || [])
      .map(u => u.replace(/[.,;]+$/, '')))];
    if (!urls.length || !r.at) continue;
    let res;
    try {
      res = await fetchJSON('/api/staleness', {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ urls, since: r.at }),
      });
    } catch (e) { continue; }
    if (!res.stale || !res.stale.length) continue;
    const warn = document.createElement('div');
    warn.className = 'stale-warn';
    const names = res.stale.map(s => s.title).join(', ');
    warn.textContent = res.stale.length === 1
      ? `The document this answer cited has been updated since: ${names}. Re-ask to get the current rule.`
      : `${res.stale.length} documents this answer cited have been updated since: ${names}. Re-ask to get the current rules.`;
    r.el.appendChild(warn);
  }
}

async function openConversation(id) {
  activeConversationId = id;
  messagesEl.innerHTML = '';
  showScreen('chat');
  let msgs = [];
  try { msgs = await fetchJSON(`/api/conversations/${id}/messages`); } catch (e) { msgs = []; }
  let lastUserQ = '';
  const rendered = [];
  for (const m of msgs) {
    if (m.role === 'user') { lastUserQ = m.content; renderUser(m.content); }
    // Stored messages carry no sources: the messages endpoint returns role and
    // content only. Citations therefore appear on live answers, not on reloaded
    // history - inventing them here is not an option. The URLs printed in the
    // answer text ARE recoverable though, which is enough for the staleness
    // check below.
    else {
      const el = renderAssistant(m.content, null, { question: lastUserQ });
      rendered.push({ el, content: m.content, at: m.created_at });
    }
  }
  markStaleAnswers(rendered);
  await loadConversations();
}

document.getElementById('new-conv-btn').onclick = async () => {
  try {
    const conv = await fetchJSON('/api/conversations', {
      method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({}),
    });
    await openConversation(conv.id);
    chatComposer.textarea.focus();
  } catch (e) { /* leave the current view alone */ }
};

/* ── sending ──────────────────────────────────────────────────────────────── */
async function send(text) {
  text = (text || '').trim();
  if (!text) return;

  if (!activeConversationId) {
    try {
      const conv = await fetchJSON('/api/conversations', {
        method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({}),
      });
      activeConversationId = conv.id;
    } catch (e) {
      alert('Could not start a conversation: ' + e.message);
      return;
    }
  }
  showScreen('chat');
  renderUser(text);
  setSending(true);

  const pending = document.createElement('div');
  pending.className = 'msg-assistant msg-pending';
  pending.textContent = 'Searching the policy documents…';
  messagesEl.appendChild(pending);
  messagesEl.scrollTop = messagesEl.scrollHeight;

  try {
    // Streamed, so text appears as it is written instead of after ~7s of
    // nothing. Falls back to the blocking endpoint if SSE fails for any
    // reason - a streaming transport problem must not cost the user an answer.
    let result;
    try {
      result = await sendStreaming(text, pending);
    } catch (streamErr) {
      if (streamErr && streamErr.committed) {
        // The server already stored the question; retrying would duplicate it.
        // Surface the failure instead of quietly asking twice.
        throw streamErr;
      }
      console.warn('stream failed before any work, retrying without streaming', streamErr);
      result = await fetchJSON(`/api/conversations/${activeConversationId}/messages`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ content: text, detail: settings.detail || 'default',
                             partner_mode: settings.partnerMode || 'essex_only' }),
      });
    }
    pending.remove();
    drip.reset();
    if (streamedEl) { streamedEl.remove(); streamedEl = null; }
    renderAssistant(result.answer, result.sources, {
      question: text,
      retrieval_query: result.retrieval_query,
      sources: result.sources,
      ranked_top_urls: result.ranked_top_urls,
      provenance: result.provenance,
    });
    await loadConversations();
    // The generated title lands ~5s after the answer (background task), so the
    // list still shows the truncated fallback until it is re-read.
    if (result.title_pending) {
      setTimeout(loadConversations, 3500);
      setTimeout(loadConversations, 9000);
    }
  } catch (err) {
    pending.remove();
    drip.reset();
    if (streamedEl) { streamedEl.remove(); streamedEl = null; }
    // err.message is already a sentence written for a reader (see
    // _friendly_error in src/app.py). Only strip a JSON wrapper if one
    // survived, rather than showing the user a payload.
    let msg = (err && err.message) || 'Something went wrong.';
    try { const j = JSON.parse(msg); msg = j.detail || j.message || msg; } catch (e) {}
    renderAssistant(msg, null, null);
  } finally {
    setSending(false);
  }
}



/* Warn when the server fell back to the local model unintentionally. */
(async function checkDegraded() {
  try {
    const cfg = await fetchJSON('/api/config');
    if (cfg && cfg.degraded) {
      const b = document.getElementById('degraded-banner');
      b.textContent = 'Running on the local model \u2014 the cloud generator is unreachable, so answers may be weaker than usual.';
      b.hidden = false;
    }
  } catch (e) { /* a missing endpoint must never break the page */ }
})();




/* ── smooth output ───────────────────────────────────────────────────────────
   The generator does not arrive smoothly. Measured on a real answer: 9 events,
   a median of 154 characters each, 614ms apart - a whole paragraph appears,
   then nothing for two-thirds of a second. Appending each chunk straight to the
   DOM reproduces that stutter exactly.

   This buffers arrivals and releases them at a steady rate, decoupling display
   from network timing. The rate ADAPTS: it speeds up when the buffer grows, so
   the answer still finishes when the answer finishes rather than seconds
   later. */
const drip = (function () {
  let buf = '', shown = '', el = null, raf = null, last = 0;
  const BASE = 220;            // chars/sec, close to the measured arrival rate

  function frame(now) {
    if (!last) last = now;
    const dt = Math.min((now - last) / 1000, 0.1);   // ignore tab-away jumps
    last = now;
    if (buf.length) {
      const rate = Math.max(BASE, buf.length * 3);   // never lag far behind
      const take = Math.max(1, Math.round(rate * dt));
      shown += buf.slice(0, take);
      buf = buf.slice(take);
      if (el) el.textContent = shown;
      const nearBottom =
        messagesEl.scrollHeight - messagesEl.scrollTop - messagesEl.clientHeight < 120;
      if (nearBottom) messagesEl.scrollTop = messagesEl.scrollHeight;
    }
    raf = buf.length ? requestAnimationFrame(frame) : null;
    if (!buf.length) last = 0;
  }

  return {
    attach(node) { el = node; buf = ''; shown = ''; },
    push(text) {
      buf += text;
      if (el && !raf) { last = 0; raf = requestAnimationFrame(frame); }
    },
    /* Stream ended: stop animating and show the rest at once. renderAssistant
       re-renders the final answer as markdown anyway, so dripping the tail out
       would only delay it. */
    finish() {
      if (raf) cancelAnimationFrame(raf);
      raf = null;
      if (el && buf.length) { shown += buf; el.textContent = shown; }
      buf = '';
    },
    reset() { this.finish(); el = null; shown = ''; },
  };
})();

/* ── streaming ────────────────────────────────────────────────────────────
   Reads the SSE endpoint and paints tokens as they arrive. The final `done`
   event carries the same JSON the blocking endpoint returns, so the caller
   re-renders from that: the streamed element is a PREVIEW, and the answer the
   user ends up with is always the stored one, formatted the normal way. */
let streamedEl = null;

async function sendStreaming(text, pending) {
  const res = await fetch(`/api/conversations/${activeConversationId}/messages/stream`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json', 'X-User': currentUser || '' },
    body: JSON.stringify({ content: text, detail: settings.detail || 'default',
                             partner_mode: settings.partnerMode || 'essex_only' }),
  });
  // Before this point the server has done nothing, so a failure here is safe to
  // retry on the blocking endpoint. After it, the user message is already
  // stored and retrying would store it TWICE.
  if (!res.ok || !res.body) throw new Error('stream unavailable (' + res.status + ')');

  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buf = '', done = null, failed = null;

  while (true) {
    const { value, done: finished } = await reader.read();
    if (finished) break;
    buf += decoder.decode(value, { stream: true });
    // SSE frames are separated by a blank line; keep any partial tail in buf
    const frames = buf.split('\n\n');
    buf = frames.pop();
    for (const frame of frames) {
      let event = 'message', data = '';
      for (const line of frame.split('\n')) {
        if (line.startsWith('event:')) event = line.slice(6).trim();
        else if (line.startsWith('data:')) data += line.slice(5).trim();
      }
      if (!data) continue;
      let parsed;
      try { parsed = JSON.parse(data); } catch (e) { continue; }
      if (event === 'stage') {
        // The ~1-3s search happens before any token exists, so streaming
        // cannot fill it. Naming the phase is the only honest thing to show.
        if (parsed === 'retrieving') pending.textContent = 'Searching the policy documents…';
      } else if (event === 'token') {
        if (pending.textContent !== 'Writing the answer…') pending.textContent = 'Writing the answer…';
        if (!streamedEl) {
          pending.remove();
          streamedEl = document.createElement('div');
          streamedEl.className = 'msg-assistant';
          messagesEl.appendChild(streamedEl);
          drip.attach(streamedEl);
        }
        drip.push(parsed);
      } else if (event === 'done') {
        done = parsed;
      } else if (event === 'error') {
        failed = parsed;
      }
    }
  }
  drip.finish();
  if (failed) { const e = new Error(failed); e.committed = true; throw e; }
  if (!done) { const e = new Error('stream ended without a result'); e.committed = true; throw e; }
  return done;
}

/* ── mobile drawer ────────────────────────────────────────────────────────── */
const backdrop = document.getElementById('drawer-backdrop');
const menuBtn = document.getElementById('menu-btn');
function setDrawer(open) {
  document.body.classList.toggle('drawer-open', open);
  menuBtn.setAttribute('aria-expanded', open ? 'true' : 'false');
}
menuBtn.onclick = () => {
  if (currentScreen !== 'chat') showScreen('chat');
  setDrawer(!document.body.classList.contains('drawer-open'));
};
backdrop.onclick = () => setDrawer(false);

/* ── swipe-to-delete (carried over from index.html) ───────────────────────── */
const SWIPE_WIDTH = 84;
const SWIPE_OPEN_AT = 42;
let openSwipeItem = null;

function closeSwipe(except) {
  if (openSwipeItem && openSwipeItem !== except) {
    const c = openSwipeItem.querySelector('.conv-content');
    if (c) c.style.transform = 'translateX(0)';
    openSwipeItem.classList.remove('swipe-open');
    openSwipeItem = null;
    return true;
  }
  if (openSwipeItem === except && except) {
    const c = except.querySelector('.conv-content');
    if (c) c.style.transform = 'translateX(0)';
    except.classList.remove('swipe-open');
    openSwipeItem = null;
    return true;
  }
  return false;
}

function attachSwipe(item, content) {
  let x0 = 0, y0 = 0, dx = 0, dragging = false, decided = false;
  item.addEventListener('touchstart', (e) => {
    if (e.touches.length !== 1) return;
    if (openSwipeItem && openSwipeItem !== item) closeSwipe(null);
    x0 = e.touches[0].clientX; y0 = e.touches[0].clientY;
    dx = 0; dragging = true; decided = false;
    // nothing visible changes here on purpose: iOS treats a first tap that
    // changes appearance as a hover pass and swallows the click
  }, { passive: true });
  item.addEventListener('touchmove', (e) => {
    if (!dragging) return;
    const nx = e.touches[0].clientX - x0;
    const ny = e.touches[0].clientY - y0;
    if (!decided) {
      if (Math.abs(ny) > Math.abs(nx)) { dragging = false; item.classList.remove('swiping'); return; }
      if (Math.abs(nx) < 8) return;
      decided = true;
      item.classList.add('swiping');
    }
    const base = openSwipeItem === item ? -SWIPE_WIDTH : 0;
    dx = Math.max(-SWIPE_WIDTH, Math.min(0, base + nx));
    content.style.transform = `translateX(${dx}px)`;
  }, { passive: true });
  function end() {
    if (!dragging) return;
    dragging = false;
    item.classList.remove('swiping');
    if (!decided) return;
    const open = dx <= -SWIPE_OPEN_AT;
    content.style.transform = open ? `translateX(${-SWIPE_WIDTH}px)` : 'translateX(0)';
    item.classList.toggle('swipe-open', open);
    openSwipeItem = open ? item : (openSwipeItem === item ? null : openSwipeItem);
  }
  item.addEventListener('touchend', end, { passive: true });
  item.addEventListener('touchcancel', end, { passive: true });
}
document.addEventListener('touchstart', (e) => {
  if (openSwipeItem && !openSwipeItem.contains(e.target)) closeSwipe(null);
}, { passive: true });

function paintSettings() {
  for (const r of document.querySelectorAll('#detail-seg input')) r.checked = (r.value === settings.detail);
  document.getElementById('tg-digest').setAttribute('aria-pressed', settings.notifyDigest ? 'true' : 'false');
}
for (const r of document.querySelectorAll('#detail-seg input')) {
  r.addEventListener('change', () => { settings.detail = r.value; saveSettings(); });
}
document.getElementById('tg-digest').onclick = () => {
  settings.notifyDigest = !settings.notifyDigest; saveSettings(); paintSettings();
};
paintSettings();

const historyStatus = document.getElementById('history-status');
function setHistoryStatus(msg) {
  historyStatus.textContent = msg;
  historyStatus.hidden = !msg;
}

document.getElementById('export-history').onclick = async () => {
  setHistoryStatus('Collecting conversations…');
  try {
    const convs = await fetchJSON('/api/conversations');
    const out = [];
    for (const c of convs) {
      let msgs = [];
      try { msgs = await fetchJSON(`/api/conversations/${c.id}/messages`); } catch (e) {}
      out.push({ ...c, messages: msgs });
    }
    const blob = new Blob([JSON.stringify(out, null, 2)], { type: 'application/json' });
    const a = document.createElement('a');
    a.href = URL.createObjectURL(blob);
    a.download = 'essex-assistant-history.json';
    a.click();
    setTimeout(() => URL.revokeObjectURL(a.href), 5000);
    setHistoryStatus(`Exported ${out.length} conversation${out.length === 1 ? '' : 's'}.`);
  } catch (e) {
    setHistoryStatus('Export failed: ' + e.message);
  }
};

document.getElementById('clear-history').onclick = async () => {
  /* A destructive action has to make its result unmistakable. The first
     version deleted everything and reported it with one muted line on the
     Settings screen, where the conversation list is not visible - success and
     failure looked identical, and it read as a dead button. It also fired
     ~115 sequential DELETEs while refreshing the list only at the very end,
     so navigating away mid-loop showed a stale sidebar that still listed
     conversations already gone. Both are fixed below. */
  let convs = [];
  try { convs = await fetchJSON('/api/conversations'); }
  catch (e) { setHistoryStatus('Could not read history — nothing was deleted.'); return; }
  if (!convs.length) { setHistoryStatus('There is no history to clear.'); return; }

  // OK/Cancel confirm. The count is in the message so the dialog states what
  // is actually at stake rather than asking abstractly.
  if (!confirm(
        `Delete all ${convs.length} conversation${convs.length === 1 ? '' : 's'}?\n\n` +
        `This cannot be undone.`)) {
    setHistoryStatus('Cancelled — nothing was deleted.');
    return;
  }

  const btn = document.getElementById('clear-history');
  btn.disabled = true;
  let done = 0, failed = 0;
  for (const c of convs) {
    try {
      const r = await fetch('/api/conversations/' + c.id, { method: 'DELETE' });
      r.ok ? done++ : failed++;
    } catch (e) { failed++; }
    // Progress, and the list refreshed AS IT GOES, so the sidebar can never
    // show conversations that no longer exist.
    setHistoryStatus(`Deleting… ${done} of ${convs.length}`);
    if (done % 10 === 0) await loadConversations();
  }
  activeConversationId = null;
  messagesEl.innerHTML = '';
  await loadConversations();
  btn.disabled = false;
  setHistoryStatus(failed
    ? `Deleted ${done} conversation${done === 1 ? '' : 's'}; ${failed} could not be deleted.`
    : `Deleted ${done} conversation${done === 1 ? '' : 's'}. Your history is now empty.`);
};

/* ── boot ─────────────────────────────────────────────────────────────────── */
showScreen('chat');
loadConversations();

/* Ask once, on first visit. Deliberately not a modal that blocks the app: a
   trial user who skips it still gets a working assistant, just a shared
   history - which is the pre-existing behaviour, not a regression. */
(function askName() {
  if (currentUser) return;
  const bar = document.createElement('div');
  bar.className = 'name-bar';
  bar.innerHTML = `
    <span>Your name, so your conversations stay separate from other people's:</span>
    <input id="name-input" placeholder="e.g. Michael" maxlength="60" autocomplete="off">
    <button id="name-save">Save</button>
    <span class="name-note">Not a login \u2014 it only separates histories.</span>`;
  document.body.insertBefore(bar, document.body.firstChild);
  const inp = bar.querySelector('#name-input');
  const go = () => {
    const v = (inp.value || '').trim();
    if (!v) return;
    currentUser = v; saveUser(v); bar.remove();
    if (typeof loadConversations === 'function') loadConversations();
  };
  bar.querySelector('#name-save').onclick = go;
  inp.addEventListener('keydown', e => { if (e.key === 'Enter') go(); });
  inp.focus();
})();
