/* Shared chrome and chart primitives for the three dashboard pages.
 *
 * WHY A SHARED FILE. The pages answer three different questions - what did
 * people think (/feedback), what is the system doing (/insights), what does it
 * cost and how fast is it (/health) - but they render the same bars, the same
 * escaping, the same percentages. Three inline copies would drift, and the
 * first thing to drift silently is the thing every panel depends on: esc().
 *
 * WHY HAND-ROLLED SVG rather than Chart.js or D3. This app has no build step
 * and loads no external JavaScript. A CDN dependency means the dashboard
 * renders blank whenever that CDN is unreachable - and everything here is
 * bars, a histogram and a sparkline, which are a few dozen lines each. A
 * library earns its keep when charts become zoomable and animated; these are
 * not that, and a blank page costs more than the polish is worth.
 */

/* ── primitives ──────────────────────────────────────────────────────────── */
function esc(s) {
  const d = document.createElement('div');
  d.textContent = s == null ? '' : String(s);
  return d.innerHTML;
}
const pct = (n, d) => (d ? Math.round((n / d) * 100) : 0);
const fileOf = u => String(u || '').replace(/\/$/, '').split('/').pop();
function when(ts) {
  if (!ts) return '';
  try { return new Date(ts).toLocaleString(); } catch (e) { return ts; }
}
/* Numbers are read off a screen, not summed by hand: 1,284 beats 1284 and
 * £0.0043 beats 0.004300000000001. */
const num = n => (n == null ? '—' : Number(n).toLocaleString());
function money(usd) {
  if (usd == null) return '—';
  if (usd === 0) return '$0';
  return usd < 0.01 ? '$' + usd.toFixed(4) : '$' + usd.toFixed(2);
}

/* ── navigation ──────────────────────────────────────────────────────────── */
/* One header across all three, so they read as one tool rather than three
 * pages that happen to share a colour. */
const DASH_PAGES = [
  ['/feedback', 'Feedback', 'what people thought'],
  ['/insights', 'Insights', 'what the system is doing'],
  ['/health',   'Health',   'speed, spend and failures'],
];
function dashHeader(active, statsHTML) {
  const tabs = DASH_PAGES.map(([href, label, sub]) =>
    `<a href="${href}" class="tab${href === active ? ' on' : ''}" title="${esc(sub)}">${esc(label)}</a>`
  ).join('');
  return `<header>
    <div class="bar"><h1>Essex Policy Assistant</h1><nav>${tabs}</nav>
      <a class="back" href="/">&larr; Back to chat</a></div>
    <div class="stats" id="stats">${statsHTML || 'loading…'}</div>
  </header>`;
}

/* ── charts ──────────────────────────────────────────────────────────────── */

/* A split bar: the down/up proportion of one row. */
function bar(dn, up) {
  const t = dn + up || 1;
  return `<div class="bar-track"><div class="dn" style="width:${(dn / t) * 100}%"></div>` +
         `<div class="up" style="width:${(up / t) * 100}%"></div></div>`;
}

/* Horizontal ranked bars. `rows` is [{label, value, title?, key?}].
 * onPick makes a bar clickable, which is what turns a chart into a filter. */
function rankedBars(rows, opts) {
  opts = opts || {};
  if (!rows.length) return '<div class="sparse">No data.</div>';
  const max = Math.max(...rows.map(r => r.value), 1);
  return rows.map(r =>
    `<div class="row${opts.pickable ? ' pickable' : ''}"${opts.pickable ? ` data-pick="${esc(r.key != null ? r.key : r.label)}"` : ''}>
       <span class="lbl" title="${esc(r.title || r.label)}">${esc(r.label)}</span>
       <div class="bar-track"><div class="fill" style="width:${(r.value / max) * 100}%"></div></div>
       <span class="n">${esc(opts.fmt ? opts.fmt(r.value) : num(r.value))}</span>
     </div>`).join('');
}

/* Percentiles. A mean hides the tail, and the tail is the complaint - the
 * project's own note says the ~9s median "has a long tail worth attributing
 * from real traffic rather than estimating a fourth time". */
function percentile(sorted, p) {
  if (!sorted.length) return null;
  const i = Math.min(sorted.length - 1, Math.floor((p / 100) * sorted.length));
  return sorted[i];
}
function pctlRow(label, values) {
  const s = values.slice().sort((a, b) => a - b);
  const p50 = percentile(s, 50), p90 = percentile(s, 90), p99 = percentile(s, 99);
  const worst = s[s.length - 1] || 1;
  const seg = (v, cls) => `<div class="${cls}" style="left:${(v / worst) * 100}%"></div>`;
  return `<div class="pctl">
    <div class="pctl-head"><span class="lbl">${esc(label)}</span>
      <span class="n">p50 ${p50.toFixed(2)}s · p90 ${p90.toFixed(2)}s · p99 ${p99.toFixed(2)}s · n=${s.length}</span></div>
    <div class="pctl-track"><div class="pctl-fill" style="width:${(p90 / worst) * 100}%"></div>
      ${seg(p50, 'tick p50')}${seg(p99, 'tick p99')}</div></div>`;
}

/* Histogram as inline SVG. Shows the SHAPE a percentile list flattens - two
 * clusters and a long tail all report the same p50. */
function histogram(values, opts) {
  opts = opts || {};
  if (!values.length) return '<div class="sparse">No data.</div>';
  const bins = opts.bins || 24;
  const max = opts.max || Math.max(...values);
  const counts = new Array(bins).fill(0);
  values.forEach(v => {
    const i = Math.min(bins - 1, Math.floor((v / max) * bins));
    if (i >= 0) counts[i]++;
  });
  const peak = Math.max(...counts, 1);
  const W = 100, H = 42;
  const bw = W / bins;
  const barsSVG = counts.map((c, i) => {
    const h = (c / peak) * H;
    return `<rect x="${(i * bw).toFixed(2)}" y="${(H - h).toFixed(2)}" width="${(bw * 0.86).toFixed(2)}" ` +
           `height="${h.toFixed(2)}" rx="0.4"><title>${(i * max / bins).toFixed(1)}–${((i + 1) * max / bins).toFixed(1)}${esc(opts.unit || '')}: ${c}</title></rect>`;
  }).join('');
  return `<svg class="hist" viewBox="0 0 ${W} ${H}" preserveAspectRatio="none" role="img"
      aria-label="${esc(opts.label || 'distribution')}">${barsSVG}</svg>
    <div class="axis"><span>0${esc(opts.unit || '')}</span><span>${max.toFixed(1)}${esc(opts.unit || '')}</span></div>`;
}

/* Daily sparkline. `series` is [[dayISO, value], ...] already sorted. */
function sparkline(series, opts) {
  opts = opts || {};
  if (series.length < 2) return '<div class="sparse">Not enough days yet.</div>';
  const vals = series.map(s => s[1]);
  const max = Math.max(...vals, 1);
  const W = 100, H = 30;
  const step = W / (series.length - 1);
  const pts = series.map((s, i) => `${(i * step).toFixed(2)},${(H - (s[1] / max) * H).toFixed(2)}`);
  const dots = series.map((s, i) =>
    `<circle cx="${(i * step).toFixed(2)}" cy="${(H - (s[1] / max) * H).toFixed(2)}" r="0.9">
       <title>${esc(s[0])}: ${esc(opts.fmt ? opts.fmt(s[1]) : num(s[1]))}</title></circle>`).join('');
  return `<svg class="spark" viewBox="0 0 ${W} ${H}" preserveAspectRatio="none" role="img"
      aria-label="${esc(opts.label || 'over time')}">
      <polyline points="${pts.join(' ')}" fill="none" stroke-width="0.8"/>${dots}</svg>
    <div class="axis"><span>${esc(series[0][0])}</span><span>peak ${esc(opts.fmt ? opts.fmt(max) : num(max))}</span><span>${esc(series[series.length - 1][0])}</span></div>`;
}

/* ── helpers ─────────────────────────────────────────────────────────────── */
function byDay(rows, tsKey, valueFn) {
  const m = {};
  rows.forEach(r => {
    let ts = r[tsKey];
    if (typeof ts === 'number') ts = new Date(ts * 1000).toISOString();
    const day = String(ts || '').slice(0, 10);
    if (!day) return;
    m[day] = (m[day] || 0) + (valueFn ? valueFn(r) : 1);
  });
  return Object.keys(m).sort().map(k => [k, m[k]]);
}
function counter(rows, keyFn) {
  const c = {};
  rows.forEach(r => { const k = keyFn(r); if (k == null || k === '') return; c[k] = (c[k] || 0) + 1; });
  return c;
}
function toRows(countObj, limit) {
  const rows = Object.keys(countObj)
    .sort((a, b) => countObj[b] - countObj[a])
    .map(k => ({ label: k, value: countObj[k] }));
  return limit ? rows.slice(0, limit) : rows;
}
/* Every panel states the sample it was drawn from. The provenance panel once
 * drew three points looking exactly like one drawn on all 38 - a denominator
 * in the subtitle is the cheapest defence against reading noise as a finding. */
function panel(title, why, body, opts) {
  opts = opts || {};
  return `<div class="panel${opts.wide ? ' wide' : ''}"${opts.id ? ` id="${opts.id}"` : ''}>
    <h2>${esc(title)}</h2><p class="why">${why}</p>${body}</div>`;
}
function fetchJSON(url) {
  return fetch(url).then(r => {
    if (!r.ok) throw new Error('HTTP ' + r.status);
    return r.json();
  });
}


/* ── alert surface ───────────────────────────────────────────────────────────
 * The macOS banner is transient and WILL be missed - the user said so, and has
 * already missed one from this channel. So the dashboard is the real alerting
 * surface, and the design rule follows from that: landing on ANY of the three
 * pages while something is broken must be impossible to mistake for normal.
 *
 * Three cues, loudest first: a full-width coloured bar at the top of the page,
 * a status pill in the header, and the browser tab title. The tab title matters
 * more than it looks - it is the one cue visible without the page in focus.
 */
const SEV_RANK = {critical: 0, high: 1, medium: 2, low: 3};
/* Hours before a silent monitor is itself treated as a problem. Generous
 * against the hourly schedule, so one missed run is not called a dead monitor. */
const MONITOR_STALE_H = 3;

function alertState(m) {
  if (!m || m.never_run) {
    return {level: 'stale', label: 'Monitor not running',
            note: 'The monitor has never run, so nothing is being checked.'};
  }
  const ageH = m.generated_at ? (Date.now() - Date.parse(m.generated_at)) / 3600000 : null;
  const stale = ageH == null || ageH > MONITOR_STALE_H;
  const staleNote = `The monitor last checked ${ageH == null ? 'at an unknown time' : ageH.toFixed(1) + ' hours ago'}.` +
    ' A silent monitor looks exactly like a healthy one, so treat this as unknown, not all-clear.';

  // NOT `SEV_RANK[x] || 9`: critical ranks 0, which is falsy, so `||` would
  // sort the most serious alert to the BOTTOM. And not `??` either - the
  // project's JS parser rejects it.
  const rank = sev => (Object.prototype.hasOwnProperty.call(SEV_RANK, sev) ? SEV_RANK[sev] : 9);
  const alerts = (m.alerts || []).slice().sort((a, b) => rank(a.severity) - rank(b.severity));

  // Staleness must NOT suppress alerts the monitor already found. A first
  // version returned early on stale and hid a CRITICAL alert behind a grey
  // "has not checked in" bar - the loudest possible thing to get wrong, and
  // caught only by rendering the combination rather than each state alone.
  // An out-of-date "the system is down" is still "the system is down".
  if (alerts.length) {
    return {level: alerts[0].severity,
            label: `${alerts.length} problem${alerts.length > 1 ? 's' : ''} detected` +
                   (stale ? ' — and the monitor is out of date' : ''),
            note: stale ? staleNote : '', alerts};
  }
  if (stale) {
    return {level: 'stale', label: 'Monitor has not checked in', note: staleNote, alerts: []};
  }
  return {level: 'ok', label: 'All clear', alerts: []};
}

function alertBarHTML(m) {
  const st = alertState(m);
  const when_ = m && m.generated_at ? `checked ${when(m.generated_at)}` : '';
  const items = (st.alerts || []).map(a => `
    <div class="ab-item">
      <span class="ab-sev ${esc(a.severity)}">${esc(a.severity)}</span>
      <div><div class="ab-title">${esc(a.title)}</div>
           <div class="ab-detail">${esc(a.detail || '')}</div></div>
    </div>`).join('');
  const foot = st.level === 'ok'
    ? `<div class="ab-foot">Nothing wrong right now. Past alerts are kept on
         <a href="/health">Health</a> even when the condition has cleared.</div>`
    : (st.note ? `<div class="ab-foot">${esc(st.note)}</div>`
               : `<div class="ab-foot">Details and history on <a href="/health">Health</a>.</div>`);
  return `<div class="alertbar ${esc(st.level)}">
    <div class="ab-head"><span class="dot"></span>${esc(st.label)}
      <span class="ab-when">${esc(when_)}</span></div>
    ${items ? `<div class="ab-list">${items}</div>` : ''}
    ${foot}</div>`;
}

/* Call once per page, after the header is in the DOM. Fetches on its own so no
 * page has to remember to; a failure leaves the page working and says so,
 * because a dashboard that hides its own broken alerting is the worst outcome
 * available here. */
function mountAlerts() {
  const main = document.querySelector('main');
  if (!main) return;
  const holder = document.createElement('div');
  main.insertBefore(holder, main.firstChild);
  const paint = (m, failed) => {
    if (failed) {
      holder.innerHTML = `<div class="alertbar stale"><div class="ab-head">
        <span class="dot"></span>Alert status unavailable</div>
        <div class="ab-foot">Could not read /api/alerts, so this page cannot say whether
          anything is wrong.</div></div>`;
      return;
    }
    holder.innerHTML = alertBarHTML(m);
    const st = alertState(m);
    const pill = document.createElement('span');
    pill.className = 'pill ' + st.level;
    pill.textContent = st.level === 'ok' ? 'All clear'
      : (st.level === 'stale' ? 'Unchecked' : st.label);
    const h1 = document.querySelector('header h1');
    if (h1 && !document.querySelector('header .pill')) h1.after(pill);
    // The tab title is the only cue visible when the page is not in focus.
    if (st.level !== 'ok' && st.level !== 'stale') {
      document.title = `(${(st.alerts || []).length}) ${document.title}`;
      const tab = document.querySelector('header .tab[href="/health"]');
      if (tab) tab.classList.add('alerting');
    }
  };
  fetchJSON('/api/alerts').then(m => paint(m, false)).catch(() => paint(null, true));
}
