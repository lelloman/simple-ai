#!/usr/bin/env node
// Local dashboard preview with mock data: no backend, no OIDC, no runners.
//
//   node backend/dev/mock-dashboard.mjs [--port 8787]
//
// Serves static/admin.html as-is (re-read on every load) with a small injected shim
// that fakes the login token and the /admin/ws WebSocket. The page reloads itself
// whenever admin.html changes on disk. Mock state lives in memory and resets on restart.
import http from 'node:http';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const here = path.dirname(fileURLToPath(import.meta.url));
const ADMIN_HTML = path.join(here, '..', 'static', 'admin.html');
const portArg = process.argv.indexOf('--port');
const PORT = Number(portArg > 0 ? process.argv[portArg + 1] : process.env.PORT || 8787);

// ---------- deterministic fake data ----------

let seed = 42;
const rand = () => ((seed = (seed * 1103515245 + 12345) % 2 ** 31) / 2 ** 31);
const pick = list => list[Math.floor(rand() * list.length)];
const int = (lo, hi) => lo + Math.floor(rand() * (hi - lo + 1));
const uuid = () => 'xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx'.replace(/[xy]/g, c => {
    const r = Math.floor(rand() * 16);
    return (c === 'x' ? r : (r & 3) | 8).toString(16);
});
const iso = ms => new Date(ms).toISOString();
const now = Date.now();

const RUNNERS = [
    { id: 'rtx3090-box', name: 'RTX 3090 box', machine_type: 'gpu-server', models: ['qwen3-coder-30b', 'qwen-cyber-14b', 'nomic-embed-text'] },
    { id: 'halo-strix', name: 'Halo Strix', machine_type: 'halo', models: ['halogen-70b', 'qwen3-coder-30b', 'whisper-large-v3'] },
    { id: 'mini-cpu', name: 'Mini CPU', machine_type: 'cpu', models: ['nomic-embed-text', 'paddle-ocr'] },
];
const MODELS = ['qwen3-coder-30b', 'qwen-cyber-14b', 'halogen-70b', 'nomic-embed-text', 'whisper-large-v3', 'paddle-ocr'];
const USERS = [
    { id: uuid(), email: 'domenico@example.com' },
    { id: uuid(), email: 'ci-bot@example.com' },
    { id: uuid(), email: 'android-app@example.com' },
    { id: uuid(), email: null },
];
const KEYS = USERS.slice(0, 3).map((u, i) => ({
    id: uuid(), user_id: u.id, user_email: u.email, name: ['laptop', 'ci pipeline', 'android'][i],
    roles: i === 1 ? ['model:specific'] : [], created_at: iso(now - 86400e3 * (30 - i * 7)),
    last_used_at: iso(now - 3600e3 * (i + 1)), revoked: false, secret_available: i !== 2,
}));
const APPS = ['open-webui', 'android-assistant', 'ci-review', 'cli', null];
const AGENTS = ['python-httpx/0.27', 'okhttp/4.12', 'curl/8.5.0', 'Mozilla/5.0 (X11; Linux x86_64) Firefox/131.0'];

const SYSTEM_PROMPT = 'You are a careful senior engineer. Answer concisely, cite file paths, and prefer minimal diffs.\n'.repeat(12).trim();
const PROMPTS = [
    'Why does my Rust future never resolve when I hold a MutexGuard across .await?',
    'Summarise this changelog in three bullet points:\n- added router telemetry\n- fixed WOL retries\n- new admin dashboard',
    'Write a bash one-liner that finds the 10 largest files under /var/log.',
    'Translate to Italian: "The runner is waking up, please wait."',
    'Explain the difference between prompt tokens and completion tokens like I am five.',
];
const ANSWERS = [
    'Holding a `std::sync::MutexGuard` across `.await` makes the future `!Send` and can deadlock if another task on the same thread needs the lock. Drop the guard before awaiting, or switch to `tokio::sync::Mutex`.',
    '- Router now records telemetry events\n- Wake-on-LAN retries are fixed\n- The admin dashboard was redesigned',
    'find /var/log -type f -printf "%s %p\\n" | sort -nr | head -10',
    '"Il runner si sta svegliando, attendere prego."',
    'Prompt tokens are the words you give the robot; completion tokens are the words it gives back. You pay for both!',
];

function sse(chunks) {
    return chunks.map(c => `data: ${JSON.stringify(c)}\n\n`).join('') + 'data: [DONE]\n\n';
}
function chatStream(model, answer, reasoning) {
    const words = answer.match(/\S+\s*/g) || [answer];
    const chunks = [];
    if (reasoning) chunks.push({ model, choices: [{ index: 0, delta: { role: 'assistant', reasoning_content: reasoning } }] });
    for (const w of words) chunks.push({ model, choices: [{ index: 0, delta: { content: w } }] });
    chunks.push({ model, choices: [{ index: 0, delta: {}, finish_reason: 'stop' }] });
    return sse(chunks);
}

function makeRequest(i, ts) {
    const kind = pick(['chat', 'chat', 'chat-stream', 'chat-stream', 'responses', 'embed', 'ocr']);
    const user = pick(USERS);
    const key = rand() < 0.6 ? KEYS.find(k => k.user_id === user.id) : null;
    const runner = pick(RUNNERS);
    const p = int(0, PROMPTS.length - 1);
    let model = pick(['qwen3-coder-30b', 'qwen-cyber-14b', 'halogen-70b']);
    let pathName = '/v1/chat/completions';
    let requestBody, responseBody;
    const messages = [];
    if (rand() < 0.4) messages.push({ role: 'system', content: SYSTEM_PROMPT });
    if (rand() < 0.15) messages.push({ role: 'user', content: [{ type: 'text', text: 'What is in this screenshot?' }, { type: 'image_url', image_url: { url: 'data:image/png;base64,iVBORw0...' } }] });
    else messages.push({ role: 'user', content: PROMPTS[p] });

    if (kind === 'chat') {
        requestBody = JSON.stringify({ model, messages, temperature: 0.7 });
        responseBody = JSON.stringify({ id: 'chatcmpl-' + i, model, choices: [{ index: 0, message: { role: 'assistant', content: ANSWERS[p] }, finish_reason: 'stop' }], usage: { prompt_tokens: 40, completion_tokens: 60 } });
    } else if (kind === 'chat-stream') {
        requestBody = JSON.stringify({ model, messages, stream: true });
        responseBody = chatStream(model, ANSWERS[p], rand() < 0.4 ? 'The user wants a short answer. Let me think about the key point first…' : null);
        if (rand() < 0.15) responseBody = responseBody.slice(0, Math.floor(responseBody.length / 2)) + '\n[Stream interrupted; partial output]';
    } else if (kind === 'responses') {
        pathName = '/v1/responses';
        requestBody = JSON.stringify({ model, messages: [{ role: 'system', content: 'Be brief.' }, { role: 'user', content: PROMPTS[p] }], stream: true });
        responseBody = (ANSWERS[p].match(/\S+\s*/g) || []).map(d =>
            `event: response.output_text.delta\ndata: ${JSON.stringify({ type: 'response.output_text.delta', delta: d })}\n\n`).join('') +
            `event: response.completed\ndata: ${JSON.stringify({ type: 'response.completed' })}\n\n`;
    } else if (kind === 'embed') {
        pathName = '/v1/embeddings';
        model = 'nomic-embed-text';
        requestBody = JSON.stringify({ model, input: [PROMPTS[p]] });
        responseBody = JSON.stringify({ object: 'list', data: [{ index: 0, embedding: Array.from({ length: 8 }, () => +rand().toFixed(4)) }] });
    } else {
        pathName = '/v1/ocr';
        model = 'paddle-ocr';
        requestBody = JSON.stringify({ language: 'en', detect_layout: true });
        responseBody = JSON.stringify({ text: 'INVOICE #2291\nTotal: 41.20 EUR' });
    }
    if (i % 97 === 5) responseBody = '[stream]'; // legacy unrecorded stream
    const status = rand() < 0.08 ? pick([429, 500, 503]) : 200;
    return {
        summary: {
            id: uuid(), timestamp: iso(ts), user_id: user.id, user_email: user.email, request_path: pathName, model,
            client_ip: `192.168.1.${int(10, 60)}`, source_app: pick(APPS), auth_method: key ? 'api_key' : 'oidc',
            api_key_id: key?.id ?? null, api_key_name: key?.name ?? null, user_agent: pick(AGENTS),
            peer_ip: rand() < 0.3 ? '10.0.0.2' : null, proxy_request_id: rand() < 0.2 ? 'px-' + int(1000, 9999) : null,
            status, latency_ms: int(80, 9000), tokens_prompt: kind === 'ocr' ? null : int(20, 4000),
            tokens_completion: kind.startsWith('chat') || kind === 'responses' ? int(10, 900) : null,
            runner_id: runner.id, wol_sent: rand() < 0.05,
        },
        bodies: { request_body: requestBody, response_body: status === 200 ? responseBody : JSON.stringify({ error: { message: 'Runner unavailable', code: status } }) },
    };
}

let insertSeq = 0;
const history = []; // newest first; { seq, summary, bodies }
function addRequest(entry) {
    history.unshift({ seq: ++insertSeq, ...entry });
}
{
    let ts = now;
    const seedRows = [];
    for (let i = 0; i <= 260; i++) seedRows.push(makeRequest(i, (ts -= int(60e3, 900e3))));
    for (const row of seedRows.reverse()) addRequest(row); // oldest first, so history stays newest-first
}
// A couple of in-flight requests
for (const entry of history.slice(0, 2)) {
    Object.assign(entry.summary, { status: null, latency_ms: null, tokens_prompt: null, tokens_completion: null });
    entry.bodies.response_body = null;
}
const cancellable = new Set(history.slice(0, 2).map(e => e.summary.id));

const lanLocal = { enabled: false, network: null, expires_at: null, remaining_seconds: 0 };

function routerRunner(r, i) {
    return {
        runner_id: r.id, name: r.name, machine_type: r.machine_type, is_online: i !== 2, health: i === 2 ? 'offline' : 'healthy',
        scheduler_state: ['ready', 'loading', 'asleep'][i], target_model: i === 1 ? 'halogen-70b' : null,
        active_requests: [2, 0, 0][i], loaded_models: i === 0 ? [r.models[0]] : [], available_models: r.models,
        protected_classes: i === 1 ? ['code:smart'] : [], updated_at: iso(now - i * 60e3),
    };
}
const EVENT_KINDS = ['request_received', 'scheduler_planned', 'request_dispatched', 'request_completed', 'model_load_started', 'model_load_ready', 'wake_started', 'wake_succeeded', 'request_failed', 'request_cancelled', 'capacity_wake_started', 'decision_ready_selected'];
function routerEvent(ts = Date.now()) {
    const kind = pick(EVENT_KINDS);
    const model = pick(MODELS);
    const runner = pick(RUNNERS);
    const messages = {
        request_received: `Request received for ${model}`, scheduler_planned: `Planned ${model} on ${runner.name}`,
        request_dispatched: `Dispatched to ${runner.name}`, request_completed: `Completed in ${int(200, 8000)}ms`,
        model_load_started: `Loading ${model} on ${runner.name}`, model_load_ready: `${model} ready on ${runner.name}`,
        wake_started: `Sending Wake-on-LAN to ${runner.name}`, wake_succeeded: `${runner.name} is awake`,
        request_failed: 'Runner returned 503', request_cancelled: 'Cancelled by admin',
        capacity_wake_started: `Queue saturated, waking ${runner.name}`, decision_ready_selected: `Selected ready runner ${runner.name}`,
    };
    return { timestamp: iso(ts), kind, message: messages[kind], request_id: uuid(), runner_id: runner.id, model };
}
const routerState = {
    runners: RUNNERS.map(routerRunner),
    queues: { 'qwen3-coder-30b': 1, 'halogen-70b': 3 },
    recent_events: Array.from({ length: 40 }, (_, i) => routerEvent(now - i * 45e3)),
    affinity: { enabled: true, bindings: 17, max_entries: 1024, ttl_secs: 900, max_extra_wait_ms: 1500, decisions: { hit: 312, miss: 88, bypass: 12 }, evictions: { ttl: 40, capacity: 2 } },
};
const batchQueue = { enabled: true, timeout_ms: 50, saturation_timeout_ms: 500, min_batch_size: 2, queues: { 'qwen3-coder-30b': { pending: 2, oldest_age_ms: 340 }, 'halogen-70b': { pending: 1, oldest_age_ms: 1200 } } };

// Pressure: hosts random-walk a waiting count; groups take their best serving host.
const PRESSURE_GROUPS = {
    fast: ['rtx3090-box'], big: ['halo-strix'], embeddings: ['rtx3090-box', 'mini-cpu'],
    tts: ['rtx3090-box'], decisions: ['halo-strix', 'rtx3090-box'], extraction: ['halo-strix', 'mini-cpu'],
};
const hostWaiting = { 'rtx3090-box': 0, 'halo-strix': 2, 'mini-cpu': 0 };
const ORDER = ['green', 'orange', 'red'];
function pressure() {
    const hosts = RUNNERS.map((r, i) => {
        const waiting = hostWaiting[r.id];
        const level = waiting === 0 ? 'green' : waiting < 3 ? 'orange' : 'red';
        const reason = waiting ? `${waiting} waiting for ${waiting * 4}s` : '';
        return { runner_id: r.id, name: r.name, online: i !== 2, level, reason, lanes: [] };
    });
    const groups = Object.fromEntries(Object.entries(PRESSURE_GROUPS).map(([name, ids]) => {
        const serving = hosts.filter(h => ids.includes(h.runner_id));
        const best = serving.reduce((a, b) => ORDER.indexOf(b.level) < ORDER.indexOf(a.level) ? b : a);
        return [name, {
            level: best.level, available: true, cold: serving.every(h => !h.online), hosts: ids,
            reason: best.reason ? `${best.name}: ${best.reason}` : '',
            ...(best.level !== 'green' ? { retry_after_secs: best.level === 'orange' ? 15 : 60 } : {}),
        }];
    }));
    const level = hosts.map(h => h.level).reduce((a, b) => ORDER.indexOf(b) > ORDER.indexOf(a) ? b : a, 'green');
    return { level, updated_at: iso(Date.now()), groups, hosts };
}

function snapshot() {
    const runners = RUNNERS.map((r, i) => ({
        id: r.id, name: r.name, machine_type: r.machine_type, health: i === 2 ? 'offline' : 'healthy',
        loaded_models: i === 0 ? [r.models[0]] : [], available_models: r.models, connected_at: iso(now - 86400e3),
        last_heartbeat: iso(Date.now() - 5e3), http_base_url: `http://${r.id}.lan:8080`, mac_address: `aa:bb:cc:00:00:0${i}`, is_online: i !== 2,
    }));
    const models = MODELS.map((m, i) => ({
        id: m, name: m, size_bytes: [18e9, 9e9, 42e9, 3e8, 3e9, 2e8][i], parameter_count: [30e9, 14e9, 70e9, 137e6, 1.5e9, 1e8][i],
        context_length: [32768, 16384, 65536, 8192, undefined, undefined][i], quantization: ['Q4_K_M', 'Q5_K_M', 'Q4_K_M', 'F16', undefined, undefined][i],
        reasoning: i === 2 ? { supported_efforts: ['low', 'medium', 'high'], supports_thinking_budget: true, default_effort: 'medium' } : undefined,
        loaded: i === 0, runners: i === 0 ? ['rtx3090-box'] : [], available_on: RUNNERS.filter(r => r.models.includes(m)).map(r => r.id),
    }));
    const done = history.filter(e => e.summary.status !== null);
    const tp = done.reduce((a, e) => a + (e.summary.tokens_prompt || 0), 0);
    const tc = done.reduce((a, e) => a + (e.summary.tokens_completion || 0), 0);
    return {
        type: 'state_snapshot', runners, models, batch_queue: batchQueue, router_state: routerState, pressure: pressure(),
        stats: { total_users: USERS.length, total_requests: history.length, requests_24h: history.filter(e => Date.parse(e.summary.timestamp) > Date.now() - 86400e3).length, total_tokens: tp + tc, tokens_prompt: tp, tokens_completion: tc },
    };
}

// ---------- browser shim: fake token + fake WebSocket ----------

const fakeJwt = ['{"alg":"none"}', JSON.stringify({ sub: 'mock-admin', email: 'mock-admin@localhost', exp: Math.floor(now / 1000) + 10 * 365 * 86400 })]
    .map(s => Buffer.from(s).toString('base64')).join('.') + '.mock';

const SHIM = `<script>
(() => {
    localStorage.setItem('access_token', ${JSON.stringify(fakeJwt)});
    localStorage.setItem('token_expires_at', String(Date.now() + 10 * 365 * 86400e3));
    class MockSocket {
        static CONNECTING = 0; static OPEN = 1; static CLOSING = 2; static CLOSED = 3;
        constructor() {
            this.readyState = 0;
            setTimeout(() => { this.readyState = 1; this.onopen?.({}); }, 50);
        }
        emit(msg) { if (this.readyState === 1) this.onmessage?.({ data: JSON.stringify(msg) }); }
        async send(raw) {
            if (JSON.parse(raw).type !== 'auth') return;
            this.emit({ type: 'auth_ok' });
            this.emit(await (await fetch('/mock/state')).json());
            clearInterval(this.timer);
            this.timer = setInterval(async () => {
                for (const msg of await (await fetch('/mock/tick', { method: 'POST' })).json()) this.emit(msg);
            }, 4000);
        }
        close(code = 1000) { clearInterval(this.timer); this.readyState = 3; this.onclose?.({ code, reason: '' }); }
    }
    window.WebSocket = MockSocket;
    new EventSource('/mock/reload').onmessage = () => location.reload();
    console.info('%cMock dashboard: fake auth, fake WebSocket, in-memory data', 'color:#7c3aed');
})();
</script>`;

// ---------- live ticks ----------

function tick() {
    const out = [];
    const event = routerEvent();
    routerState.recent_events = [event, ...routerState.recent_events].slice(0, 100);
    out.push({ type: 'router_event', event });
    for (const q of Object.values(batchQueue.queues)) {
        q.pending = Math.max(0, q.pending + int(-1, 1));
        q.oldest_age_ms = q.pending ? int(50, 2500) : null;
    }
    out.push({ type: 'batch_queue_updated', batch_queue: batchQueue });
    if (rand() < 0.4) {
        const id = pick(['rtx3090-box', 'halo-strix']);
        hostWaiting[id] = Math.max(0, Math.min(4, hostWaiting[id] + int(-1, 1)));
        out.push({ type: 'pressure_updated', pressure: pressure() });
    }
    if (rand() < 0.5) {
        const r = pick(routerState.runners);
        if (r.is_online) r.active_requests = Math.max(0, r.active_requests + int(-1, 2));
        if (r.runner_id === 'halo-strix' && rand() < 0.3) {
            r.scheduler_state = r.scheduler_state === 'loading' ? 'ready' : 'loading';
            r.loaded_models = r.scheduler_state === 'ready' ? [r.target_model || 'halogen-70b'] : [];
        }
        r.updated_at = iso(Date.now());
        out.push({ type: 'router_state_updated', router_state: routerState });
    }
    if (rand() < 0.6) {
        const entry = makeRequest(insertSeq, Date.now());
        addRequest(entry);
        out.push({ type: 'new_request', ...entry.summary });
        out.push({ type: 'stats_updated', stats: snapshot().stats });
    }
    return out;
}

// ---------- HTTP ----------

function json(res, body, status = 200) {
    res.writeHead(status, { 'Content-Type': 'application/json' });
    res.end(JSON.stringify(body));
}
async function readBody(req) {
    let raw = '';
    for await (const chunk of req) raw += chunk;
    try { return JSON.parse(raw || '{}'); } catch { return {}; }
}
const reloadClients = new Set();
let reloadTimer;
fs.watch(path.dirname(ADMIN_HTML), (_, file) => {
    if (file !== path.basename(ADMIN_HTML)) return;
    clearTimeout(reloadTimer);
    reloadTimer = setTimeout(() => { for (const c of reloadClients) c.write('data: reload\n\n'); }, 100);
});

function listRequests(q) {
    const page = Math.max(1, Number(q.get('page')) || 1);
    const perPage = Math.min(100, Math.max(1, Number(q.get('per_page')) || 50));
    const snap = q.has('snapshot') ? Number(q.get('snapshot')) : insertSeq;
    const model = q.get('model')?.toLowerCase();
    const origin = q.get('origin')?.toLowerCase();
    const since = q.get('since') ? Date.parse(q.get('since')) : null;
    const until = q.get('until') ? Date.parse(q.get('until')) : null;
    const failedOnly = q.get('status') === 'failed';
    const rows = history.filter(({ seq, summary: s }) => seq <= snap
        && (!model || s.model?.toLowerCase().includes(model))
        && (!origin || [s.source_app, s.client_ip, s.api_key_name, s.user_agent].some(v => v?.toLowerCase().includes(origin)))
        && (since === null || Date.parse(s.timestamp) >= since)
        && (until === null || Date.parse(s.timestamp) <= until)
        && (!failedOnly || s.status >= 400));
    return {
        requests: rows.slice((page - 1) * perPage, page * perPage).map(e => e.summary),
        cancellable_request_ids: [...cancellable], page, per_page: perPage,
        total_pages: Math.ceil(rows.length / perPage), snapshot: snap,
    };
}

function summary(hours) {
    const since = Date.now() - hours * 3600e3;
    const rows = history.map(e => e.summary).filter(s => Date.parse(s.timestamp) >= since);
    const completed = rows.filter(s => s.status !== null);
    const failed = completed.filter(s => s.status >= 400);
    const latencies = completed.filter(s => s.status < 400).map(s => s.latency_ms).sort((a, b) => a - b);
    const pct = p => latencies.length ? latencies[Math.max(1, Math.ceil(latencies.length * p / 100)) - 1] : null;
    const top = (keyOf) => {
        const groups = new Map();
        for (const s of rows) {
            const [kind, label] = keyOf(s);
            const g = groups.get(kind + label) || { kind, label, requests: 0, failed: 0, tokens: 0 };
            g.requests++; g.failed += s.status >= 400 ? 1 : 0; g.tokens += (s.tokens_prompt || 0) + (s.tokens_completion || 0);
            groups.set(kind + label, g);
        }
        return [...groups.values()].sort((a, b) => b.requests - a.requests || a.label.localeCompare(b.label)).slice(0, 5);
    };
    return {
        since: iso(since), hours, requests: rows.length, completed: completed.length, failed: failed.length,
        tokens_prompt: completed.reduce((a, s) => a + (s.tokens_prompt || 0), 0),
        tokens_completion: completed.reduce((a, s) => a + (s.tokens_completion || 0), 0),
        latency_p50_ms: pct(50), latency_p95_ms: pct(95),
        top_models: top(s => ['model', s.model || s.request_path]),
        top_origins: top(s => s.source_app ? ['app', s.source_app] : s.api_key_name ? ['key', s.api_key_name] : s.user_email ? ['user', s.user_email] : ['ip', s.client_ip]),
        recent_failures: failed.slice(0, 6),
    };
}

async function route(req, res, url) {
    const p = url.pathname;
    const m = req.method;
    let match;
    if (m === 'GET' && (p === '/' || p === '/admin-ui' || p === '/admin-ui/')) {
        const html = fs.readFileSync(ADMIN_HTML, 'utf8').replace('<head>', `<head>${SHIM}`);
        res.writeHead(200, { 'Content-Type': 'text/html; charset=utf-8', 'Cache-Control': 'no-store' });
        return res.end(html);
    }
    if (p === '/mock/reload') {
        res.writeHead(200, { 'Content-Type': 'text/event-stream', 'Cache-Control': 'no-store' });
        res.write(': connected\n\n');
        reloadClients.add(res);
        return req.on('close', () => reloadClients.delete(res));
    }
    if (p === '/mock/state') return json(res, snapshot());
    if (p === '/mock/tick') return json(res, tick());

    if (p === '/admin/api/lan-local') {
        if (m === 'PUT') {
            const b = await readBody(req);
            Object.assign(lanLocal, b.enabled
                ? { enabled: true, network: b.network, expires_at: iso(Date.now() + b.duration_seconds * 1e3), remaining_seconds: b.duration_seconds }
                : { enabled: false, network: null, expires_at: null, remaining_seconds: 0 });
        } else if (lanLocal.enabled) {
            lanLocal.remaining_seconds = Math.max(0, Math.round((Date.parse(lanLocal.expires_at) - Date.now()) / 1e3));
        }
        return json(res, lanLocal);
    }
    if (p === '/admin/api/model-speeds') {
        const metrics = RUNNERS.flatMap(r => r.models.filter(x => !['nomic-embed-text', 'paddle-ocr', 'whisper-large-v3'].includes(x)).flatMap(model => [4096, 16384, 32768].map(ctx => {
            const pr = 300 + rand() * 1500, cr = 15 + rand() * 60;
            return { resolved_model: model, runner_id: r.id, context_window: ctx, sample_count: int(3, 400), prompt_tokens_total: int(1e4, 1e6), completion_tokens_total: int(1e3, 2e5),
                avg_prompt_tokens_per_sec: pr, avg_completion_tokens_per_sec: cr, prompt_tps_min: pr * 0.7, prompt_tps_max: pr * 1.3, completion_tps_min: cr * 0.8, completion_tps_max: cr * 1.2, last_updated_at: iso(Date.now() - int(1, 600) * 60e3) };
        })));
        return json(res, { metrics, total: metrics.length });
    }
    if (p === '/admin/api/users') {
        const users = USERS.map((u, i) => ({ id: u.id, email: u.email, created_at: iso(now - 86400e3 * (60 - i)), last_seen_at: iso(now - 3600e3 * i), is_enabled: i !== 3, request_count: history.filter(e => e.summary.user_id === u.id).length }));
        return json(res, { users, total: users.length });
    }
    if (p === '/v1/pressure') {
        const { level, updated_at, groups } = pressure();
        return json(res, { level, updated_at, groups: Object.fromEntries(Object.entries(groups).map(([k, g]) => [k, { level: g.level, available: g.available, cold: g.cold, ...(g.retry_after_secs ? { retry_after_secs: g.retry_after_secs } : {}) }])) });
    }
    if (p === '/admin/api/summary') return json(res, summary(Math.min(720, Math.max(1, Number(url.searchParams.get('hours')) || 24))));
    if (p === '/admin/api/requests' && m === 'GET') return json(res, listRequests(url.searchParams));
    if ((match = p.match(/^\/admin\/api\/requests\/([^/]+)\/cancel$/)) && m === 'POST') {
        const entry = history.find(e => e.summary.id === match[1]);
        if (!entry || !cancellable.delete(match[1])) return json(res, { error: 'not cancellable' }, 404);
        Object.assign(entry.summary, { status: 499, latency_ms: 1234 });
        return json(res, { cancelled: true });
    }
    if ((match = p.match(/^\/admin\/api\/requests\/([^/]+)$/)) && m === 'GET') {
        const entry = history.find(e => e.summary.id === decodeURIComponent(match[1]));
        return entry ? json(res, entry.bodies) : json(res, { error: 'not found' }, 404);
    }
    if (p === '/admin/api/keys' && m === 'GET') return json(res, { keys: KEYS, total: KEYS.length });
    if (p === '/admin/api/keys' && m === 'POST') {
        const b = await readBody(req);
        const user = USERS.find(u => u.id === b.user_id);
        const key = { id: uuid(), user_id: b.user_id, user_email: user?.email ?? null, name: b.name, roles: b.roles || [], created_at: iso(Date.now()), last_used_at: null, revoked: false, secret_available: true };
        KEYS.push(key);
        return json(res, { ...key, secret: 'sk-mock-' + key.id.replace(/-/g, '') });
    }
    if ((match = p.match(/^\/admin\/api\/keys\/([^/]+)\/secret$/))) return json(res, { secret: 'sk-mock-' + match[1].replace(/-/g, '') });
    if ((match = p.match(/^\/admin\/api\/keys\/([^/]+)$/))) {
        const key = KEYS.find(k => k.id === decodeURIComponent(match[1]));
        if (!key) return json(res, { error: 'not found' }, 404);
        if (m === 'DELETE') key.revoked = true;
        if (m === 'PATCH') Object.assign(key, await readBody(req));
        return json(res, key);
    }
    if ((match = p.match(/^\/admin\/runners\/([^/]+)\/(wake|load-model|unload-model)$/)) && m === 'POST') {
        return json(res, { success: true, message: `Mock ${match[2]} sent to ${match[1]}` });
    }
    if (p === '/v1/models') {
        return json(res, { object: 'list', data: MODELS.map(id => ({ id, object: 'model', owned_by: 'local' })).concat([{ id: 'code:smart', object: 'model', owned_by: 'class' }]) });
    }
    if (p === '/v1/chat/completions' && m === 'POST') {
        const b = await readBody(req);
        const last = [...(b.messages || [])].reverse().find(x => x.role === 'user');
        const answer = `(mock ${b.model || 'model'}) You said: ${typeof last?.content === 'string' ? last.content : '[multimodal input]'}`;
        if (b.stream) {
            res.writeHead(200, { 'Content-Type': 'text/event-stream' });
            for (const chunk of chatStream(b.model, answer).split('\n\n').filter(Boolean)) {
                res.write(chunk + '\n\n');
                await new Promise(r => setTimeout(r, 40));
            }
            return res.end();
        }
        return json(res, { id: 'chatcmpl-mock', model: b.model, choices: [{ index: 0, message: { role: 'assistant', content: answer }, finish_reason: 'stop' }], usage: { prompt_tokens: 12, completion_tokens: 20, total_tokens: 32 } });
    }
    if (p === '/v1/decisions' && m === 'POST') {
        const b = await readBody(req);
        await new Promise(r => setTimeout(r, 1500));
        const answers = (b.questions || []).map(q => {
            if (q.type === 'boolean') { const y = rand(); return { id: q.id, type: 'boolean', value: y > 0.5, probabilities: { true: y, false: 1 - y } }; }
            if (q.type === 'choice') {
                const w = q.options.map(() => rand()); const s = w.reduce((a, x) => a + x, 0);
                const probabilities = Object.fromEntries(q.options.map((o, i) => [o.id, w[i] / s]));
                return { id: q.id, type: 'choice', selected: q.options[w.indexOf(Math.max(...w))].id, probabilities };
            }
            const w = q.levels.map(() => rand()); const s = w.reduce((a, x) => a + x, 0);
            const probabilities = Object.fromEntries(w.map((x, i) => [String(i), x / s]));
            return { id: q.id, type: q.type, value: w.reduce((a, x, i) => a + i * x / s, 0), probabilities };
        });
        return json(res, { model: 'jev-mock', answers, timing: { total_ms: 1500 }, usage: { prompt_tokens: 321 } });
    }
    json(res, { error: `mock: no handler for ${m} ${p}` }, 404);
}

http.createServer((req, res) => {
    const url = new URL(req.url, `http://${req.headers.host}`);
    route(req, res, url).catch(error => {
        console.error(error);
        if (!res.headersSent) json(res, { error: String(error) }, 500);
    });
}).listen(PORT, () => {
    console.log(`Mock dashboard on http://localhost:${PORT}/admin-ui  (edits to static/admin.html auto-reload)`);
});
