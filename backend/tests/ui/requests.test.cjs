const { test } = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const html = fs.readFileSync(`${__dirname}/../../static/admin.html`, 'utf8');
function dashboard() {
    const elements = new Map();
    const element = id => {
        if (!elements.has(id)) elements.set(id, {value: '', textContent: '', innerHTML: '', open: false, showModal() {this.open = true;}});
        return elements.get(id);
    };
    const calls = [];
    const context = vm.createContext({ URLSearchParams, document: {getElementById: element},
        apiFetch: url => new Promise((resolve, reject) => calls.push({url, resolve, reject})),
        escapeHtml: String, escapeAttr: String,
    });
    vm.runInContext(html.slice(html.indexOf('        let requestsPage = 1;'), html.indexOf('        // ========== User Detail')), context);
    return {context, element, calls};
}
const result = (snapshot = 4) => ({requests: [], page: 1, total_pages: 2, snapshot});
test('Traffic leaves history untouched and does not fetch or replace rows', () => {
    const {context, element, calls} = dashboard();
    element('requests-ul').innerHTML = 'Reading an older request';
    for (let i = 0; i < 100; i++) context.prependNewRequest({id: String(i)});
    assert.equal(element('requests-ul').innerHTML, 'Reading an older request');
    assert.equal(calls.length, 0);
    assert.match(element('request-refresh').textContent, /Updates available/);
});
test('Pagination preserves snapshot and filters; stale responses cannot replace current results', async () => {
    const {context, element, calls} = dashboard();
    element('request-model').value = 'model one';
    element('request-origin').value = 'my app';
    context.refreshRequests();
    calls[0].resolve(result(42));
    await new Promise(setImmediate);
    const first = context.loadRequests(2);
    const url = new URL(calls[1].url, 'http://local');
    assert.equal(url.searchParams.get('snapshot'), '42');
    assert.equal(url.searchParams.get('model'), 'model one');
    assert.equal(url.searchParams.get('origin'), 'my app');
    context.refreshRequests();
    assert.equal(new URL(calls[2].url, 'http://local').searchParams.has('snapshot'), false);
    calls[2].resolve(result(99));
    await new Promise(setImmediate);
    calls[1].reject(new Error('stale failure'));
    await first;
    assert.equal(element('request-load-status').textContent, 'History loaded');
});
test('Inspector renders payloads as text and ignores a previous selection', async () => {
    const {context, element, calls} = dashboard();
    const first = context.inspectRequest('first');
    const second = context.inspectRequest('second');
    calls[1].resolve({request_body: '<script>bad()</script>', response_body: '{"answer":"hello"}'});
    await second;
    calls[0].resolve({request_body: 'old', response_body: 'old'});
    await first;
    assert.equal(element('request-body').textContent, '<script>bad()</script>');
    assert.match(element('response-body').textContent, /hello/);
    assert.equal(element('request-body').innerHTML, '');
});
test('Live event during a fetch remains visible as an available update', async () => {
    const {context, element, calls} = dashboard();
    const load = context.loadRequests();
    context.prependNewRequest({id: 'new'});
    calls[0].resolve(result());
    await load;
    assert.match(element('request-refresh').textContent, /Updates available/);
});
