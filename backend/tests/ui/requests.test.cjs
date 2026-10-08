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
        escapeHtml: s => s ? String(s).replace(/[&<>]/g, c => ({'&': '&amp;', '<': '&lt;', '>': '&gt;'}[c])) : '', escapeAttr: String,
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
function row(id) {
    const panels = [];
    const li = {dataset: {requestId: id}, querySelector: () => panels[0] || null, appendChild: p => panels.push(p)};
    const button = {textContent: 'Peek', closest: () => li};
    return {button, panels};
}
test('Peek escapes payloads and ignores a superseded load', async () => {
    const {context, calls} = dashboard();
    context.document.createElement = () => ({className: '', dataset: {}, innerHTML: '', classList: {
        hidden: false, add() {this.hidden = true;}, remove() {this.hidden = false;}, contains() {return this.hidden;},
    }});
    const {button, panels} = row('first');
    const first = context.peekRequest(button);
    await context.peekRequest(button); // hide
    const second = context.peekRequest(button); // reopen
    assert.equal(calls.length, 2);
    calls[1].resolve({request_body: JSON.stringify({messages: [{role: 'user', content: '<script>bad()</script>'}]}), response_body: '{"choices":[{"message":{"content":"hello"}}]}'});
    await second;
    calls[0].resolve({request_body: 'old', response_body: 'old'});
    await first;
    assert.match(panels[0].innerHTML, /&lt;script&gt;bad\(\)&lt;\/script&gt;/);
    assert.doesNotMatch(panels[0].innerHTML, /<script>/);
    assert.match(panels[0].innerHTML, /hello/);
    assert.doesNotMatch(panels[0].innerHTML, />old</);
});
test('Prompt extraction handles chat, multimodal and Responses inputs', () => {
    const {context} = dashboard();
    const chat = context.promptMessages(JSON.stringify({messages: [
        {role: 'system', content: 'be nice'},
        {role: 'user', content: [{type: 'text', text: 'look'}, {type: 'image_url', image_url: {url: 'x'}}]},
    ]}));
    assert.deepEqual(JSON.parse(JSON.stringify(chat.map(m => [m.role, m.text]))), [['system', 'be nice'], ['user', 'look\n[image_url]']]);
    const responses = context.promptMessages(JSON.stringify({instructions: 'sys', input: 'hi'}));
    assert.deepEqual([...responses.map(m => m.role)], ['system', 'user']);
    assert.equal(context.promptMessages('not json'), null);
});
test('Response extraction joins streamed deltas and keeps capture notes', () => {
    const {context} = dashboard();
    const chat = context.responseText([
        'data: {"choices":[{"delta":{"reasoning_content":"think"}}]}',
        'data: {"choices":[{"delta":{"content":"Hel"}}]}',
        'data: {"choices":[{"delta":{"content":"lo"}}]}',
        'data: [DONE]', '', '[Stream interrupted; partial output]',
    ].join('\n'));
    assert.equal(chat.text, 'Hello');
    assert.equal(chat.reasoning, 'think');
    assert.deepEqual([...chat.notes], ['[Stream interrupted; partial output]']);
    const responses = context.responseText('event: response.output_text.delta\ndata: {"type":"response.output_text.delta","delta":"Hi"}\n\n');
    assert.equal(responses.text, 'Hi');
    assert.equal(context.responseText('[stream]'), null);
    assert.equal(context.responseText('{"data":[1,2]}'), null);
});
test('Live event during a fetch remains visible as an available update', async () => {
    const {context, element, calls} = dashboard();
    const load = context.loadRequests();
    context.prependNewRequest({id: 'new'});
    calls[0].resolve(result());
    await load;
    assert.match(element('request-refresh').textContent, /Updates available/);
});
