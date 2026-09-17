// Run with: node --test backend/tests/ui/lan_local.test.cjs
const { test } = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');

function dashboard() {
    const html = fs.readFileSync(path.join(__dirname, '../../static/admin.html'), 'utf8');
    const start = html.indexOf('        let lanLocalLoading = false;');
    const end = html.indexOf('        // Render dashboard stats from local state', start);
    assert.ok(start >= 0 && end > start);
    const elements = new Map();
    const requests = [];
    const element = id => {
        if (!elements.has(id)) elements.set(id, { value: '', disabled: false, textContent: '' });
        return elements.get(id);
    };
    const context = vm.createContext({
        document: { getElementById: element },
        apiFetch: (url, options = {}) => new Promise((resolve, reject) => {
            requests.push({ url, options, resolve, reject });
        }),
    });
    vm.runInContext(html.slice(start, end), context);
    return { context, requests, element };
}

for (const staleFailure of [false, true]) {
    test(`Disable during polling sends PUT and ignores stale ${staleFailure ? 'errors' : 'status'}`, async () => {
        const { context, requests, element } = dashboard();
        const poll = context.loadLanLocal();
        const update = context.updateLanLocal({ preventDefault() {} }, false);
        assert.equal(requests.length, 2);
        assert.equal(requests[1].options.method, 'PUT');
        assert.deepEqual(JSON.parse(requests[1].options.body), { enabled: false });
        requests[1].resolve({ enabled: false });
        await update;
        if (staleFailure) requests[0].reject(new Error('old poll failed'));
        else requests[0].resolve({ enabled: true, network: '192.168.1.0/24', expires_at: '2099-01-01' });
        await poll;
        assert.match(element('lan-local-status').textContent, /^OFF/);
        assert.equal(element('lan-local-error').textContent, '');
        assert.equal(element('lan-local-disable').disabled, true);
        const nextPoll = context.loadLanLocal();
        assert.equal(requests.length, 3);
        requests[2].resolve({ enabled: false });
        await nextPoll;
    });
}

test('An old poll cannot re-enable controls while a toggle update is pending', async () => {
    const { context, requests, element } = dashboard();
    const poll = context.loadLanLocal();
    const update = context.updateLanLocal({ preventDefault() {} }, false);
    requests[0].resolve({ enabled: true, network: '192.168.1.0/24', expires_at: '2099-01-01' });
    await poll;
    assert.equal(element('lan-local-enable').disabled, true);
    assert.equal(element('lan-local-disable').disabled, true);
    await context.loadLanLocal();
    assert.equal(requests.length, 2);
    requests[1].resolve({ enabled: false });
    await update;
    assert.equal(element('lan-local-enable').disabled, false);
});
