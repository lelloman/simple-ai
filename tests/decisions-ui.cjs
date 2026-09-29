// Run with PLAYWRIGHT_MODULE pointing at an installed Playwright package if needed.
const {chromium} = require(process.env.PLAYWRIGHT_MODULE || 'playwright');
const fs = require('node:fs');
const assert = require('node:assert/strict');
(async () => {
    const browser = await chromium.launch({headless: true});
    try {
        const page = await browser.newPage();
        const errors = [];
        page.on('pageerror', error => errors.push(error.message));
        // Keep the actual page functions and auth-aware fetch; bypass only OIDC startup/CDNs.
        const html = fs.readFileSync('backend/static/admin.html', 'utf8')
            .replace(/<script src="[^"]+"><\/script>/g, '')
            .replace('<head>', '<head><script>window.tailwind={};</script><style>.hidden{display:none}</style>')
            .replace('        init();', '        // OIDC initialization supplied by test.');
        await page.route('http://simple-ai.test/admin', route => route.fulfill({contentType:'text/html', body:html}));
        let calls = 0, lastRequest, failure = false;
        await page.route('**/v1/decisions', async route => {
            calls++;
            assert.equal(route.request().headers().authorization, 'Bearer browser-test-token');
            lastRequest = route.request().postDataJSON();
            await new Promise(resolve => setTimeout(resolve, 100));
            if (failure) return route.fulfill({status:429, body:JSON.stringify({error:{message:'Decision queue full'}})});
            const answers = lastRequest.questions.map(q => {
                const base = {id:q.id, type:q.type};
                if (q.type === 'boolean') return {...base, value:true, probabilities:{false:0.1,true:0.9}};
                if (q.type === 'choice') return {...base, selected:q.options[0].id, probabilities:Object.fromEntries(q.options.map((o,i)=>[o.id,i===0 ? 0.9 : 0.1]))};
                return {...base,value:3,levels:q.levels,probabilities:{0:0,1:0,2:0,3:1,4:0,5:0}};
            });
            await route.fulfill({contentType:'application/json', body:JSON.stringify({model:'autotrust/JEV-9B',answers,timing:{total_ms:300},usage:{prompt_tokens:100}})});
        });
        await page.goto('http://simple-ai.test/admin#decisions');
        await page.evaluate(() => {
            accessToken='browser-test-token'; getValidAccessToken=async()=>accessToken;
            document.getElementById('main-app').classList.remove('hidden');
            showPage(getPageFromHash());
        });
        assert.equal(await page.locator('#page-title').textContent(), 'Decisions');
        assert.equal(await page.locator('#decision-questions > article').count(), 1);
        await page.getByRole('button', {name:'Load example',exact:true}).click();
        assert.equal(await page.locator('#decision-questions > article').count(), 3);
        await page.locator('#decision-format').selectOption('json');
        await page.locator('#decision-state').fill('{bad');
        await page.locator('#decision-run').click();
        assert.match(await page.locator('#decision-error').textContent(), /not valid JSON/);
        assert.equal(calls, 0);
        await page.locator('#decision-state').fill('{"message":"refund"}');
        await page.locator('[data-options]').fill('Only one');
        await page.locator('#decision-run').click();
        assert.match(await page.locator('#decision-error').textContent(), /2–16/);
        await page.locator('[data-options]').fill('<img src=x onerror=alert(1)>\nTechnical support');
        await page.evaluate(() => { runDecisions(); runDecisions(); });
        await page.waitForFunction(() => !decisionBusy);
        assert.equal(calls, 1);
        assert.equal(lastRequest.model, 'class:semantic_decisions');
        assert.deepEqual(lastRequest.state, {message:'refund'});
        assert.equal(lastRequest.questions[2].levels.length, 6);
        assert.equal(await page.locator('#decision-results > article').count(), 3);
        assert.equal(await page.locator('#decision-results img').count(), 0);
        assert.match(await page.locator('#decision-results').textContent(), /<img src=x onerror=alert\(1\)>/);
        assert.match(await page.locator('#decision-results').textContent(), /3.00 \/ 5/);
        assert.equal(await page.locator('#decision-copy').isDisabled(), false);
        await page.evaluate(() => { showPage('chat'); showPage('decisions'); renderCurrentPage(); });
        assert.equal(await page.locator('#decision-state').inputValue(), '{"message":"refund"}');
        // Switching types preserves option drafts.
        const type = page.locator('[data-type]').nth(1);
        await type.selectOption('boolean'); await type.selectOption('choice');
        assert.match(await page.locator('[data-options]').inputValue(), /Technical support/);
        failure = true;
        await page.locator('#decision-run').click();
        await page.waitForFunction(() => !decisionBusy);
        assert.match(await page.locator('#decision-error').textContent(), /Server busy.*Decision queue full/);
        assert.equal(await page.locator('#decision-fields').isDisabled(), false);
        assert.equal(await page.locator('#decision-copy').isDisabled(), true);
        assert.equal(await page.locator('#decision-state').inputValue(), '{"message":"refund"}');
        assert.deepEqual(errors, []);
        console.log('PASS: navigation, mixed questions, JSON/choice validation, auth header, duplicate submission, probabilities, safe text rendering, draft preservation, and retry after HTTP 429');
    } finally { await browser.close(); }
})().catch(error => {console.error(error); process.exitCode=1;});
