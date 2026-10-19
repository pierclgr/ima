import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import test from 'node:test';

const page = readFileSync(new URL('../../examples/snake/index.html', import.meta.url), 'utf8');
const PNG = 'data:image/png;base64,';
let nextApp = 0;

/** Creates a canvas whose data URL records its current drawing commands. */
function makeCanvas(element) {
  element.width = element.height = 512;
  element.frame = [];
  element.captures = 0;
  const context = new Proxy({ canvas: element }, {
    get(target, name) {
      if (name in target) return target[name];
      return (...args) => {
        if (name === 'fillRect' || name === 'clearRect') element.frame = [];
        element.frame.push([name, ...args]);
      };
    },
  });
  element.getContext = () => context;
  element.toDataURL = (type) => {
    assert.equal(type, 'image/png');
    element.captures += 1;
    return PNG + Buffer.from(JSON.stringify(element.frame)).toString('base64');
  };
}

/** Loads the real app and renderer with DOM, canvas, clock and HTTP fakes. */
async function setup(t, context = {}) {
  const previousDocument = globalThis.document;
  const previousWindow = globalThis.window;
  t.after(() => {
    if (previousDocument === undefined) delete globalThis.document;
    else globalThis.document = previousDocument;
    if (previousWindow === undefined) delete globalThis.window;
    else globalThis.window = previousWindow;
  });
  const elements = new Map();
  const timers = new Map();
  const calls = [];
  let nextTimer = 1;

  function makeElement(id, value = '') {
    const element = {
      value, checked: false, events: {},
      addEventListener(name, callback) { this.events[name] = callback; },
      click() { this.events.click(); },
      removeAttribute(name) { delete this[name]; },
      insertAdjacentHTML(_position, html) {
        for (const match of html.matchAll(/<input[^>]*id="([^"]+)"([^>]*)>/g)) {
          makeElement(match[1]).checked = match[2].includes(' checked');
        }
      },
    };
    elements.set(id, element);
    return element;
  }
  for (const match of page.matchAll(/<[^>]+\bid="([^"]+)"[^>]*>/g)) {
    makeElement(match[1], match[0].match(/\bvalue="([^"]*)"/)?.[1] ?? '');
  }
  const board = elements.get('board');
  makeCanvas(board);
  makeCanvas(elements.get('overlay'));
  globalThis.document = { getElementById: (id) => elements.get(id) };
  globalThis.window = {
    setInterval(callback, ms) {
      const id = nextTimer++;
      timers.set(id, { callback, ms });
      return id;
    },
    clearInterval(id) { timers.delete(id); },
    fetch(url, options) {
      if (url.endsWith('/health')) return Promise.resolve({ ok: true });
      return new Promise((resolve) => calls.push({ body: JSON.parse(options.body), resolve }));
    },
  };
  await import(`../../examples/snake/js/app.js?test=${nextApp++}`);
  for (const [item, checked] of Object.entries(context)) {
    elements.get(`context-${item}`).checked = checked;
  }
  elements.get('reset').click();
  return {
    board, calls, elements,
    tick: () => [...timers.values()].find((timer) => timer.ms === 200).callback(),
  };
}

/** Resolves one API reply and lets the app finish handling it. */
async function respond(call) {
  call.resolve({
    ok: true, status: 200,
    text: async () => JSON.stringify({
      model: 'snake',
      answers: { direction: { choice: 'up', probabilities: { up: 0.7, down: 0.1, left: 0.1, right: 0.1 } } },
    }),
  });
  await new Promise((resolve) => setImmediate(resolve));
}

test('each screenshot matches the current rendered board and request facts', async (t) => {
  const { board, calls, elements, tick } = await setup(t, { head: true });
  elements.get('play').click();
  let previousImage = null;
  for (let tickNumber = 0; tickNumber < 3; tickNumber++) {
    const request = calls[tickNumber].body;
    const frame = JSON.parse(Buffer.from(request.files[0].slice(PNG.length), 'base64').toString());
    assert.deepEqual(frame, board.frame, `screenshot differs from board at tick ${tickNumber}`);
    assert.match(request.state, new RegExp(`The head is at column 4, row ${5 - tickNumber}\\.`));
    assert.notEqual(request.files[0], previousImage);
    previousImage = request.files[0];
    await respond(calls[tickNumber]);
    tick();
  }
});

test('the latest input preview updates on send and stays fixed while waiting', async (t) => {
  const { calls, elements, tick } = await setup(t, { board: true });
  const preview = elements.get('screenshot');
  const text = elements.get('board-text');
  assert.equal(preview.hidden, true);
  assert.equal(text.hidden, true);
  elements.get('play').click();
  const firstImage = calls[0].body.files[0];
  assert.equal(preview.hidden, false);
  assert.equal(preview.src, firstImage);
  assert.equal(text.hidden, false);
  assert.equal(text.textContent, calls[0].body.state.split('\n\n').at(-1));
  tick();
  tick();
  assert.equal(calls.length, 1);
  assert.equal(preview.src, firstImage);
  assert.equal(text.textContent, calls[0].body.state.split('\n\n').at(-1));
  await respond(calls[0]);
  tick();
  assert.equal(calls.length, 2);
  assert.equal(preview.src, calls[1].body.files[0]);
  assert.notEqual(preview.src, firstImage);
  assert.equal(text.textContent, calls[1].body.state.split('\n\n').at(-1));
});

test('reset clears the preview and an old reply cannot replace a new input', async (t) => {
  const { calls, elements, tick } = await setup(t, { board: true });
  elements.get('play').click();
  tick();
  elements.get('seed').value = '2';
  elements.get('reset').click();
  assert.equal(elements.get('screenshot').hidden, true);
  assert.equal(elements.get('screenshot').src, undefined);
  assert.equal(elements.get('board-text').hidden, true);
  assert.equal(elements.get('board-text').textContent, '');
  elements.get('play').click();
  const image = calls[1].body.files[0];
  assert.notEqual(image, calls[0].body.files[0]);
  await respond(calls[0]);
  assert.equal(elements.get('screenshot').src, image);
  assert.equal(elements.get('board-text').textContent, calls[1].body.state.split('\n\n').at(-1));
  assert.equal(elements.get('response-json').textContent, 'null');
});

test('text-only requests update the preview without capturing a screenshot', async (t) => {
  const { board, calls, elements } = await setup(t, { screenshot: false, board: true });
  elements.get('play').click();
  assert.equal(board.captures, 0);
  assert.equal(calls[0].body.files, undefined);
  assert.equal(elements.get('screenshot').hidden, true);
  assert.equal(elements.get('board-text').hidden, false);
  assert.equal(elements.get('board-text').textContent, calls[0].body.state.split('\n\n').at(-1));
});
