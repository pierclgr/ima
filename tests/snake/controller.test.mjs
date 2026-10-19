import assert from 'node:assert/strict';
import { test } from 'node:test';

import { DemoController } from '../../examples/snake/js/controller.js';
import { toCsv } from '../../examples/snake/js/metrics.js';
import { CONTEXT_ITEMS, DEFAULT_CONTEXT } from '../../examples/snake/js/request.js';

const NOTHING = Object.fromEntries(CONTEXT_ITEMS.map((item) => [item, false]));
// the facts and the safety net, on top of the screenshot-only default
const FACTS = { ...DEFAULT_CONTEXT, head: true, food: true, direction: true, cells: true, safety: true };

const TICK = 200;

/** Fake monotonic clock with interval timers. */
class FakeClock {
  constructor() {
    this.time = 0;
    this.timers = new Map();
    this.nextId = 1;
  }

  now() {
    return this.time;
  }

  setInterval(callback, ms) {
    const id = this.nextId++;
    this.timers.set(id, { callback, ms, due: this.time + ms });
    return id;
  }

  clearInterval(id) {
    this.timers.delete(id);
  }

  /** Runs the timers due in the next `ms` milliseconds. */
  advance(ms) {
    const end = this.time + ms;
    for (;;) {
      const due = [...this.timers.values()].filter((timer) => timer.due <= end);
      if (!due.length) break;
      const timer = due.reduce((a, b) => (a.due <= b.due ? a : b));
      this.time = timer.due;
      timer.due += timer.ms;
      timer.callback();
    }
    this.time = end;
  }
}

/** Fake fetch whose replies the test resolves one by one. */
function fakeFetch() {
  const calls = [];
  const fetch = (url, options) =>
    new Promise((resolve) => calls.push({ url, body: JSON.parse(options.body), resolve }));
  return { fetch, calls };
}

/** @returns {object} a fetch reply with a direction answer */
function reply(direction, status = 200) {
  const body = {
    model: 'snake',
    answers: {
      direction: {
        type: 'choice',
        choice: direction,
        probabilities: { up: 0.1, down: 0.1, left: 0.1, right: 0.1, [direction]: 0.7 },
      },
    },
  };
  return { ok: status === 200, status, text: async () => JSON.stringify(body) };
}

const flush = () => new Promise((resolve) => setImmediate(resolve));

/**
 * @param {object} context items sent to the model
 * @returns {object} controller with a seeded game, food out of the way
 */
function setup(context = DEFAULT_CONTEXT) {
  const clock = new FakeClock();
  const { fetch, calls } = fakeFetch();
  const controller = new DemoController({
    capture: () => 'data:image/png;base64,AAAA',
    fetch,
    clock,
    onChange: () => {},
  });
  controller.reset(TICK, 1, context);
  controller.game.food = { x: 0, y: 0 };
  return { clock, calls, controller };
}

test('a reply before a tick is applied at that tick', async () => {
  const { clock, calls, controller } = setup();
  controller.play();
  assert.equal(calls.length, 1);
  assert.equal(calls[0].url, 'http://127.0.0.1:8080/v1/systemone');
  clock.advance(50);
  calls[0].resolve(reply('up'));
  await flush();
  assert.equal(controller.metrics.latencies[0], 50);
  clock.advance(150);
  assert.equal(controller.game.heading, 'up');
  assert.deepEqual(controller.log[0], {
    captureTick: 0, applyTick: 1, age: 1, ageMs: 200, direction: 'up', latency: 50, code: false,
  });
  assert.equal(controller.metrics.late, 0);
});

test('a late reply is applied at the next tick with its age', async () => {
  const { clock, calls, controller } = setup();
  controller.play();
  clock.advance(2 * TICK + 50);
  assert.equal(controller.game.tick, 2);
  assert.equal(controller.game.heading, 'right');
  calls[0].resolve(reply('down'));
  await flush();
  clock.advance(TICK - 50);
  assert.equal(controller.game.heading, 'down');
  assert.equal(controller.log[0].age, 3);
  assert.equal(controller.log[0].ageMs, 3 * TICK);
  assert.equal(controller.metrics.late, 1);
});

test('only one request is in flight and one answer waits', async () => {
  const { clock, calls, controller } = setup();
  controller.play();
  clock.advance(3 * TICK);
  assert.equal(calls.length, 1);
  calls[0].resolve(reply('up'));
  await flush();
  assert.equal(calls.length, 1);
  clock.advance(TICK);
  assert.equal(calls.length, 2);
  assert.equal(controller.metrics.requests, 2);
});

test('failed and malformed replies keep the heading and the clock', async () => {
  const { clock, calls, controller } = setup();
  controller.play();
  calls[0].resolve(reply('up', 500));
  await flush();
  assert.match(controller.lastError, /HTTP 500/);
  clock.advance(TICK);
  assert.equal(controller.game.heading, 'right');
  assert.equal(calls.length, 2);
  calls[1].resolve({ ok: true, status: 200, text: async () => '{"answers": {}}' });
  await flush();
  assert.match(controller.lastError, /invalid response/);
  clock.advance(TICK);
  assert.equal(controller.game.tick, 2);
  assert.equal(controller.metrics.failures, 2);
  assert.equal(controller.log.length, 0);
});

test('pause ignores the pending reply and resume sends a new one', async () => {
  const { clock, calls, controller } = setup();
  controller.play();
  controller.pause();
  calls[0].resolve(reply('up'));
  await flush();
  assert.equal(controller.pending, null);
  clock.advance(5 * TICK);
  assert.equal(controller.game.tick, 0);
  controller.play();
  assert.equal(calls.length, 2);
  clock.advance(TICK);
  assert.equal(controller.game.heading, 'right');
});

test('a reply from before a reset does not affect the new game', async () => {
  const { clock, calls, controller } = setup();
  controller.play();
  clock.advance(TICK);
  controller.reset(TICK, 1, DEFAULT_CONTEXT);
  calls[0].resolve(reply('down'));
  await flush();
  assert.equal(controller.pending, null);
  assert.equal(controller.game.tick, 0);
  assert.equal(controller.running, false);
  clock.advance(TICK);
  assert.equal(controller.game.tick, 0);
});

test('the direction back is not sent and is rejected as an answer', async () => {
  const { clock, calls, controller } = setup(FACTS);
  controller.play();
  assert.deepEqual(Object.keys(calls[0].body.questions.direction.criteria), ['up', 'down', 'right']);
  assert.match(calls[0].body.state, /The head is at column 4, row 5\. The food is at column 0, row 0\./);
  assert.equal(calls[0].body.questions.direction.criteria.up, 'toward the top of the board; the next square is empty');
  calls[0].resolve(reply('left'));
  await flush();
  assert.match(controller.lastError, /invalid response/);
  clock.advance(TICK);
  assert.equal(controller.game.heading, 'right');
  assert.equal(controller.metrics.failures, 1);
  assert.equal(controller.log.length, 0);
});

test('with one safe move, code takes it without asking the model', async () => {
  const { clock, calls, controller } = setup(FACTS);
  controller.game.snake = [{ x: 9, y: 0 }, { x: 8, y: 0 }, { x: 7, y: 0 }];
  controller.play();
  assert.equal(calls.length, 0);
  clock.advance(TICK);
  assert.equal(controller.game.heading, 'down');
  assert.equal(controller.log[0].code, true);
  assert.equal(controller.metrics.codeMoves, 1);
  assert.equal(controller.metrics.ages.length, 0);
  assert.equal(calls.length, 1);
});

test('with no safe move, nothing is sent and the snake goes straight', async () => {
  const { clock, calls, controller } = setup(FACTS);
  controller.game.snake = [{ x: 9, y: 0 }, { x: 9, y: 1 }, { x: 8, y: 1 }, { x: 8, y: 0 }, { x: 7, y: 0 }];
  controller.play();
  assert.equal(calls.length, 0);
  clock.advance(TICK);
  assert.equal(controller.game.over, true);
  assert.equal(controller.game.reason, 'wall');
});

test('without the screenshot nothing is captured or sent as a file', async () => {
  const { calls, controller } = setup();
  controller.capture = () => { throw new Error('captured'); };
  controller.reset(TICK, 1, { ...DEFAULT_CONTEXT, screenshot: false });
  controller.play();
  assert.equal(calls[0].body.files, undefined);
  assert.match(calls[0].body.state, /^This is a Snake game/);
});

test('without the safety net deadly options are sent and no code moves', async () => {
  const { clock, calls, controller } = setup();
  controller.reset(TICK, 1, { ...NOTHING, safety: false });
  controller.game.snake = [{ x: 9, y: 0 }, { x: 8, y: 0 }, { x: 7, y: 0 }];
  controller.play();
  assert.deepEqual(Object.keys(calls[0].body.questions.direction.criteria), ['up', 'down', 'right']);
  calls[0].resolve(reply('up'));
  await flush();
  clock.advance(TICK);
  assert.equal(controller.game.reason, 'wall');
  assert.equal(controller.metrics.codeMoves, 0);
});

test('game over stops the clock and saves a results row', async () => {
  const { clock, calls, controller } = setup();
  controller.reset(300, 9, { ...DEFAULT_CONTEXT, screenshot: false, board: true, progress: true });
  controller.game.food = { x: 0, y: 0 };
  controller.play();
  calls[0].resolve(reply('right'));
  await flush();
  clock.advance(10 * 300);
  assert.equal(controller.game.over, true);
  assert.equal(controller.running, false);
  assert.equal(controller.results.length, 1);
  const row = controller.results[0];
  assert.equal(calls[0].body.files, undefined);
  assert.match(calls[0].body.state, /\| \. \. \. # H/);
  assert.equal(controller.log.length > 0 && 'board' in controller.decision, true);
  assert.equal(row.context, 'board+progress');
  assert.equal(row.seed, 9);
  assert.equal(row.tick_ms, 300);
  assert.equal(row.ticks, 6);
  assert.equal(row.duration_ms, 1800);
  assert.equal(row.end_reason, 'wall');
  assert.equal(row.mean_latency_ms, 0);
  const csv = toCsv(controller.results).split('\n');
  assert.equal(csv[0].split(',')[0], 'context');
  assert.equal(csv[1].split(',').at(-1), 'wall');
});
