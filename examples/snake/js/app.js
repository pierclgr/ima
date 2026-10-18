// Page wiring: controls, panels and API health

import { DemoController } from './controller.js';
import { CONTEXT_ITEMS, DEFAULT_CONTEXT } from './request.js';
import { mean, median, toCsv } from './metrics.js';
import { drawBoard, drawOverlay } from './render.js';

const element = (id) => document.getElementById(id);
const board = element('board');
const boardContext = board.getContext('2d');
const overlayContext = element('overlay').getContext('2d');
const LABELS = {
  screenshot: 'Screenshot',
  board: 'Text board',
  head: 'Head position',
  food: 'Food position',
  direction: 'Food direction',
  cells: 'Next cells',
  safety: 'Safety net (safe options only)',
};

// one checkbox per context item, checked as in the default context
element('context').insertAdjacentHTML('beforeend', CONTEXT_ITEMS.map((item) => `
  <label><input type="checkbox" id="context-${item}"${DEFAULT_CONTEXT[item] ? ' checked' : ''}> ${LABELS[item]}</label>`).join(''));
const checkboxes = CONTEXT_ITEMS.map((item) => element(`context-${item}`));
const inputs = [element('speed'), element('seed'), ...checkboxes];

/** @returns {string} number with a fixed precision, or a dash */
const format = (value, digits = 0) => (value === null || value === undefined ? '–' : value.toFixed(digits));

// the constructor calls onChange before the assignment, so render waits for it
let controller = null;
controller = new DemoController({
  capture: () => board.toDataURL('image/png'),
  fetch: (...args) => window.fetch(...args),
  clock: {
    now: () => performance.now(),
    setInterval: (callback, ms) => window.setInterval(callback, ms),
    clearInterval: (id) => window.clearInterval(id),
  },
  onChange: () => controller && render(),
});

/** Starts a new game with the selected tick interval, seed and context. */
function reset() {
  controller.reset(
    Number(element('speed').value),
    Math.trunc(Number(element('seed').value)),
    Object.fromEntries(CONTEXT_ITEMS.map((item, index) => [item, checkboxes[index].checked])),
  );
}

/** Draws the board and refreshes all panels from the controller state. */
function render() {
  const { game, metrics, decision } = controller;
  drawBoard(boardContext, game);
  drawOverlay(overlayContext, game, decision);

  const started = controller.running || game.tick > 0;
  inputs.forEach((input) => { input.disabled = started; });
  const play = element('play');
  play.disabled = game.over;
  play.textContent = game.over ? 'Game over' : controller.running ? 'Pause' : 'Play';

  element('model').textContent = `Model: ${controller.model ?? '–'}`;
  element('error').hidden = !controller.lastError;
  element('error').textContent = controller.lastError ?? '';

  element('probabilities').innerHTML = decision
    ? Object.entries(decision.probabilities).map(([direction, probability]) => `
      <div class="bar${direction === decision.direction ? ' chosen' : ''}">
        <span>${direction}</span><span><i style="width:${probability * 100}%"></i></span>
        <span>${(probability * 100).toFixed(1)}%</span>
      </div>`).join('')
    : '<p>No answer yet</p>';

  const lastMove = controller.log[0];
  const stats = {
    Score: game.score,
    Ticks: game.tick,
    Duration: `${format(game.tick * controller.tickMs / 1000, 1)} s`,
    'Tick interval': `${controller.tickMs} ms`,
    Seed: controller.seed,
    'Model requests': metrics.requests,
    'Code moves (one safe move)': metrics.codeMoves,
    'Round-trip last / mean / median': `${format(metrics.latencies.at(-1))} / ${format(mean(metrics.latencies))} / ${format(median(metrics.latencies))} ms`,
    'Answer age last': lastMove ? `${lastMove.age} ticks, ${format(lastMove.ageMs)} ms` : '–',
    'Answer age mean': `${format(mean(metrics.ages), 2)} ticks, ${format(mean(metrics.agesMs))} ms`,
    'Late answers': metrics.late,
    'API failures': metrics.failures,
    'End reason': game.reason ?? '–',
  };
  element('stats').innerHTML = Object.entries(stats).map(([key, value]) => `<dt>${key}</dt><dd>${value}</dd>`).join('');

  if (decision?.image) element('screenshot').src = decision.image;
  else element('screenshot').removeAttribute('src');
  element('screenshot').hidden = !decision?.image;
  element('board-text').hidden = !decision?.board;
  element('board-text').textContent = decision?.board ?? '';
  const files = controller.lastRequest?.files;
  const request = files
    ? { ...controller.lastRequest, files: files.map((file) => `${file.slice(0, 40)}… (${file.length} chars)`) }
    : controller.lastRequest;
  element('json').textContent = JSON.stringify({ request, response: controller.lastResponse }, null, 2);

  element('log').innerHTML = controller.log.slice(0, 100).map((move) => `
    <tr>
      <td>${move.captureTick}</td><td>${move.applyTick}</td><td>${move.age}</td>
      <td>${format(move.ageMs)}</td><td>${move.direction}${move.code ? ' (code)' : ''}</td><td>${move.code ? '–' : format(move.latency)}</td>
    </tr>`).join('');

  const rows = controller.results;
  element('results').innerHTML = rows.length
    ? `<thead><tr>${Object.keys(rows[0]).map((key) => `<th>${key}</th>`).join('')}</tr></thead>
       <tbody>${rows.map((row) => `<tr>${Object.values(row).map((value) =>
         `<td>${typeof value === 'number' && !Number.isInteger(value) ? value.toFixed(2) : value ?? '–'}</td>`).join('')}</tr>`).join('')}</tbody>`
    : '<tbody><tr><td>No completed games</td></tr></tbody>';
  element('csv').disabled = !rows.length;
}

/** Shows whether the API answers `/health`. */
async function checkHealth() {
  const status = element('health');
  try {
    const reply = await window.fetch(`${controller.apiUrl}/health`);
    status.textContent = reply.ok ? 'ready' : `error: HTTP ${reply.status}`;
    status.className = `status ${reply.ok ? 'ok' : 'bad'}`;
  } catch {
    status.textContent = 'error: unreachable';
    status.className = 'status bad';
  }
}

element('play').addEventListener('click', () => (controller.running ? controller.pause() : controller.play()));
element('reset').addEventListener('click', reset);
inputs.forEach((input) => input.addEventListener('change', reset));
element('speed').addEventListener('input', () => {
  element('speed-value').textContent = `${element('speed').value} ms`;
});
element('api').addEventListener('change', () => {
  controller.apiUrl = element('api').value.replace(/\/+$/, '');
  checkHealth();
});
element('csv').addEventListener('click', () => {
  const link = document.createElement('a');
  link.href = URL.createObjectURL(new Blob([toCsv(controller.results)], { type: 'text/csv' }));
  link.download = 'snake_results.csv';
  link.click();
  URL.revokeObjectURL(link.href);
});

reset();
checkHealth();
window.setInterval(checkHealth, 5000);
