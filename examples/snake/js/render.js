// Board drawing (the model screenshot) and the separate decision overlay

import { SIZE } from './game.js';

const COLORS = {
  background: '#f4f4f1',
  grid: '#e4e4df',
  body: '#3fa66b',
  head: '#23744a',
  eye: '#ffffff',
  pupil: '#1d1d1b',
  food: '#e5484d',
  stem: '#6b4f2a',
  leaf: '#3fa66b',
  overlay: '37, 99, 235',
  label: '#1e3a8a',
};

const OFFSETS = { up: [0, -1], down: [0, 1], left: [-1, 0], right: [1, 0] };

/**
 * @param {{x: number, y: number}} cell
 * @param {number} size cell size in pixels
 * @returns {[number, number]} pixel center of the cell
 */
const center = (cell, size) => [(cell.x + 0.5) * size, (cell.y + 0.5) * size];

/**
 * Draws a red apple in a cell.
 * @param {CanvasRenderingContext2D} context
 * @param {{x: number, y: number}} cell
 * @param {number} size cell size in pixels
 */
function drawApple(context, cell, size) {
  const [x, y] = center(cell, size);
  context.fillStyle = COLORS.food;
  context.beginPath();
  context.arc(x, y + size * 0.05, size * 0.3, 0, 2 * Math.PI);
  context.fill();
  context.strokeStyle = COLORS.stem;
  context.lineWidth = size * 0.06;
  context.lineCap = 'round';
  context.beginPath();
  context.moveTo(x, y - size * 0.2);
  context.lineTo(x + size * 0.04, y - size * 0.34);
  context.stroke();
  context.fillStyle = COLORS.leaf;
  context.beginPath();
  context.ellipse(x + size * 0.14, y - size * 0.3, size * 0.1, size * 0.05, -0.5, 0, 2 * Math.PI);
  context.fill();
}

/**
 * Draws the snake as one continuous line with a round head and eyes.
 * @param {CanvasRenderingContext2D} context
 * @param {import('./game.js').SnakeGame} game
 * @param {number} size cell size in pixels
 */
function drawSnake(context, game, size) {
  const points = game.snake.map((cell) => center(cell, size));
  context.strokeStyle = COLORS.body;
  context.lineWidth = size * 0.62;
  context.lineCap = 'round';
  context.lineJoin = 'round';
  context.beginPath();
  points.forEach(([x, y], index) => (index ? context.lineTo(x, y) : context.moveTo(x, y)));
  context.stroke();

  const [x, y] = points[0];
  context.fillStyle = COLORS.head;
  context.beginPath();
  context.arc(x, y, size * 0.4, 0, 2 * Math.PI);
  context.fill();
  // eyes sit forward on the head, one on each side of the heading
  const [dx, dy] = OFFSETS[game.heading];
  for (const side of [-1, 1]) {
    const eyeX = x + dx * size * 0.12 - dy * side * size * 0.17;
    const eyeY = y + dy * size * 0.12 + dx * side * size * 0.17;
    context.fillStyle = COLORS.eye;
    context.beginPath();
    context.arc(eyeX, eyeY, size * 0.1, 0, 2 * Math.PI);
    context.fill();
    context.fillStyle = COLORS.pupil;
    context.beginPath();
    context.arc(eyeX + dx * size * 0.04, eyeY + dy * size * 0.04, size * 0.05, 0, 2 * Math.PI);
    context.fill();
  }
}

/**
 * Draws the clean board that is sent to the model.
 * @param {CanvasRenderingContext2D} context
 * @param {import('./game.js').SnakeGame} game
 */
export function drawBoard(context, game) {
  const { width } = context.canvas;
  const size = width / SIZE;
  context.fillStyle = COLORS.background;
  context.fillRect(0, 0, width, width);
  context.strokeStyle = COLORS.grid;
  context.lineWidth = 1;
  context.beginPath();
  for (let i = 1; i < SIZE; i += 1) {
    context.moveTo(i * size, 0);
    context.lineTo(i * size, width);
    context.moveTo(0, i * size);
    context.lineTo(width, i * size);
  }
  context.stroke();
  if (game.food) drawApple(context, game.food, size);
  drawSnake(context, game, size);
}

/**
 * Draws direction probabilities around the head; never part of a screenshot.
 * @param {CanvasRenderingContext2D} context
 * @param {import('./game.js').SnakeGame} game
 * @param {{direction: string, probabilities: object} | null} decision
 */
export function drawOverlay(context, game, decision) {
  const { width } = context.canvas;
  const size = width / SIZE;
  context.clearRect(0, 0, width, width);
  if (!decision) return;
  const head = game.snake[0];
  context.font = `600 ${Math.round(size / 4)}px system-ui, sans-serif`;
  context.textAlign = 'center';
  context.textBaseline = 'middle';
  for (const [direction, probability] of Object.entries(decision.probabilities)) {
    const [dx, dy] = OFFSETS[direction];
    const cell = { x: head.x + dx, y: head.y + dy };
    context.beginPath();
    context.roundRect(cell.x * size + 3, cell.y * size + 3, size - 6, size - 6, size * 0.18);
    context.fillStyle = `rgba(${COLORS.overlay}, ${0.08 + 0.3 * probability})`;
    context.fill();
    if (direction === decision.direction) {
      context.strokeStyle = `rgb(${COLORS.overlay})`;
      context.lineWidth = 2;
      context.stroke();
    }
    context.fillStyle = COLORS.label;
    context.fillText(`${Math.round(probability * 100)}%`, ...center(cell, size));
  }
}
