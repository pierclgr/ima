// System One request and response; the context items are chosen in the UI

import { SIZE } from './game.js';

// what the request can contain, in the order of the UI and the results
export const CONTEXT_ITEMS = ['screenshot', 'board', 'head', 'food', 'direction', 'cells', 'safety'];

export const DEFAULT_CONTEXT = {
  screenshot: true, board: false, head: true, food: true, direction: true, cells: true, safety: true,
};

const GAME =
  'This is a Snake game on a 10 by 10 grid. The snake moves one square at ' +
  'a time and the game ends if the head hits a wall or the body.';

const IMAGE =
  'The image shows the board. The background is light gray with a grid. The ' +
  'snake is a thick green line and its head is the dark green circle with ' +
  'eyes. The food is the red apple. Up is the top of the image, down is the ' +
  'bottom, left is the left side and right is the right side.';

const BOARD =
  'The board is written below, one row per line, top row first. H is the ' +
  'snake head, B is the snake body, F is the food and . is an empty square. ' +
  'Up is the first line, down is the last line, left is the start of a line ' +
  'and right is the end of a line.';

const QUESTION =
  'Which direction should the snake head move next to reach the food ' +
  'without hitting a wall or its body?';

const CRITERIA = {
  up: 'toward the top of the board',
  down: 'toward the bottom of the board',
  left: 'toward the left side of the board',
  right: 'toward the right side of the board',
};

const CONTENTS = { wall: 'a wall', body: 'the snake body', food: 'the food', empty: 'empty' };

/**
 * Writes the board as text, one row per line, top row first.
 * @param {import('./game.js').SnakeGame} game
 * @returns {string} rows of `H` (head), `B` (body), `F` (food), `.` (empty)
 */
export function boardText(game) {
  const rows = Array.from({ length: SIZE }, () => Array(SIZE).fill('.'));
  if (game.food) rows[game.food.y][game.food.x] = 'F';
  game.snake.forEach((cell, index) => { rows[cell.y][cell.x] = index ? 'B' : 'H'; });
  return rows.map((row) => row.join(' ')).join('\n');
}

/**
 * Gives the food direction seen from the head, e.g. `down-left` or `right`.
 * @param {{x: number, y: number}} head
 * @param {{x: number, y: number}} food
 * @returns {string} one of the 8 compass directions
 */
export function foodDirection(head, food) {
  const vertical = food.y < head.y ? 'up' : food.y > head.y ? 'down' : '';
  const horizontal = food.x < head.x ? 'left' : food.x > head.x ? 'right' : '';
  return [vertical, horizontal].filter(Boolean).join('-');
}

/**
 * Writes the facts chosen in `context`, computed from the game: head and
 * food positions, food direction and what the head would enter in each
 * direction except back.
 * @param {import('./game.js').SnakeGame} game
 * @param {Object<string, boolean>} context chosen items (`CONTEXT_ITEMS`)
 * @returns {string} e.g. "Columns go from ... The head is at column 4, row
 *   5. Food direction from the head: up-right. Next to the head: ...", or ''
 */
export function contextText(game, context) {
  const head = game.snake[0];
  const facts = [];
  if (context.head || context.food) {
    facts.push(`Columns go from 0 (left) to ${SIZE - 1} (right) and rows from 0 (top) to ${SIZE - 1} (bottom).`);
  }
  if (context.head) facts.push(`The head is at column ${head.x}, row ${head.y}.`);
  if (context.food) facts.push(`The food is at column ${game.food.x}, row ${game.food.y}.`);
  if (context.direction) facts.push(`Food direction from the head: ${foodDirection(head, game.food)}.`);
  if (context.cells) {
    const cells = Object.entries(game.nextCells())
      .map(([direction, content]) => `${direction} is ${CONTENTS[content]}`);
    facts.push(`Next to the head: ${cells.join(', ')}.`);
  }
  return facts.join(' ');
}

/**
 * Gives the answer options: the directions except back, and with the safety
 * net only the ones that do not hit a wall or the body.
 * @param {import('./game.js').SnakeGame} game
 * @param {boolean} safety true to use the safety net
 * @returns {string[]} directions in a fixed order
 */
export function options(game, safety) {
  return safety ? game.safeMoves() : Object.keys(game.nextCells());
}

/**
 * Builds a `/v1/systemone` request with the items chosen in `context`.
 * @param {import('./game.js').SnakeGame} game
 * @param {Object<string, boolean>} context chosen items (`CONTEXT_ITEMS`)
 * @param {string} [image] PNG data URL of the clean board, with `screenshot`
 * @returns {object} request body
 */
export function buildRequest(game, context, image) {
  const description = [GAME];
  if (context.screenshot) description.push(IMAGE);
  if (context.board) description.push(BOARD);
  const blocks = [description.join(' '), contextText(game, context)];
  if (context.board) blocks.push(boardText(game));
  const criteria = Object.fromEntries(
    options(game, context.safety).map((direction) => [direction, CRITERIA[direction]]),
  );
  return {
    state: blocks.filter(Boolean).join('\n\n'),
    ...(context.screenshot ? { files: [image] } : {}),
    questions: { direction: { type: 'choice', instructions: QUESTION, criteria } },
  };
}

/**
 * Reads the chosen direction from a `/v1/systemone` response.
 * @param {object} response parsed response body
 * @param {string[]} options directions sent in the request
 * @returns {{direction: string, probabilities: object, model: string}}
 * @throws {Error} when the answer is missing or malformed
 */
export function parseAnswer(response, options) {
  const answer = response?.answers?.direction;
  const probabilities = answer?.probabilities;
  if (
    !options.includes(answer?.choice) ||
    !probabilities ||
    options.some((direction) => typeof probabilities[direction] !== 'number')
  ) {
    throw new Error('invalid response: no direction choice with probabilities');
  }
  return { direction: answer.choice, probabilities, model: response.model };
}
