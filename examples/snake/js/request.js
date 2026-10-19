// System One request and response; the context items are chosen in the UI

import { SIZE } from './game.js';

// what the request can contain, in the order of the UI and the results
export const CONTEXT_ITEMS = ['screenshot', 'board', 'head', 'food', 'direction', 'score', 'safety'];

// facts added to each option's description, as Decisio does
export const OPTION_FACTS = ['position', 'cells', 'progress', 'distance', 'exits', 'area', 'loop'];

// only the screenshot is on by default; every addition is opt-in
export const DEFAULT_CONTEXT = {
  screenshot: true, board: false, head: false, food: false, direction: false, score: false, safety: false,
  position: false, cells: false, progress: false, distance: false, exits: false, area: false, loop: false,
};

const GAME =
  'This is a Snake game on a 10 by 10 grid. The snake moves one square at ' +
  'a time and the game ends if the head hits a wall or the body.';

// always sent: Decisio's decision priorities, in the same order
const RULES =
  'Choose the move with these priorities, in this order. First, survive: ' +
  'never move into a wall or into the snake body, and keep space to move ' +
  'later. Then avoid dead ends and moves that leave the snake little free ' +
  'space. Then eat the food when the snake can still escape after eating. ' +
  'Then move toward the food when it is safe. Finally, prefer squares the ' +
  'snake has not visited recently. Moving away from the food can be the ' +
  'right choice when it keeps escape space or avoids a loop.';

const IMAGE =
  'The image shows the board. The background is light gray with a grid. The ' +
  'snake is a thick green line and its head is the dark green circle with ' +
  'eyes. The food is the red apple. Up is the top of the image, down is the ' +
  'bottom, left is the left side and right is the right side.';

const BOARD =
  'The board is written below, one row per line, top row first. H is the ' +
  'snake head, # is the snake body, @ is the food and . is an empty square. ' +
  'The border made of +, - and | is the wall. Up is the first line, down is ' +
  'the last line, left is the start of a line and right is the end of a line.';

const QUESTION =
  'Which direction should the snake head move next to reach the food ' +
  'without hitting a wall or its body?';

const CRITERIA = {
  up: 'toward the top of the board',
  down: 'toward the bottom of the board',
  left: 'toward the left side of the board',
  right: 'toward the right side of the board',
};

const CONTENTS = { wall: 'a wall', body: 'the snake body' };

const PROGRESS = {
  closer: 'closer to the food',
  farther: 'farther from the food',
  eats: 'eats the food',
};

/**
 * Writes the board as text inside a border, one row per line, top row first.
 * @param {import('./game.js').SnakeGame} game
 * @returns {string} rows of `H` (head), `#` (body), `@` (food), `.` (empty)
 *   between `|`, with `+---+` lines above and below
 */
export function boardText(game) {
  const rows = Array.from({ length: SIZE }, () => Array(SIZE).fill('.'));
  if (game.food) rows[game.food.y][game.food.x] = '@';
  game.snake.forEach((cell, index) => { rows[cell.y][cell.x] = index ? '#' : 'H'; });
  const border = `+${'-'.repeat(2 * SIZE + 1)}+`;
  return [border, ...rows.map((row) => `| ${row.join(' ')} |`), border].join('\n');
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
 * food positions, food direction and score. The column and row sentence is
 * also sent for the next cell positions of the options.
 * @param {import('./game.js').SnakeGame} game
 * @param {Object<string, boolean>} context chosen items (`CONTEXT_ITEMS`
 *   and `position`)
 * @returns {string} e.g. "Columns go from ... The head is at column 4, row
 *   5. Food direction from the head: up-right.", or ''
 */
export function contextText(game, context) {
  const head = game.snake[0];
  const facts = [];
  if (context.head || context.food || context.position) {
    facts.push(`Columns go from 0 (left) to ${SIZE - 1} (right) and rows from 0 (top) to ${SIZE - 1} (bottom).`);
  }
  if (context.head) facts.push(`The head is at column ${head.x}, row ${head.y}.`);
  if (context.food) facts.push(`The food is at column ${game.food.x}, row ${game.food.y}.`);
  if (context.direction) facts.push(`Food direction from the head: ${foodDirection(head, game.food)}.`);
  if (context.score) facts.push(`Score: ${game.score} (food eaten so far).`);
  return facts.join(' ');
}

/**
 * Writes one option's description: its direction, then the option facts
 * chosen in `context`. A deadly option (no safety net) only says so.
 * @param {import('./game.js').SnakeGame} game
 * @param {string} direction the option
 * @param {Object<string, boolean>} context chosen items (`OPTION_FACTS`)
 * @returns {string} e.g. "toward the left side of the board; the next
 *   square is empty; closer to the food; 3 safe moves after (up, down, left)"
 */
export function optionText(game, direction, context) {
  const parts = [CRITERIA[direction]];
  if (!OPTION_FACTS.some((fact) => context[fact])) return parts[0];
  if (!game.safeMoves().includes(direction)) {
    return `${parts[0]}; the snake dies: it hits ${CONTENTS[game.nextCells()[direction]]}`;
  }
  const facts = game.moveFacts(direction);
  if (context.position) parts.push(`the next square is at column ${facts.next.x}, row ${facts.next.y}`);
  if (context.cells) parts.push(facts.eats ? 'the next square has the food' : 'the next square is empty');
  if (context.progress) parts.push(PROGRESS[facts.progress]);
  if (context.distance) parts.push(`food distance ${facts.distanceBefore} -> ${facts.distanceAfter}`);
  if (context.exits) parts.push(`${facts.exits.length} safe moves after (${facts.exits.join(', ') || 'none'})`);
  if (context.area) parts.push(`${facts.reachable} free squares reachable after`);
  if (context.loop) parts.push(`loop risk ${facts.loopRisk} (visited ${facts.visits} times recently)`);
  return parts.join('; ');
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
 * @param {Object<string, boolean>} context chosen items (`CONTEXT_ITEMS`
 *   and `OPTION_FACTS`)
 * @param {string} [image] PNG data URL of the clean board, with `screenshot`
 * @returns {object} request body
 */
export function buildRequest(game, context, image) {
  const description = [GAME, RULES];
  if (context.screenshot) description.push(IMAGE);
  if (context.board) description.push(BOARD);
  const blocks = [description.join(' '), contextText(game, context)];
  if (context.board) blocks.push(boardText(game));
  const criteria = Object.fromEntries(
    options(game, context.safety).map((direction) => [direction, optionText(game, direction, context)]),
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
