import assert from 'node:assert/strict';
import { test } from 'node:test';

import { SnakeGame } from '../../examples/snake/js/game.js';
import {
  boardText, buildRequest, CONTEXT_ITEMS, contextText, DEFAULT_CONTEXT, foodDirection, OPTION_FACTS, optionText,
  options, parseAnswer,
} from '../../examples/snake/js/request.js';

const IMAGE = 'data:image/png;base64,AAAA';
const ITEMS = [...CONTEXT_ITEMS, ...OPTION_FACTS];
const NOTHING = Object.fromEntries(ITEMS.map((item) => [item, false]));
const EVERYTHING = Object.fromEntries(ITEMS.map((item) => [item, true]));

/** @returns {SnakeGame} head at (9, 0) going right, food just below */
function cornerGame() {
  const game = new SnakeGame(1);
  game.snake = [{ x: 9, y: 0 }, { x: 8, y: 0 }, { x: 7, y: 0 }];
  game.food = { x: 9, y: 1 };
  return game;
}

test('the priority rules are always sent, right after the game description', () => {
  for (const context of [NOTHING, DEFAULT_CONTEXT, EVERYTHING]) {
    const description = buildRequest(cornerGame(), context, IMAGE).state.split('\n\n')[0];
    assert.match(description, /the body\. Choose the move with these priorities, in this order\. First, survive: never move into a wall/);
  }
});

test('the default context sends only the screenshot and plain options', () => {
  assert.deepEqual(ITEMS.filter((item) => DEFAULT_CONTEXT[item]), ['screenshot']);
  const game = cornerGame();
  const request = buildRequest(game, DEFAULT_CONTEXT, IMAGE);
  assert.deepEqual(Object.keys(request), ['state', 'files', 'questions']);
  assert.deepEqual(request.files, [IMAGE]);
  assert.match(request.state, /^This is a Snake game .* The image shows the board\./);
  assert.doesNotMatch(request.state, /\n|column|Score/);
  const question = request.questions.direction;
  assert.equal(question.type, 'choice');
  assert.deepEqual(question.criteria, {
    up: 'toward the top of the board',
    down: 'toward the bottom of the board',
    right: 'toward the right side of the board',
  });
});

test('only the image changes between two screenshot requests', () => {
  const game = cornerGame();
  const first = buildRequest(game, DEFAULT_CONTEXT, IMAGE);
  const second = buildRequest(game, DEFAULT_CONTEXT, 'data:image/png;base64,BBBB');
  assert.deepEqual({ ...first, files: [] }, { ...second, files: [] });
});

test('an empty context sends only the game description and options', () => {
  const request = buildRequest(new SnakeGame(1), NOTHING);
  assert.deepEqual(Object.keys(request), ['state', 'questions']);
  assert.match(
    request.state,
    /^This is a Snake game on a 10 by 10 grid\. The snake moves one square at a time and the game ends if the head hits a wall or the body\. Choose the move with these priorities, in this order\. First, survive: .* avoids a loop\.$/,
  );
  assert.deepEqual(Object.keys(request.questions.direction.criteria), ['up', 'down', 'right']);
});

test('the text board comes last, with its legend in the description', () => {
  const game = cornerGame();
  const request = buildRequest(game, { ...NOTHING, board: true });
  assert.deepEqual(Object.keys(request), ['state', 'questions']);
  assert.match(request.state, /H is the snake head/);
  assert.ok(request.state.endsWith(`\n\n${boardText(game)}`));
});

test('everything together keeps the order description, facts, board', () => {
  const game = cornerGame();
  const request = buildRequest(game, EVERYTHING, IMAGE);
  const blocks = request.state.split('\n\n');
  assert.equal(blocks.length, 3);
  assert.match(blocks[0], /The image shows the board\..* The board is written below/);
  assert.equal(blocks[1], contextText(game, EVERYTHING));
  assert.equal(blocks[2], boardText(game));
});

test('each fact appears only when chosen', () => {
  const game = cornerGame();
  assert.equal(contextText(game, NOTHING), '');
  assert.equal(
    contextText(game, EVERYTHING),
    'Columns go from 0 (left) to 9 (right) and rows from 0 (top) to 9 (bottom). ' +
      'The head is at column 9, row 0. The food is at column 9, row 1. ' +
      'Food direction from the head: down. Score: 0 (food eaten so far).',
  );
  assert.equal(
    contextText(game, { ...NOTHING, head: true }),
    'Columns go from 0 (left) to 9 (right) and rows from 0 (top) to 9 (bottom). The head is at column 9, row 0.',
  );
  assert.equal(contextText(game, { ...NOTHING, direction: true }), 'Food direction from the head: down.');
  assert.equal(contextText(game, { ...NOTHING, cells: true }), '');
  assert.equal(
    contextText(game, { ...NOTHING, position: true }),
    'Columns go from 0 (left) to 9 (right) and rows from 0 (top) to 9 (bottom).',
  );
});

test('the safety net keeps only safe options, without it all but back', () => {
  const game = cornerGame();
  assert.deepEqual(options(game, true), ['down']);
  assert.deepEqual(options(game, false), ['up', 'down', 'right']);
  const request = buildRequest(game, { ...NOTHING, safety: false });
  assert.deepEqual(Object.keys(request.questions.direction.criteria), ['up', 'down', 'right']);
});

test('the food direction uses the 8 compass directions', () => {
  const head = { x: 4, y: 4 };
  const cases = [
    [{ x: 4, y: 1 }, 'up'], [{ x: 4, y: 8 }, 'down'], [{ x: 0, y: 4 }, 'left'], [{ x: 9, y: 4 }, 'right'],
    [{ x: 1, y: 1 }, 'up-left'], [{ x: 7, y: 0 }, 'up-right'], [{ x: 2, y: 9 }, 'down-left'], [{ x: 5, y: 5 }, 'down-right'],
  ];
  for (const [food, direction] of cases) assert.equal(foodDirection(head, food), direction);
});

test('the board text has a border and one row per line', () => {
  const game = new SnakeGame(1);
  game.food = { x: 9, y: 0 };
  const rows = boardText(game).split('\n');
  assert.equal(rows.length, 12);
  assert.equal(rows[0], '+---------------------+');
  assert.equal(rows[1], '| . . . . . . . . . @ |');
  assert.equal(rows[6], '| . . . # H . . . . . |');
  assert.equal(rows[11], rows[0]);
});

test('option facts go into each option description when chosen', () => {
  const game = new SnakeGame(1);
  game.food = { x: 0, y: 5 };
  game.snake = [{ x: 4, y: 4 }, { x: 4, y: 5 }, { x: 3, y: 5 }, { x: 3, y: 4 }, { x: 3, y: 3 }];
  game.heading = 'up';
  assert.equal(optionText(game, 'right', NOTHING), 'toward the right side of the board');
  assert.equal(
    optionText(game, 'right', EVERYTHING),
    'toward the right side of the board; the next square is at column 5, row 4; the next square is empty; ' +
      'farther from the food; ' +
      'food distance 5 -> 6; 3 safe moves after (up, down, right); 95 free squares reachable after; ' +
      'loop risk low (visited 0 times recently)',
  );
  assert.equal(optionText(game, 'right', { ...NOTHING, progress: true }), 'toward the right side of the board; farther from the food');
  const request = buildRequest(game, { ...NOTHING, safety: true, distance: true });
  assert.deepEqual(request.questions.direction.criteria, {
    up: 'toward the top of the board; food distance 5 -> 6',
    right: 'toward the right side of the board; food distance 5 -> 6',
  });
});

test('next cell content says empty or food for each safe option', () => {
  const game = cornerGame();
  assert.equal(optionText(game, 'down', { ...NOTHING, cells: true }), 'toward the bottom of the board; the next square has the food');
  const open = new SnakeGame(1);
  assert.equal(optionText(open, 'up', { ...NOTHING, cells: true }), 'toward the top of the board; the next square is empty');
});

test('without the safety net a deadly option says the snake dies', () => {
  const game = cornerGame();
  assert.equal(optionText(game, 'up', { ...NOTHING, progress: true }), 'toward the top of the board; the snake dies: it hits a wall');
  assert.equal(optionText(game, 'down', { ...NOTHING, progress: true }), 'toward the bottom of the board; eats the food');
});

test('parses a valid answer', () => {
  const answer = parseAnswer({
    model: 'snake',
    answers: {
      direction: {
        type: 'choice',
        choice: 'left',
        probabilities: { up: 0.2, down: 0.1, left: 0.7 },
      },
    },
  }, ['up', 'down', 'left']);
  assert.equal(answer.direction, 'left');
  assert.equal(answer.model, 'snake');
  assert.equal(answer.probabilities.left, 0.7);
});

test('rejects malformed answers', () => {
  const choices = ['up', 'down', 'left'];
  const probabilities = { up: 0.2, down: 0.1, left: 0.7 };
  for (const response of [
    {},
    { answers: { direction: { choice: 'jump', probabilities } } },
    { answers: { direction: { choice: 'up' } } },
    { answers: { direction: { choice: 'right', probabilities } } },
    { answers: { direction: { choice: 'up', probabilities: { up: 1 } } } },
  ]) {
    assert.throws(() => parseAnswer(response, choices), /invalid response/);
  }
});
