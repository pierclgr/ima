import assert from 'node:assert/strict';
import { test } from 'node:test';

import { SnakeGame } from '../../examples/snake/js/game.js';
import {
  boardText, buildRequest, CONTEXT_ITEMS, contextText, DEFAULT_CONTEXT, foodDirection, options, parseAnswer,
} from '../../examples/snake/js/request.js';

const IMAGE = 'data:image/png;base64,AAAA';
const NOTHING = Object.fromEntries(CONTEXT_ITEMS.map((item) => [item, false]));
const EVERYTHING = Object.fromEntries(CONTEXT_ITEMS.map((item) => [item, true]));

/** @returns {SnakeGame} head at (9, 0) going right, food just below */
function cornerGame() {
  const game = new SnakeGame(1);
  game.snake = [{ x: 9, y: 0 }, { x: 8, y: 0 }, { x: 7, y: 0 }];
  game.food = { x: 9, y: 1 };
  return game;
}

test('the default context sends one PNG and the facts, not the board', () => {
  const game = cornerGame();
  const request = buildRequest(game, DEFAULT_CONTEXT, IMAGE);
  assert.deepEqual(Object.keys(request), ['state', 'files', 'questions']);
  assert.deepEqual(request.files, [IMAGE]);
  assert.match(request.state, /^This is a Snake game .* The image shows the board\./);
  assert.ok(request.state.endsWith(`\n\n${contextText(game, DEFAULT_CONTEXT)}`));
  assert.doesNotMatch(request.state, /\. \. /);
  const question = request.questions.direction;
  assert.equal(question.type, 'choice');
  assert.deepEqual(Object.keys(question.criteria), ['down']);
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
  assert.equal(
    request.state,
    'This is a Snake game on a 10 by 10 grid. The snake moves one square at a time and the game ends if ' +
      'the head hits a wall or the body.',
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
      'Food direction from the head: down. Next to the head: up is a wall, down is the food, right is a wall.',
  );
  assert.equal(
    contextText(game, { ...NOTHING, head: true }),
    'Columns go from 0 (left) to 9 (right) and rows from 0 (top) to 9 (bottom). The head is at column 9, row 0.',
  );
  assert.equal(contextText(game, { ...NOTHING, direction: true }), 'Food direction from the head: down.');
  assert.equal(
    contextText(game, { ...NOTHING, cells: true }),
    'Next to the head: up is a wall, down is the food, right is a wall.',
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

test('the board text has one row per line with head, body and food', () => {
  const game = new SnakeGame(1);
  game.food = { x: 9, y: 0 };
  const rows = boardText(game).split('\n');
  assert.equal(rows.length, 10);
  assert.equal(rows[0], '. . . . . . . . . F');
  assert.equal(rows[5], '. . B B H . . . . .');
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
