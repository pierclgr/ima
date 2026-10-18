import assert from 'node:assert/strict';
import { test } from 'node:test';

import { SnakeGame } from '../../examples/snake/js/game.js';

/** @returns {SnakeGame} game with its food far from the snake */
function gameWithFood(food = { x: 9, y: 0 }) {
  const game = new SnakeGame(1);
  game.food = food;
  return game;
}

test('moves one cell per tick and keeps the heading without a direction', () => {
  const game = gameWithFood();
  game.step(null);
  assert.deepEqual(game.snake, [{ x: 5, y: 5 }, { x: 4, y: 5 }, { x: 3, y: 5 }]);
  game.step('up');
  assert.deepEqual(game.snake[0], { x: 5, y: 4 });
  assert.equal(game.heading, 'up');
  assert.equal(game.tick, 2);
});

test('eating food grows the snake, adds a point and places new food', () => {
  const game = gameWithFood({ x: 5, y: 5 });
  game.step(null);
  assert.equal(game.score, 1);
  assert.equal(game.snake.length, 4);
  assert.ok(!game.snake.some((cell) => cell.x === game.food.x && cell.y === game.food.y));
});

test('hitting a wall ends the game', () => {
  const game = gameWithFood();
  for (let i = 0; i < 6 && !game.over; i += 1) game.step(null);
  assert.equal(game.over, true);
  assert.equal(game.reason, 'wall');
  assert.equal(game.tick, 6);
});

test('hitting the body ends the game', () => {
  const game = gameWithFood();
  game.snake = [
    { x: 4, y: 5 }, { x: 3, y: 5 }, { x: 2, y: 5 }, { x: 2, y: 4 }, { x: 3, y: 4 }, { x: 4, y: 4 }, { x: 5, y: 4 },
  ];
  game.step('up');
  assert.equal(game.over, true);
  assert.equal(game.reason, 'body');
});

test('moving into the tail cell is allowed when not eating', () => {
  const game = gameWithFood();
  game.snake = [{ x: 4, y: 5 }, { x: 4, y: 4 }, { x: 3, y: 4 }, { x: 3, y: 5 }];
  game.heading = 'down';
  game.step('left');
  assert.equal(game.over, false);
  assert.deepEqual(game.snake[0], { x: 3, y: 5 });
});

test('a reverse turn is reported and the snake continues straight', () => {
  const game = gameWithFood();
  assert.equal(game.step('left'), true);
  assert.equal(game.heading, 'right');
  assert.deepEqual(game.snake[0], { x: 5, y: 5 });
  assert.equal(game.step('up'), false);
});

test('next cells give wall, body, food or empty, without the way back', () => {
  const game = gameWithFood({ x: 5, y: 4 });
  game.snake = [{ x: 4, y: 4 }, { x: 4, y: 5 }, { x: 3, y: 5 }, { x: 3, y: 4 }, { x: 3, y: 3 }];
  game.heading = 'up';
  assert.deepEqual(game.nextCells(), { up: 'empty', left: 'body', right: 'food' });
  game.snake = [{ x: 0, y: 0 }, { x: 1, y: 0 }];
  game.heading = 'left';
  assert.deepEqual(game.nextCells(), { up: 'wall', down: 'empty', left: 'wall' });
});

test('safe moves leave out walls and the body', () => {
  const game = gameWithFood({ x: 5, y: 4 });
  game.snake = [{ x: 4, y: 4 }, { x: 4, y: 5 }, { x: 3, y: 5 }, { x: 3, y: 4 }, { x: 3, y: 3 }];
  game.heading = 'up';
  assert.deepEqual(game.safeMoves(), ['up', 'right']);
  game.snake = [{ x: 0, y: 0 }, { x: 1, y: 0 }];
  game.heading = 'left';
  assert.deepEqual(game.safeMoves(), ['down']);
});

test('the tail cell counts as empty because the tail moves away', () => {
  const game = gameWithFood();
  game.snake = [{ x: 4, y: 5 }, { x: 4, y: 4 }, { x: 3, y: 4 }, { x: 3, y: 5 }];
  game.heading = 'down';
  assert.equal(game.nextCells().left, 'empty');
});

test('the same seed gives the same board and food sequence', () => {
  const play = (seed) => {
    const game = new SnakeGame(seed);
    const foods = [game.food];
    for (const direction of ['up', 'up', 'left', 'left', 'down']) {
      game.food = foods.at(-1);
      game.step(direction);
      foods.push(game.placeFood());
    }
    return foods;
  };
  assert.deepEqual(play(7), play(7));
  assert.notDeepEqual(play(7), play(8));
});

test('a new game with the same seed resets the state', () => {
  const game = new SnakeGame(3);
  const start = JSON.stringify(new SnakeGame(3));
  game.step('up');
  game.step('up');
  assert.notEqual(JSON.stringify(game), start);
  assert.equal(JSON.stringify(new SnakeGame(3)), start);
});
