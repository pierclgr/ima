// Snake rules on a fixed square board, with a seeded food sequence

export const SIZE = 10;
export const DIRECTIONS = ['up', 'down', 'left', 'right'];

const MOVES = { up: [0, -1], down: [0, 1], left: [-1, 0], right: [1, 0] };
export const OPPOSITE = { up: 'down', down: 'up', left: 'right', right: 'left' };

/**
 * Makes a repeatable random generator (mulberry32).
 * @param {number} seed integer seed
 * @returns {() => number} values in [0, 1)
 */
function seededRandom(seed) {
  let state = seed >>> 0;
  return () => {
    state = (state + 0x6d2b79f5) >>> 0;
    let value = state;
    value = Math.imul(value ^ (value >>> 15), value | 1);
    value ^= value + Math.imul(value ^ (value >>> 7), value | 61);
    return ((value ^ (value >>> 14)) >>> 0) / 4294967296;
  };
}

/** One Snake game. The same seed gives the same start and food draws. */
export class SnakeGame {
  /** @param {number} seed integer seed */
  constructor(seed) {
    this.random = seededRandom(seed);
    this.snake = [{ x: 4, y: 5 }, { x: 3, y: 5 }, { x: 2, y: 5 }];
    this.heading = 'right';
    this.tick = 0;
    this.score = 0;
    this.over = false;
    this.reason = null;
    this.food = this.placeFood();
  }

  /** @returns {{x: number, y: number} | null} a free cell, null if full */
  placeFood() {
    const free = [];
    for (let y = 0; y < SIZE; y += 1) {
      for (let x = 0; x < SIZE; x += 1) {
        if (!this.snake.some((cell) => cell.x === x && cell.y === y)) {
          free.push({ x, y });
        }
      }
    }
    return free.length ? free[Math.floor(this.random() * free.length)] : null;
  }

  /**
   * Moves the snake one cell. A reverse turn keeps the current heading.
   * @param {string | null} direction new direction, null keeps the heading
   * @returns {boolean} true when the direction was a reverse turn
   */
  step(direction) {
    const reverse = direction === OPPOSITE[this.heading];
    if (direction && !reverse) this.heading = direction;
    const [dx, dy] = MOVES[this.heading];
    const head = { x: this.snake[0].x + dx, y: this.snake[0].y + dy };
    const eats = head.x === this.food.x && head.y === this.food.y;
    // the tail moves away this tick unless the snake grows
    const body = eats ? this.snake : this.snake.slice(0, -1);
    this.tick += 1;
    if (head.x < 0 || head.y < 0 || head.x >= SIZE || head.y >= SIZE) {
      this.end('wall');
    } else if (body.some((cell) => cell.x === head.x && cell.y === head.y)) {
      this.end('body');
    } else {
      this.snake = [head, ...body];
      if (eats) {
        this.score += 1;
        this.food = this.placeFood();
        if (!this.food) this.end('board full');
      }
    }
    return reverse;
  }

  /**
   * Describes the cell the head would enter in each direction except back.
   * @returns {Object<string, string>} direction to `wall`, `body`, `food`
   *   or `empty`; the tail cell is empty because the tail moves away
   */
  nextCells() {
    const body = this.snake.slice(0, -1);
    return Object.fromEntries(
      DIRECTIONS.filter((direction) => direction !== OPPOSITE[this.heading]).map((direction) => {
        const [dx, dy] = MOVES[direction];
        const cell = { x: this.snake[0].x + dx, y: this.snake[0].y + dy };
        const same = (other) => other.x === cell.x && other.y === cell.y;
        if (cell.x < 0 || cell.y < 0 || cell.x >= SIZE || cell.y >= SIZE) return [direction, 'wall'];
        if (this.food && same(this.food)) return [direction, 'food'];
        return [direction, body.some(same) ? 'body' : 'empty'];
      }),
    );
  }

  /**
   * Safety net: the directions that do not hit a wall or the body next tick.
   * @returns {string[]} directions in `DIRECTIONS` order, without back
   */
  safeMoves() {
    return Object.entries(this.nextCells())
      .filter(([, content]) => content === 'empty' || content === 'food')
      .map(([direction]) => direction);
  }

  /** @param {string} reason why the game ended */
  end(reason) {
    this.over = true;
    this.reason = reason;
  }
}
