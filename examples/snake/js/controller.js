// Fixed game clock with at most one request in flight and one answer waiting

import { SnakeGame } from './game.js';
import { mean, median, RunMetrics } from './metrics.js';
import {
  boardText, buildRequest, CONTEXT_ITEMS, DEFAULT_CONTEXT, options, parseAnswer,
} from './request.js';

/** Runs games on a fixed tick and applies the newest model answer. */
export class DemoController {
  /**
   * @param {object} options
   * @param {() => string} options.capture clean board as a PNG data URL
   * @param {typeof fetch} options.fetch HTTP client
   * @param {{now: () => number, setInterval: Function, clearInterval: Function}}
   *   options.clock monotonic clock and timers
   * @param {() => void} options.onChange called after each state change
   */
  constructor({ capture, fetch, clock, onChange }) {
    this.capture = capture;
    this.fetch = fetch;
    this.clock = clock;
    this.onChange = onChange;
    this.apiUrl = 'http://127.0.0.1:8080';
    this.results = [];
    this.timer = null;
    this.generation = 0;
    this.reset(200, 1, DEFAULT_CONTEXT);
  }

  /** @returns {boolean} true while the timer runs */
  get running() {
    return this.timer !== null;
  }

  /**
   * Starts a new game. Any pending reply is ignored.
   * @param {number} tickMs fixed tick interval of the game
   * @param {number} seed board seed
   * @param {Object<string, boolean>} context items sent to the model
   *   (`CONTEXT_ITEMS`)
   */
  reset(tickMs, seed, context) {
    this.stop();
    this.tickMs = tickMs;
    this.seed = seed;
    this.context = { ...context };
    this.game = new SnakeGame(seed);
    this.metrics = new RunMetrics();
    this.log = [];
    this.decision = null;
    this.lastRequest = null;
    this.lastResponse = null;
    this.lastError = null;
    this.model = null;
    this.onChange();
  }

  /** Starts or resumes the game clock and sends a screenshot. */
  play() {
    if (this.running || this.game.over) return;
    this.timer = this.clock.setInterval(() => this.tick(), this.tickMs);
    this.requestIfIdle();
    this.onChange();
  }

  /** Stops the clock. Any pending reply is ignored. */
  pause() {
    this.stop();
    this.onChange();
  }

  /** Stops the timer and invalidates the request and waiting answer. */
  stop() {
    if (this.timer !== null) this.clock.clearInterval(this.timer);
    this.timer = null;
    this.generation += 1;
    this.inFlight = false;
    this.pending = null;
  }

  /** Moves the game one tick with the newest answer, then asks again. */
  tick() {
    const answer = this.pending;
    this.pending = null;
    this.game.step(answer ? answer.direction : null);
    if (answer) {
      const age = this.game.tick - answer.captureTick;
      const ageMs = this.clock.now() - answer.captureTime;
      if (!answer.code) this.metrics.recordMove(age, ageMs);
      this.log.unshift({
        captureTick: answer.captureTick,
        applyTick: this.game.tick,
        age,
        ageMs,
        direction: answer.direction,
        latency: answer.latency,
        code: answer.code,
      });
    }
    if (this.game.over) {
      this.stop();
      this.results.push(this.summary());
    } else {
      this.requestIfIdle();
    }
    this.onChange();
  }

  /**
   * Sends the current board unless a request or answer is waiting. With the
   * safety net and one safe move, code takes it without the model; with no
   * safe move, nothing is sent.
   */
  async requestIfIdle() {
    if (this.inFlight || this.pending || this.game.over) return;
    const choices = options(this.game, this.context.safety);
    const captureTick = this.game.tick;
    const captureTime = this.clock.now();
    if (choices.length < 2) {
      if (choices.length === 1) {
        this.metrics.codeMoves += 1;
        this.pending = {
          direction: choices[0], probabilities: { [choices[0]]: 1 }, code: true, captureTick, captureTime, latency: 0,
        };
      }
      return;
    }
    const generation = this.generation;
    const image = this.context.screenshot ? this.capture() : undefined;
    const board = this.context.board ? boardText(this.game) : undefined;
    const request = buildRequest(this.game, this.context, image);
    this.inFlight = true;
    this.metrics.requests += 1;
    this.lastRequest = request;
    let response = null;
    try {
      const reply = await this.fetch(`${this.apiUrl}/v1/systemone`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(request),
      });
      const text = await reply.text();
      if (!reply.ok) throw new Error(`HTTP ${reply.status}: ${text}`);
      response = JSON.parse(text);
      const answer = parseAnswer(response, Object.keys(request.questions.direction.criteria));
      if (generation !== this.generation) return;
      const latency = this.clock.now() - captureTime;
      this.metrics.latencies.push(latency);
      this.pending = { ...answer, image, board, code: false, captureTick, captureTime, latency };
      this.decision = this.pending;
      this.model = answer.model;
      this.lastError = null;
    } catch (error) {
      if (generation !== this.generation) return;
      this.metrics.failures += 1;
      this.lastError = error.message;
    } finally {
      if (generation === this.generation) {
        this.inFlight = false;
        this.lastResponse = response;
        this.onChange();
      }
    }
  }

  /** @returns {object} results row of the current game */
  summary() {
    const { metrics, game } = this;
    return {
      context: CONTEXT_ITEMS.filter((item) => this.context[item]).join('+') || 'none',
      seed: this.seed,
      tick_ms: this.tickMs,
      score: game.score,
      ticks: game.tick,
      duration_ms: game.tick * this.tickMs,
      requests: metrics.requests,
      code_moves: metrics.codeMoves,
      mean_latency_ms: mean(metrics.latencies),
      median_latency_ms: median(metrics.latencies),
      mean_age_ticks: mean(metrics.ages),
      late_answers: metrics.late,
      failures: metrics.failures,
      end_reason: game.reason,
    };
  }
}
