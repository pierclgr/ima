// Per-game decision metrics and CSV export of completed games

/**
 * @param {number[]} values
 * @returns {number | null} mean, null when empty
 */
export function mean(values) {
  return values.length
    ? values.reduce((sum, value) => sum + value, 0) / values.length
    : null;
}

/**
 * @param {number[]} values
 * @returns {number | null} median, null when empty
 */
export function median(values) {
  if (!values.length) return null;
  const sorted = [...values].sort((a, b) => a - b);
  const middle = Math.floor(sorted.length / 2);
  return sorted.length % 2 ? sorted[middle] : (sorted[middle - 1] + sorted[middle]) / 2;
}

/** Counts requests, latencies and applied answers of one game. */
export class RunMetrics {
  constructor() {
    this.requests = 0;
    this.failures = 0;
    this.latencies = [];
    this.ages = [];
    this.agesMs = [];
    this.late = 0;
    this.codeMoves = 0;
  }

  /**
   * Records an answer applied at a tick.
   * @param {number} age screenshot age in ticks (1 means fresh)
   * @param {number} ageMs milliseconds from capture to apply
   */
  recordMove(age, ageMs) {
    this.ages.push(age);
    this.agesMs.push(ageMs);
    if (age > 1) this.late += 1;
  }
}

/**
 * @param {object[]} rows objects with the same keys
 * @returns {string} CSV text with a header line
 */
export function toCsv(rows) {
  if (!rows.length) return '';
  const keys = Object.keys(rows[0]);
  const cell = (value) => (value === null ? '' : String(value));
  return [keys.join(','), ...rows.map((row) => keys.map((key) => cell(row[key])).join(','))]
    .join('\n') + '\n';
}
