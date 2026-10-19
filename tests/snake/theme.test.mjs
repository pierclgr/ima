import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import test from 'node:test';
import vm from 'node:vm';

const source = readFileSync(new URL('../../examples/snake/js/theme.js', import.meta.url), 'utf8');

function setup({ saved = null, dark = false, blocked = false } = {}) {
  const events = {};
  const stored = new Map(saved === null ? [] : [['ima-snake-theme', saved]]);
  const buttons = ['light', 'dark'].map((theme) => ({
    dataset: { themeChoice: theme },
    attributes: {},
    setAttribute(name, value) { this.attributes[name] = value; },
    addEventListener(name, callback) { this[name] = callback; },
  }));
  const document = {
    documentElement: { dataset: {} },
    querySelectorAll: () => buttons,
    addEventListener(name, callback) { events[name] = callback; },
  };
  const preference = {
    matches: dark,
    addEventListener(name, callback) { this[name] = callback; },
  };
  const localStorage = {
    getItem(key) {
      if (blocked) throw new Error('storage blocked');
      return stored.get(key) ?? null;
    },
    setItem(key, value) {
      if (blocked) throw new Error('storage blocked');
      stored.set(key, value);
    },
  };
  vm.runInNewContext(source, { document, localStorage, window: { matchMedia: () => preference } });
  events.DOMContentLoaded();
  return { document, buttons, stored, preference };
}

test('the first visit uses the system theme and marks the active button', () => {
  for (const dark of [false, true]) {
    const { document, buttons } = setup({ dark });
    assert.equal(document.documentElement.dataset.theme, dark ? 'dark' : 'light');
    assert.equal(buttons[0].attributes['aria-pressed'], String(!dark));
    assert.equal(buttons[1].attributes['aria-pressed'], String(dark));
  }
});

test('a saved choice overrides the system theme on reload', () => {
  assert.equal(setup({ saved: 'light', dark: true }).document.documentElement.dataset.theme, 'light');
  assert.equal(setup({ saved: 'dark' }).document.documentElement.dataset.theme, 'dark');
});

test('switching updates the theme, button states and saved preference', () => {
  const { document, buttons, stored } = setup();
  buttons[1].click();
  assert.equal(document.documentElement.dataset.theme, 'dark');
  assert.equal(buttons[0].attributes['aria-pressed'], 'false');
  assert.equal(buttons[1].attributes['aria-pressed'], 'true');
  assert.equal(stored.get('ima-snake-theme'), 'dark');
  buttons[0].click();
  assert.equal(document.documentElement.dataset.theme, 'light');
  assert.equal(stored.get('ima-snake-theme'), 'light');
});

test('system changes apply until a theme is chosen', () => {
  const { document, buttons, preference } = setup();
  preference.change({ matches: true });
  assert.equal(document.documentElement.dataset.theme, 'dark');
  buttons[0].click();
  preference.change({ matches: true });
  assert.equal(document.documentElement.dataset.theme, 'light');
});

test('an unknown saved value falls back to the system theme', () => {
  assert.equal(setup({ saved: 'unknown', dark: true }).document.documentElement.dataset.theme, 'dark');
});

test('switching still works when browser storage is blocked', () => {
  const { document, buttons } = setup({ blocked: true, dark: true });
  assert.equal(document.documentElement.dataset.theme, 'dark');
  buttons[0].click();
  assert.equal(document.documentElement.dataset.theme, 'light');
});
