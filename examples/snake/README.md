# Screenshot Snake demo

A browser Snake game where the llama.cpp System One server chooses each
direction from a PNG screenshot of the board. The game moves on a fixed
clock. Each answer is applied at the next tick, even when its screenshot is
old, so slow answers stay visible in the results.

Design: [docs/SNAKE_DEMO_PLAN.md](../../docs/SNAKE_DEMO_PLAN.md).

## Run

The demo and the API are two separate local processes. Start the API with a
GGUF export (see the main README for the export):

```bash
python -m src.llamacpp.serve exports/qwen3_5_0_8b_q8 --llama-cpp-dir ../llama.cpp
```

Start the demo page server (Python standard library only):

```bash
python examples/snake/server.py
```

Open <http://127.0.0.1:8081/>. The API URL defaults to
`http://127.0.0.1:8080`; the indicator shows its `/health` status. The
browser calls `POST /v1/systemone` directly. The default llama-server CORS
setting allows this.

## How it works

- 10 × 10 board on a 512 × 512 canvas. The clean board is sent as one PNG;
  probability marks are drawn on a separate overlay canvas.
- One `choice` question with `up`, `down`, `left`, `right`, minus the
  direction back into the body. `state` starts with a short game
  description.
- **Context sent to the model**: checkboxes, fixed per game, choose what is
  added. All facts are computed by the game code:
  - Screenshot: the clean board as one PNG, with its color description
  - Text board: the grid as text (one row per line, top row first, `H`
    head, `B` body, `F` food, `.` empty), at the end of `state`
  - Head position, food position: column and row from the top-left, 0 to 9
  - Food direction: from the head (`up`, `down-left`, ...)
  - Next cells: what the head would enter in each direction (a wall, the
    snake body, the food, or empty; the tail counts as empty because it
    moves away)
  - Safety net: only safe directions (no wall, no body) are options. With
    one safe move, code takes it without the model ("code moves"); with
    none, no request is sent. Safety is checked on the board the model
    sees, so a late answer or straight moves while waiting can still hit a
    wall.
- One request in flight and one answer waiting at most. A new screenshot is
  sent after a tick only when both are free.
- Each screenshot is tagged with its tick. The answer is applied at the next
  tick; its age is the apply tick minus the screenshot tick (1 is fresh).
- Pause and reset ignore any reply that is still on its way.

Choose the tick (100–5000 ms), the seed and the context before a game.
They are fixed for that game. The same seed gives the same start board, so
you can compare speeds and contexts. Each finished game adds a row to the
results table (CSV download), with the context as `screenshot+cells+...`.

Latency is the browser round-trip time (`performance.now()`), not inference
time: the API does not return inference timing.

## Tests

The game, request and timing logic have tests with Node's built-in test
runner (no packages):

```bash
node --test 'tests/snake/*.test.mjs'
```

## Credits

Inspired by the Snake page of
[Rizzo Flow](https://github.com/Rizzo-AI-Academy/rizzo-flow) (board,
controls, probabilities, move log, timing). Also see
[typesafe-snake](https://github.com/sorrycc/typesafe-snake) and
[jev-snake](https://github.com/dbssman/jev-snake).
