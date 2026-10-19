@AGENTS.md

Distillation design and cache/loss details: @docs/DISTILLATION.md
The cache task can build teacher soft labels; training with a `distillation`
section builds them if missing.
Modal uses one cache function for regular inputs and distillation targets.

GGUF offline testing uses `src.testing.test` with `backend: llamacpp`.
Startup status and question progress go to stderr, outside native timing.
Runtime and timing contract: @docs/LLAMACPP.md

Test source stays tracked in `tests/`; benchmark outputs go to `runs/`.
Regression suite and native integration setup: @README.md

## Conventions

Follow the project conventions in @AGENTS.md.
Snake page themes, layout, controls, input legends and capture timing:
@examples/snake/README.md
