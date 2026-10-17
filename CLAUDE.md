@AGENTS.md

Distillation design and cache/loss details: @docs/DISTILLATION.md
The cache task can build teacher soft labels; training with a `distillation`
section builds them if missing.
Modal uses one cache function for regular inputs and distillation targets.

GGUF offline testing uses `src.testing.test` with `backend: llamacpp`.
Runtime and timing contract: @docs/LLAMACPP.md
