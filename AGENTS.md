# AGENTS.md

## Project
IMA: Instant Multimodal Answers.

Decoder-based System One decision model. A pretrained multimodal HF LLM
answers typed questions in one forward pass (no generation). Probabilities are
read from the logits of the option letters.

## Layout
- `src/`: source code (package `src`, imports as `from src.x import ...`)
  - `constants.py`: shared constants
  - `model/`: the model
    - `config.py`: `DecisionEngineConfig` (`PreTrainedConfig`)
    - `output.py`: `DecisionEngineOutput` (shared model output;
      `from_logits`: temperature softmax and loss, shared by both systems)
    - `system_one.py`: `PreTrainedSystemOneModel` (`PreTrainedModel`
      wrapping the backbone, option logits, probabilities and loss)
    - `system_two.py`: `SystemTwoModel` (wraps a System One model, the
      backbone generates its answer; System Two test mode;
      `from_system_one` builds it with its prompt and thinking variables)
    - `loader.py`: `ModelLoader` (loads a trained model or wraps a raw
      backbone, picked from `model_type`; returns model and processor)
  - `common/`: shared code
    - `types.py`: `Question`
    - `request.py`: `RequestParser` (request parsing)
    - `question_types.py`: `QuestionType` and one class per type
      (`ChoiceType`, `NoulType`, `ScoreType`: options, label index, answer
      format); add a type here only
    - `prompt.py`: `PromptBuilder` (chat messages, letter token ids, and
      `encode`: the only chat template call, left padding)
    - `config_parser.py`: `ConfigParser` (YAML configs with CLI overrides,
      section helpers, used by `TrainConfig`, `CalibrationConfig` and
      `TestConfig`)
  - `data/`: training data
    - `dataset.py`: `DecisionEngineDataset` (labelled records, one item per
      question, choice option shuffle)
    - `collator.py`: `DecisionEngineCollator` (batch with `labels`,
      `option_mask`)
    - `hub.py`: `HubRecordLoader` (HF Hub or local Parquet splits in the
      kev-vision layout to records)
    - `config.py`: `DataSettings` (training data on the HF Hub, shared by
      the training and calibration configs)
  - `pipeline/`: inference
    - `system_one.py`: `SystemOnePipeline` (HF `ChunkPipeline`, Jev
      `/v1/systemone` request and response; `from_pretrained` loads a
      trained model or a raw backbone)
  - `llamacpp/`: GGUF export, native System One serving and offline runtime
    - `export.py`: lazily unwraps merged checkpoints for llama.cpp's
      converter; adds the trained prompt and calibration metadata
    - `prompt.py`: native `systemone` template from the source processor
    - `serve.py`: standard-library launcher for native `llama-server`
    - `runtime.py`: persistent native worker, built under `.cache/llamacpp`
    - `native/`: C++17 adapter linked to the checkout's shared libraries
  - `evaluation/`: running models on labelled records (training and testing)
    - `evaluator.py`: `DecisionEvaluator` (runs System One or Two on
      records)
    - `metrics.py`: `DecisionMetrics` (accuracy, Brier score, ECE, System
      Two errors)
    - `llamacpp.py`: native predictions with shared targets and metrics
  - `training/`: LoRA training
    - `config.py`: `TrainConfig` (YAML run config: `backbone`, `seed`,
      `lora`, `data`, `training`, optional `calibration` and
      `distillation`)
    - `train.py`: training script (`Trainer`, LoRA on the language model;
      calibrates at the end if configured)
  - `calibration/`: temperature calibration (after training or standalone)
    - `config.py`: `CalibrationSettings` (also used by `TrainConfig`) and
      `CalibrationConfig` (standalone calibration: `model`, `seed`, `data`,
      `calibration`, `calibrating`)
    - `calibrate.py`: `TemperatureCalibrator` (global temperature) and
      the standalone calibration script
  - `distillation/`: offline System One teacher-to-student distillation
    - `config.py`: `DistillationSettings` (teacher, `soft_labels`, loss;
      the `distillation` section of a train config) and `DistillationConfig`
      (cache-only config)
    - `cache.py`: atomic target export and checked record alignment
    - `generate.py`: builds missing targets or checks/reuses existing targets
    - `trainer.py`: teacher collator, masked CE/KL loss, human-only eval
  - `testing/`: standalone testing of a trained model
    - `config.py`: `TestConfig` (YAML: `model`, optional `temperature`,
      `backend` default `hf`, `data`, `testing`, optional `llamacpp`)
    - `test.py`: test script (metrics and seconds per question on a
      dataset split: model time only, after 10 warm-up batches; optional
      random `data.sample`)
  - `modal/`: Modal cloud runs
    - `app.py`: Modal app with `cache`, `train`, `calibrate`, `test`
      (GPUs, Volumes `models`, `datasets`, `runs` as persistent caches)
    - `tensorboard_app.py`: separate Modal app serving TensorBoard (web
      app on the `runs` Volume), so runs of `app.py` can go in parallel
    - `cache.py`: downloads model/data or builds a distillation target cache
  - `runpod/`: RunPod cloud runs
    - `app.py`: launcher, one pod per task (`cache`, `train`, `calibrate`,
      `test`); the pod clones the repo at a commit, runs the task (and
      TensorBoard for `train`) and deletes itself; network volume at
      `/workspace` with `models`, `datasets`, `runs`, `uv`; REST API v2
      with `urllib`
- `examples/snake/`: screenshot Snake demo (dependency-free)
  - `server.py`: standard-library static server on `127.0.0.1:8081`
  - `js/`: ES modules: `game.js` (seeded rules, next cells, safe moves,
    shortest path, per-move facts), `request.js` (request from the context items and
    option facts chosen in the UI, answer parsing), `controller.js` (fixed
    clock, one request in flight, one answer waiting), `metrics.js`,
    `render.js`, `app.js` (UI), `theme.js` (saved light/dark preference)
- `configs/train/`: training run configs (YAML)
- `configs/calibration/`: standalone calibration configs (YAML)
- `configs/test/`: test run configs (YAML)
- `configs/distillation/`: soft label caching configs (`data`, `distillation`)
- `tests/`: tracked regression tests; no model downloads or cloud jobs
- `assets/`: images (e.g. the results graph used by `README.md`)
- `README.md`: project overview (install, usage, cloud runs, Snake demo,
  results)

## Commands
- `uv sync`: install the environment (includes `flash-linear-attention`,
  fast Qwen3.5/3.8 linear attention on CUDA)
- `uv run pytest`: run the tests
- `uv sync --extra llamacpp`: add the GGUF export dependency
- `uv run --extra llamacpp python -m src.llamacpp.export MODEL OUTPUT
  --llama-cpp-dir CHECKOUT [--quantization F16|Q8_0|Q4_K_M]`: export a
  merged checkpoint and its projector (source stays unchanged)
- `uv run python -m src.llamacpp.serve EXPORT --llama-cpp-dir CHECKOUT
  [--host HOST] [--port PORT] [--ctx-size N] [--gpu-layers N]
  [--api-key KEY]`: start the native System One API
- `uv run python examples/snake/server.py`: serve the Snake demo; it calls a
  separately started llama-server API from the browser
- `node --test 'tests/snake/*.test.mjs'`: run the Snake demo tests (Node's
  built-in runner, no packages)
- `uv run python -m src.training.train configs/train/<run>.yml
  [--section.key value]`: train (the gated dataset needs `hf auth login`)
- `uv run python -m src.calibration.calibrate configs/calibration/<run>.yml
  [--section.key value]`: calibrate a saved model
- `uv run python -m src.testing.test configs/test/<run>.yml
  [--section.key value]`: test a trained model (or a raw HF backbone in
  `model`, zero-shot; a `system_two` section for System Two), prints the
  metrics
- The same test command with `backend: llamacpp` uses a GGUF export as
  `model`, one question per call, and local Parquet or Hub data. Native
  inference seconds exclude preparation and IPC; the first ten questions
  count in accuracy but not timing. GGUF testing does not use HF arguments
  and shows startup status and question progress on stderr
- `uv run python -m src.modal.cache configs/train/<run>.yml`: download
  the backbone and dataset of a config
- `uv run python -m src.modal.cache
  configs/distillation/<run>.yml [--section.key value]`: cache teacher
  soft labels (a train config with a `distillation` section distills offline)
- `modal run [--detach] src/modal/app.py --task cache|train|calibrate|test
  --config <config.yml> [--overrides "..."] [--gpu X]`:
  same tasks on Modal (needs `modal setup` and Secret `huggingface` with
  `HF_TOKEN`); GPU from the config's `modal.gpu` (default H200), `--gpu`
  wins
- `modal deploy src/modal/tensorboard_app.py`: serve TensorBoard on the
  training logs (URL printed)
- `uv run python -m src.runpod.app --task cache|train|calibrate|test
  --config <config.yml> [--overrides "..."] [--gpu ID] [--ref COMMIT]`:
  same tasks on RunPod (needs `RUNPOD_API_KEY`, `RUNPOD_VOLUME_ID` and a
  RunPod secret `huggingface`); GPU from the config's `runpod.gpu` (default
  `NVIDIA H200`); runs the pushed local `HEAD`
- Both cloud launchers accept `cache` with a distillation config (needs
  GPU; regular caching uses CPU). Outputs stay on the runs volume; soft
  labels go on the datasets volume (`datasets/...` paths).

## Distillation
- Teacher and student: fine-tuned System One models or raw backbones
- Teacher probabilities already include its saved calibration temperature
  (1 for a raw backbone)
- Cache and train (with `distillation`) both build missing soft labels and
  reuse valid ones
- Use one Modal cache function with CPU/GPU invocation resource options
- Release the teacher before loading the student for training
- Human targets remain separate from teacher targets; shuffle both together
- Attach teacher targets before splitting whole calibration records
- Validation and calibration use only human supervision
- Resume a student adapter over the same original fine-tuned backbone

## Conventions
- Snake UI: use shared CSS theme variables; keep the model's board colors
  independent of the page theme and preserve the square board layout
- Snake capture: draw the current game before serializing the PNG; update
  the input preview on request send, independently of the latest answer
- Snake text board: include the character legend directly above the grid
  in both the request and input preview
- System One API: native llama.cpp `files` (`images` alias), image data
  URLs, at most eight images, score questions with 2–10 levels, native
  score confidence; dataset records keep PIL images in `files`
- Offline score datasets keep all levels up to the model's option limit;
  enforce the 10-level API limit only at the public pipeline boundary
- GGUF: use llama.cpp's `openjev` scoring metadata with this project's
  prompt and temperature; never copy OpenJev's calibration constants
- Python: PEP8, Google docstrings, type hints everywhere (PEP 484)
- Native offline adapter: C++17, existing llama.cpp public/helper APIs,
  device synchronization around model work, no upstream source changes
- Comments: start lowercase, no ending period
- File names: `snake_case.py`; Markdown: `UPPER_SNAKE_CASE.md`
- Markdown: max 80 chars per line
- Structure: `src/` code, `tests/` tests
- Keep test source tracked; benchmark outputs belong in ignored `runs/`
- Native integration tests opt in with `LLAMA_CPP_DIR` and `IMA_TEST_EXPORTS`
- Git: branches `feature/snake_case`, `fix/snake_case`; commit messages in
  past tense, one short sentence, name the files affected
