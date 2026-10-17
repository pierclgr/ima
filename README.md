<p align="center">
  <img src="assets/logo.png" alt="IMA" width="150">
</p>

<h2 align="center">IMA: Instant Multimodal Answers</h2>

<p align="center"><em>A decoder-based image-text System One model</em></p>

<p align="center">
  <a href="https://www.repostatus.org/#active"><img
    src="https://flat.badgen.net/badge/repo%20status/Active/green"
    alt="Project Status: Active – The project has reached a stable, usable
    state and is being actively developed."></a>
</p>

A fast decision model built on a pretrained multimodal LLM (Hugging Face).
It answers typed questions about a state (text and images) in one forward
pass per question, with no text generation. It reads the probabilities from
the logits of the option letters (A, B, C, ...).

This is the "System One" model: fast, one shot. For comparison, the project
also has a "System Two" test mode, where the same LLM generates its answer
like a chat model.

## How it works
1. Each question becomes a lettered prompt: images, state, question, options
   (`A. ...`, `B. ...`) and a request to answer with the letter only.
2. The backbone runs one forward pass. Only the logits of the option letters
   at the answer position are kept.
3. A softmax with a global temperature turns them into one probability per
   option. The temperature is fitted after training (calibration).

Question types:
- `noul`: yes or no, returns the probability of "yes"
- `choice`: one option out of many, returns the best option, a confidence
  and the probability of each option
- `score`: ordered levels, returns the expected level, a confidence and the
  probability of each level

Up to 26 options per question (letters A-Z).

## Install
Needs Python 3.11+ and [uv](https://docs.astral.sh/uv/).

```bash
uv sync
```

On CUDA, `flash-linear-attention` makes the Qwen3.5/3.8 linear attention
fast. Without CUDA (e.g. macOS) a slower PyTorch fallback is used.

The training dataset (`Jacqkues/kev-vision-decisions-full`) is gated, so log
in first with `hf auth login`.

## Usage

### Inference
The pipeline takes a `/v1/systemone` request and returns the typed answers.

```python
from src.pipeline.system_one import SystemOnePipeline

# a trained model, or a raw backbone id for zero-shot answers
pipeline = SystemOnePipeline.from_pretrained("runs/qwen3_5_0_8b/final")
response = pipeline({
    "state": "A customer writes: 'I was charged twice this month.'",
    "files": [],
    "questions": {
        "billing": {"type": "noul", "instructions": "Is it about billing?"},
        "tone": {
            "type": "choice",
            "instructions": "What is the tone?",
            "criteria": {"calm": None, "angry": None, "sad": None},
        },
    },
})
# {"model": ..., "answers": {"billing": {"type": "noul", "noul": <p_yes>},
#  "tone": {"type": "choice", "choice": "angry", ...}}, "usage": {...}}
```

Requests follow llama.cpp's native System One format. Put images in
`files` (or `images`) as base64 image data URLs, up to eight images per
request. This replaces the old `media` field. Score questions accept 2–10
levels; score confidence follows llama.cpp's formula.

### GGUF export and API server

Train with the existing scripts, export the merged `final/` directory, then
host it with llama.cpp's native `POST /v1/systemone` API. The server performs
the model computation on its CPU or GPU. It needs no Python training
environment. This integration serves System One only.

Build the tested llama.cpp revision (needs Git, CMake, and a C++ compiler):

```bash
git clone https://github.com/ggml-org/llama.cpp.git ../llama.cpp
git -C ../llama.cpp checkout c0fd78626405c6d9b934efef0ec7291fcc20e3d6
cmake -S ../llama.cpp -B ../llama.cpp/build -DCMAKE_BUILD_TYPE=Release
cmake --build ../llama.cpp/build --config Release --target llama-server llama-quantize -j 4
uv sync --extra llamacpp
```

For NVIDIA GPUs, add `-DGGML_CUDA=ON` to the CMake configure command.
Metal is enabled by default on macOS; CPU serving is also supported.

Export and start the server:

```bash
uv run --extra llamacpp python -m src.llamacpp.export runs/qwen3_5_0_8b/final exports/qwen3_5_0_8b --llama-cpp-dir ../llama.cpp
python -m src.llamacpp.serve exports/qwen3_5_0_8b --llama-cpp-dir ../llama.cpp
```

The export contains `model-F16.gguf` and `mmproj-F16.gguf`. It preserves
the trained weights, prompt, and calibration temperature. It reads the
source weights lazily and does not change the source checkpoint. Export a
merged model, not an adapter checkpoint. Choose a new output directory for
each export.

Use `--quantization Q8_0` or `--quantization Q4_K_M` on the export command
for a smaller language model. The projector stays F16. Quantization can
change probabilities and calibration; GGUF recalibration is not included.

The server defaults to `127.0.0.1:8080`, an 8192-token context, and automatic
GPU use. The launcher accepts `--host`, `--port`, `--ctx-size`,
`--gpu-layers` (`0` for CPU), and `--api-key`. For remote access, set
`--host 0.0.0.0`. Native `/health` and `/v1/models` endpoints are available.

```bash
curl http://127.0.0.1:8080/v1/systemone -H 'Content-Type: application/json' -d '{"state":"I was charged twice.","questions":{"billing":{"type":"noul","instructions":"Is this about billing?"}}}'
```

For images, add `"files": ["data:image/png;base64,..."]` to the JSON body.
See [docs/LLAMACPP.md](docs/LLAMACPP.md) for the export contract and checks,
and the [native API documentation](https://github.com/ggml-org/llama.cpp/blob/c0fd78626405c6d9b934efef0ec7291fcc20e3d6/tools/server/README.md)
for the complete request and response format.

### Train, calibrate, test
Each task reads a YAML config. You can change any value from the command
line with `--key value` or `--section.key value`.

```bash
# LoRA training (calibrates at the end if the config has `calibration`)
uv run python -m src.training.train configs/train/qwen3_5_0_8b.yml

# calibrate a saved model again, standalone (e.g. after quantization)
uv run python -m src.calibration.calibrate configs/calibration/qwen3_5_0_8b.yml

# test a trained model or a raw backbone; prints and saves the metrics
uv run python -m src.testing.test configs/test/qwen3_5_0_8b.yml

# same test in System Two mode (generated answer)
uv run python -m src.testing.test configs/test/qwen3_5_0_8b_system_two.yml
```

Example override:

```bash
uv run python -m src.training.train configs/train/qwen3_5_0_8b.yml --training.learning_rate 5e-5 --data.train "train[:1000]"
```

- Training: LoRA on the linear layers of the language model. The vision
  tower and `lm_head` stay frozen. The merged model goes to
  `<output_dir>/final`.
- Calibration: fits one temperature on a part of the training records that
  is kept out of training. Only the model's `config.json` changes.
- Testing: reports accuracy, Brier score, expected calibration error (ECE),
  loss and seconds per question. System Two also reports the share of
  outputs with no valid answer.

### Test local GGUF models

Use the same test command with the GGUF config. It loads the model once
and runs one question at a time through a small native worker. No server
is needed. CMake and a C++17 compiler build the worker on first use in
the ignored `.cache/llamacpp/` directory. The llama.cpp checkout must
already have shared libraries (`BUILD_SHARED_LIBS=ON`, the default).

```bash
uv run python -m src.testing.test configs/test/qwen3_5_0_8b_llamacpp.yml
```

The config uses `~/llama.cpp`, Metal/CUDA when available, and the local
Parquet files in `datasets/test/` and `datasets/test_ood/`. Set
`--llamacpp.llama_cpp_dir PATH` for another checkout, or
`--llamacpp.gpu_layers 0` for CPU inference, including the projector.
`data.dataset` also accepts the usual HF Hub dataset id. Split slices and
`data.sample` work as before; sampling selects records with a fixed seed.

Run all three exports on both full splits:

```bash
for format in f16 q8 q4; do
  for split in test test_ood; do
    uv run python -m src.testing.test \
      configs/test/qwen3_5_0_8b_llamacpp.yml \
      --model "exports/qwen3_5_0_8b_${format}" \
      --data.split "$split" \
      --testing.output_dir "runs/llamacpp/${format}/${split}"
  done
done
```

Each output directory gets `test_metrics.json`. It includes accuracy,
loss, Brier score, ECE, backend/device, evaluated/skipped/timed question
counts, and `test_seconds_per_question`. The native clock measures vision
encoding, projection, and the language model forward pass. It excludes
loading, image preparation, tokenization, IPC, and metric computation.
Model memory is reset for each question. The first ten questions count in
accuracy but not timing; more than ten eligible questions are required.
`--temperature 1` overrides the saved calibration for all question types.

Offline scoring keeps up to 26 levels, including the 11-level questions
in `test_ood`. The API limit remains 10. With the downloaded datasets,
`test` evaluates 4855 questions and skips 64 choices with 77 options;
`test_ood` evaluates all 1477 questions.

### Offline distillation
Distill the calibrated fine-tuned Qwen3.8-27B into the fine-tuned
Qwen3.5-0.8B. It is a normal training run with a `distillation` section in
the config; training builds missing teacher soft labels automatically.
You can also build them first with the cache task and a distillation config:

```bash
# optional: cache the teacher option probabilities (soft labels)
uv run python -m src.modal.cache configs/distillation/qwen3_8_27b.yml

# build missing soft labels, then continue the fine-tuned student's LoRA
uv run python -m src.training.train configs/train/qwen3_5_0_8b_distilled_27b.yml
```

The teacher reads `runs/qwen3_8_27b/final` and uses its saved calibration.
The student loads the config's `backbone` and its trained LoRA from
`adapter`. Both models can use raw HF backbones (e.g. `Qwen/Qwen3.5-0.8B`).
A raw teacher has temperature 1 (no calibration). Fine-tuned model directories
must include their processors. Generation saves only soft labels and metadata
under `datasets/qwen3_8_27b/soft_labels`; images stay in the original HF
dataset. The `distillation` section sets `teacher`, `soft_labels` (the cache
directory), `alpha`, `temperature` and `generating` (the teacher's prediction
arguments). The distillation config for the cache task has only `data` and
`distillation`. Valid soft labels are reused without loading the teacher.
Loading checks dataset fingerprints, split expressions, question identities,
and option order. Use a new directory when the teacher or source data
changes. Failed exports do not publish a partial cache and can be retried.

The loss is `alpha * CE(human) + (1 - alpha) * T^2 * KL(teacher || student)`.
Defaults are `distillation.alpha: 0.25` and `distillation.temperature: 1.0`.
`T` is extra softening after teacher calibration. Human soft targets, such
as vote counts, remain separate. Choice shuffling moves both targets with
their options. The teacher is released before student training starts.

Validation and final calibration use human targets. The calibration split
is carved out before training with the original seed and fraction, so the
fine-tuned student's held-out records remain held out. Merged student
models are saved under their run's `final/`.

Run the same commands on either cloud with the tasks `cache` (distillation
config) and `train` (training config):

```bash
uv run python -m src.runpod.app --task cache \
    --config configs/distillation/qwen3_8_27b.yml --gpu "NVIDIA H200"
uv run python -m src.runpod.app --task train \
    --config configs/train/qwen3_5_0_8b_distilled_27b.yml

modal run src/modal/app.py --task cache \
    --config configs/distillation/qwen3_8_27b.yml --gpu H200
modal run --detach src/modal/app.py --task train \
    --config configs/train/qwen3_5_0_8b_distilled_27b.yml
```

Place both fine-tuned models on that cloud's persistent `runs` volume. The
soft labels are saved on its `datasets` volume.
Regular model/dataset caching uses CPU; caching teacher soft labels uses GPU.
Both use the same Modal cache function with different resource options.
If `train` must build soft labels, its GPU must fit the teacher too. Pre-cache
on a larger GPU to train the student on a smaller GPU afterward.
Resume the student's adapter with
`--checkpoint runs/qwen3_5_0_8b_distilled/checkpoint-<step>`, keeping the
original `backbone` and cache unchanged.

Test the baseline and distilled models on `test` and `test_ood`.
Use the existing test config with `--model`, `--data.split`, and a distinct
`--testing.output_dir` per run. Compare accuracy, Brier, ECE, and latency
on matching hardware. No benchmark gain is assumed before these runs.
See [the distillation spec](docs/DISTILLATION.md) for the cache and loss.

### Modal (cloud GPUs)
The same tasks run on [Modal](https://modal.com). This needs `modal setup`
and a Secret `huggingface` with `HF_TOKEN`.

```bash
modal secret create huggingface HF_TOKEN=<token>

# download the backbone and dataset once, into persistent Volumes
modal run src/modal/app.py --task cache --config configs/train/qwen3_8_27b.yml

# train (keeps running if you close the terminal)
modal run --detach src/modal/app.py --task train --config configs/train/qwen3_8_27b.yml

# calibrate or test
modal run src/modal/app.py --task test --config configs/test/qwen3_8_27b.yml

# TensorBoard on the training logs (prints the URL)
modal deploy src/modal/tensorboard_app.py
```

The GPU comes from the config's `modal.gpu` (default H200). `--gpu` wins over
it, and `--overrides "..."` passes config overrides.

### RunPod (cloud GPUs)
The same tasks run on [RunPod](https://www.runpod.io) with prepaid credits.
Each task gets its own pod: the pod clones this repository at a commit, runs
the task and then deletes itself. Results stay on a network volume. A network
volume works with one running pod at a time, so run one task at a time.

Setup, once:
- create a network volume (Secure Cloud, in a data center with the GPU)
- create a RunPod secret `huggingface` with your HF token
- set `RUNPOD_API_KEY` and `RUNPOD_VOLUME_ID` in your shell

```bash
# download the backbone and dataset once, into the network volume
uv run python -m src.runpod.app --task cache --config configs/train/qwen3_8_27b.yml

# train (prints the TensorBoard URL)
uv run python -m src.runpod.app --task train --config configs/train/qwen3_8_27b.yml

# calibrate or test
uv run python -m src.runpod.app --task test --config configs/test/qwen3_8_27b.yml
```

The pod runs the local `HEAD` commit, so push it first (`--ref` picks
another commit). The GPU comes from the config's `runpod.gpu` (default
`NVIDIA H200`). `--gpu` wins over it, and `--overrides "..."` passes config
overrides. Task logs go to `runs/logs/<pod_id>.log` on the network volume.

## Configs
- `configs/train/`: training runs (`backbone`, `seed`, `lora`, `data`,
  `training`, optional `calibration`)
- `configs/calibration/`: standalone calibration (`model`, `seed`, `data`,
  `calibration`, `calibrating`)
- `configs/test/`: test runs (`model`, optional `temperature`, `data`,
  `testing`, optional `system_two`)
- `configs/distillation/`: soft label caching configs (`data`,
  `distillation`); a training config can also have a `distillation` section

The `training`, `calibrating`, `testing`, and `generating` sections take any
Hugging Face `TrainingArguments` key.

## Project layout
```
src/
  model/        System One and System Two models, config, loader
  common/       request parsing, question types, prompt, config parser
  data/         dataset, collator, Hub loader
  pipeline/     inference pipeline (/v1/systemone)
  evaluation/   evaluator and metrics
  training/     LoRA training
  distillation/ offline teacher targets and student distillation
  calibration/  temperature calibration
  testing/      test script
  modal/        Modal apps (tasks and TensorBoard)
  runpod/       RunPod launcher (tasks, TensorBoard in the train pod)
configs/        YAML run configs
```

## Results
On `Jacqkues/kev-vision-decisions-full`:

| Backbone | Model | Mode | Test set | Accuracy | Brier | ECE | Seconds/question |
|---|---|---|---|---|---|---|---|
| Qwen3.5-0.8B | zero-shot | System Two | `test` | 49.7% | 0.623 | 0.266 | 2.156 |
| Qwen3.5-0.8B | zero-shot | System One | `test` | 52.1% | 0.512 | 0.100 | 0.078 |
| Qwen3.5-0.8B | fine-tuned | System One | `test` | 75.4% | 0.262 | 0.031 | 0.083 |
| Qwen3.5-0.8b-distil-Qwen3.8-27b | distilled (base student) | System One | `test` | 76.9% | 0.259 | 0.042 | 0.081 |
| Qwen3.5-0.8b-distil-Qwen3.8-27b | distilled (fine-tuned student) | System One | `test` | 77.9% | 0.255 | 0.059 | 0.077 |
| Qwen3.8-27B | zero-shot | System Two | `test` | 62.9% | 0.497 | 0.232 | 4.857 |
| Qwen3.8-27B | zero-shot | System One | `test` | 65.2% | 0.427 | 0.122 | 0.129 |
| Qwen3.8-27B | fine-tuned | System One | `test` | 79.2% | 0.218 | 0.037 | 0.194 |
| Qwen3.5-0.8B | zero-shot | System Two | `test_ood` | 60.4% | 0.477 | 0.143 | 2.293 |
| Qwen3.5-0.8B | zero-shot | System One | `test_ood` | 58.6% | 0.505 | 0.049 | 0.088 |
| Qwen3.5-0.8B | fine-tuned | System One | `test_ood` | 68.3% | 0.414 | 0.031 | 0.070 |
| Qwen3.5-0.8b-distil-Qwen3.8-27b | distilled (base student) | System One | `test_ood` | 67.9% | 0.399 | 0.028 | 0.090 |
| Qwen3.5-0.8b-distil-Qwen3.8-27b | distilled (fine-tuned student) | System One | `test_ood` | 70.3% | 0.415 | 0.083 | 0.063 |
| Qwen3.8-27B | zero-shot | System Two | `test_ood` | 82.2% | 0.212 | 0.075 | 3.977 |
| Qwen3.8-27B | zero-shot | System One | `test_ood` | 74.5% | 0.338 | 0.046 | 0.147 |
| Qwen3.8-27B | fine-tuned | System One | `test_ood` | 89.4% | 0.158 | 0.015 | 0.157 |

Both distilled models use the fine-tuned Qwen3.8-27B teacher. The labels
refer to the student's starting weights: the pretrained base model or an
adapter already fine-tuned on the task.

`test_ood` holds sources not seen in training.

Accuracy against time per question, one graph per test set:

![Accuracy vs seconds per question on test](assets/accuracy_vs_time_test.png)

![Accuracy vs seconds per question on test_ood](assets/accuracy_vs_time_test_ood.png)

## Limits
- Max 26 options per question.
- One GPU only, no multi-GPU training yet.
- Images only, no video.
- Each question of a request is a separate forward pass (no shared KV cache).

## License

The original source code in this repository is licensed under the
[Apache License 2.0](LICENSE).

Datasets and model weights, including trained adapters and checkpoints, are
excluded from this code license and require separate license terms. The
`Jacqkues/kev-vision-decisions-full` dataset requires non-commercial research
use and compliance with each source's license; see its
[dataset card](https://huggingface.co/datasets/Jacqkues/kev-vision-decisions-full).
