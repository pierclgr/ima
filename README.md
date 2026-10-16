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
    "media": [],
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

### Offline distillation
Distill the calibrated fine-tuned Qwen3.8-27B into the fine-tuned
Qwen3.5-0.8B. It is a normal training run with a `distillation` section in
the config; training builds missing teacher soft labels automatically.
You can also build them first with the cache task and a distillation config:

```bash
# optional: cache the teacher option probabilities (soft labels)
uv run python -m src.modal.cache configs/distillation/qwen3_8_27b.yml

# build missing soft labels, then train a new LoRA on the fine-tuned student
uv run python -m src.training.train configs/train/qwen3_5_0_8b_distilled.yml

# control: the same student and extra epoch, using human targets only
uv run python -m src.training.train configs/train/qwen3_5_0_8b_control.yml
```

The teacher reads `runs/qwen3_8_27b/final` and uses its saved calibration.
The student is the config's `backbone`, `runs/qwen3_5_0_8b/final`. Both can
also be raw HF backbones (e.g. `Qwen/Qwen3.5-0.8B`), as in training; a raw
teacher has temperature 1 (no calibration). Fine-tuned model directories must
include their processors. Generation saves only soft labels and metadata
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
fine-tuned student's held-out records remain held out. The student and
control have matched settings and separate output directories. Their
merged models are saved under their run's `final/`.

Run the same commands on either cloud with the tasks `cache` (distillation
config) and `train` (training config):

```bash
uv run python -m src.runpod.app --task cache \
    --config configs/distillation/qwen3_8_27b.yml --gpu "NVIDIA H200"
uv run python -m src.runpod.app --task train \
    --config configs/train/qwen3_5_0_8b_distilled.yml

modal run src/modal/app.py --task cache \
    --config configs/distillation/qwen3_8_27b.yml --gpu H200
modal run --detach src/modal/app.py --task train \
    --config configs/train/qwen3_5_0_8b_distilled.yml
```

Place both fine-tuned models on that cloud's persistent `runs` volume. The
soft labels are saved on its `datasets` volume.
Regular model/dataset caching uses CPU; caching teacher soft labels uses GPU.
Both use the same Modal cache function with different resource options.
If `train` must build soft labels, its GPU must fit the teacher too. Pre-cache
on a larger GPU to train the student on a smaller GPU afterward.
Resume the student's new adapter with
`--checkpoint runs/qwen3_5_0_8b_distilled/checkpoint-<step>`, keeping the
original fine-tuned `backbone` and cache unchanged.

Test the baseline, control, and distilled model on `test` and `test_ood`.
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
| Qwen3.8-27B | zero-shot | System Two | `test` | 62.9% | 0.497 | 0.232 | 4.857 |
| Qwen3.8-27B | zero-shot | System One | `test` | 65.2% | 0.427 | 0.122 | 0.129 |
| Qwen3.8-27B | fine-tuned | System One | `test` | 79.2% | 0.218 | 0.037 | 0.194 |
| Qwen3.5-0.8B | zero-shot | System Two | `test_ood` | 60.4% | 0.477 | 0.143 | 2.293 |
| Qwen3.5-0.8B | zero-shot | System One | `test_ood` | 58.6% | 0.505 | 0.049 | 0.088 |
| Qwen3.5-0.8B | fine-tuned | System One | `test_ood` | 68.3% | 0.414 | 0.031 | 0.070 |
| Qwen3.8-27B | zero-shot | System Two | `test_ood` | 82.2% | 0.212 | 0.075 | 3.977 |
| Qwen3.8-27B | zero-shot | System One | `test_ood` | 74.5% | 0.338 | 0.046 | 0.147 |
| Qwen3.8-27B | fine-tuned | System One | `test_ood` | 89.4% | 0.158 | 0.015 | 0.157 |

`test_ood` holds sources not seen in training.

Accuracy against time per question, one graph per test set:

![Accuracy vs seconds per question on test](assets/accuracy_vs_time_test.png)

![Accuracy vs seconds per question on test_ood](assets/accuracy_vs_time_test_ood.png)

Seconds/question is the model's time only (System One: the forward pass;
System Two: the generation up to the final answer, thinking off), without
data loading. All times come from the same setup: Modal, NVIDIA H100 80GB
(pinned), batch 1, 4 data workers, 8 reserved CPU cores, runs one after the
other, 200 random records per test set (about 265 questions on `test`, 200 on
`test_ood`), after 10 warm-up questions. Accuracy, Brier and ECE come from
runs on the full test sets.

## Limits
- Max 26 options per question.
- One GPU only, no multi-GPU training yet.
- Images only, no video.
- Each question of a request is a separate forward pass (no shared KV cache).
