"""Modal app: caching, training, calibration and testing on cloud GPUs.

TensorBoard is a separate app (`src/modal/tensorboard_app.py`), so runs of
this app create no web endpoint and can run in parallel.

Persistent Volumes (created on first use), so big data is downloaded once:
- `models`: HF Hub cache (backbones), mounted at `/root/models`
- `datasets`: HF datasets cache (raw files and prepared splits), mounted at
  `/root/datasets`
- `runs`: run outputs (checkpoints, `final/`, test metrics), mounted at
  `/root/runs`, so the configs' `runs/...` paths land on it

Needs `modal setup` and a Secret `huggingface` with `HF_TOKEN` for the gated
dataset (`modal secret create huggingface HF_TOKEN=<token>`). Run from the
repository root:

    modal run src/modal/app.py --task cache \\
        --config configs/train/qwen3_8_27b.yml
    modal run --detach src/modal/app.py --task train \\
        --config configs/train/qwen3_8_27b.yml
    modal run src/modal/app.py --task calibrate \\
        --config configs/calibration/qwen3_8_27b.yml
    modal run src/modal/app.py --task test --config configs/test/qwen3_8_27b.yml

GPU: the config's `modal.gpu` (default H200), e.g.

    modal:
      gpu: H100

Options: `--overrides "--training.learning_rate 5e-5"` (config overrides),
`--gpu H100` (instead of the config's GPU). `--detach` keeps the run going
if the local client disconnects (Ctrl+C stops only the log streaming).

"""

import shlex
from pathlib import Path

import modal
import yaml

from src.modal.cache import cache_uses_gpu

ROOT: str = "/root"
# modal's maximum function timeout
DAY: int = 24 * 60 * 60
# gpu of the tasks when the config's `modal` section names none
GPU: str = "H200"

image = (
    modal.Image.debian_slim(python_version="3.13")
    .uv_sync()
    .env({"HF_HUB_CACHE": f"{ROOT}/models", "HF_DATASETS_CACHE": f"{ROOT}/datasets"})
    .workdir(ROOT)
    .add_local_python_source("src")
    .add_local_dir("configs", remote_path=f"{ROOT}/configs")
)

VOLUMES: dict[str, modal.Volume] = {
    name: modal.Volume.from_name(name, create_if_missing=True)
    for name in ("models", "datasets", "runs")
}

app = modal.App(
    "decision-engine",
    image=image,
    secrets=[modal.Secret.from_name("huggingface")],
    volumes={f"{ROOT}/{name}": volume for name, volume in VOLUMES.items()},
)


@app.function(cpu=4, memory=16384, timeout=DAY)
def cache(argv: list[str]) -> None:
    """Runs `src.modal.cache`; the launcher selects CPU or GPU resources."""
    from src.modal.cache import main

    main(argv)


@app.function(gpu=GPU, timeout=DAY)
def train(argv: list[str]) -> None:
    """Runs `src.training.train` with `argv`."""
    from src.training.train import main

    main(argv)


@app.function(gpu=GPU, timeout=DAY)
def calibrate(argv: list[str]) -> None:
    """Runs `src.calibration.calibrate` with `argv`."""
    from src.calibration.calibrate import main

    main(argv)


# fixed cpu and memory, so test times do not depend on the host's free cores
@app.function(gpu=GPU, cpu=8, memory=32768, timeout=DAY)
def test(argv: list[str]) -> None:
    """Runs `src.testing.test` with `argv`."""
    from src.testing.test import main

    main(argv)


def config_gpu(config: str, gpu: str) -> str:
    """Returns the GPU of a run.

    Args:
        config: Config path, with an optional `modal` section.
        gpu: GPU from the command line, empty if not given.

    Returns:
        `gpu` if given, else the config's `modal.gpu`, else `GPU`.
    """
    values: dict = yaml.safe_load(Path(config).read_text()) or {}
    return gpu or (values.get("modal") or {}).get("gpu") or GPU


TASKS: dict[str, modal.Function] = {
    "cache": cache,
    "train": train,
    "calibrate": calibrate,
    "test": test,
}


@app.local_entrypoint()
def main(task: str, config: str, overrides: str = "", gpu: str = "") -> None:
    """Runs a task on Modal.

    Args:
        task: `cache`, `train`, `calibrate` or `test`.
        config: Config path, relative to the repository root (a training
            config for `cache` and `train`, a calibration config for
            `calibrate`, a test config for `test`; a distillation config
            for `cache` builds the teacher soft labels).
        overrides: Config overrides, e.g. `"--lora.r 8 --checkpoint x"`.
        gpu: GPU instead of the config's `modal.gpu` (e.g. `H100`,
            `A100-80GB`). Ignored by regular model/data caching (CPU only).

    Raises:
        ValueError: If the task is unknown.
    """
    if task not in TASKS:
        raise ValueError(f"task must be one of {sorted(TASKS)}")
    function = (
        TASKS[task]
        if task == "cache"
        else TASKS[task].with_options(gpu=config_gpu(config, gpu))
    )
    if task == "cache" and cache_uses_gpu(config):
        # the 27b teacher is loaded into host RAM before moving to the GPU
        function = function.with_options(
            gpu=config_gpu(config, gpu), memory=65536,
        )
    # a spawned call survives ctrl+c with --detach, a `.remote()` one is cancelled
    function.spawn([config, *shlex.split(overrides)]).get()
