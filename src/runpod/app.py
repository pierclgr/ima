"""RunPod launcher: caching, training, calibration and testing on cloud GPUs.

Creates one pod per task. The pod clones this repository at a commit, runs
the task and then deletes itself. Its network volume, mounted at
`/workspace`, keeps big data between pods:
- `models`: HF Hub cache (backbones)
- `datasets`: HF datasets cache (raw files and prepared splits) and the
  distillation soft labels, linked as the repository's `datasets`, so the
  configs' `datasets/...` paths land on it
- `runs`: run outputs (checkpoints, `final/`, test metrics, task logs in
  `runs/logs/<pod_id>.log`), linked as the repository's `runs`, so the
  configs' `runs/...` paths land on it
- `uv`: uv cache, so later pods install the environment faster

A network volume serves one running pod at a time, so run one task at a
time. A `train` pod also serves TensorBoard on the training logs (port 6006,
URL printed).

Needs, once: a network volume (Secure Cloud, in a data center with the
config's GPU) and a RunPod secret `huggingface` with the HF token (gated
dataset). Locally: the `RUNPOD_API_KEY` and `RUNPOD_VOLUME_ID` environment
variables. The commit must be pushed: the pod clones it from GitHub. Run
from the repository root:

    uv run python -m src.runpod.app --task cache \\
        --config configs/train/qwen3_8_27b.yml
    uv run python -m src.runpod.app --task train \\
        --config configs/train/qwen3_8_27b.yml
    uv run python -m src.runpod.app --task test \\
        --config configs/test/qwen3_8_27b.yml

GPU: the config's `runpod.gpu` (default `NVIDIA H200`), e.g.

    runpod:
      gpu: NVIDIA H100 80GB HBM3

Options: `--overrides "--training.learning_rate 5e-5"` (config overrides),
`--gpu ID` (instead of the config's GPU), `--ref COMMIT` (instead of the
local `HEAD`).
"""

import argparse
import json
import os
import shlex
import subprocess
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import yaml

from src.modal.cache import cache_uses_gpu

API: str = "https://api.runpod.io/v2"
REPOSITORY: str = "https://github.com/pierclgr/ima.git"
# has git and curl
IMAGE: str = "ghcr.io/astral-sh/uv:0.11.21-python3.13-trixie"
VOLUME: str = "/workspace"
CODE: str = "/root/ima"
# gpu of the tasks when the config's `runpod` section names none
GPU: str = "NVIDIA H200"
# cpu pod for regular model/data caching: 4 vCPUs, 16 GB RAM
CPU: dict[str, str | int] = {"id": "cpu5g", "vcpuCount": 4}
# container disk in GB (the code and its environment)
DISK: int = 50
TENSORBOARD_PORT: int = 6006
MODULES: dict[str, str] = {
    "cache": "src.modal.cache",
    "train": "src.training.train",
    "calibrate": "src.calibration.calibrate",
    "test": "src.testing.test",
}


def api(method: str, path: str, key: str, body: dict | None = None) -> dict:
    """Calls the RunPod REST API.

    Args:
        method: HTTP method.
        path: Path after the API base, e.g. `/pods`.
        key: RunPod API key.
        body: JSON body, if any.

    Returns:
        The JSON response (empty for no content).

    Raises:
        RuntimeError: If the API answers with an error.
    """
    request = Request(
        f"{API}{path}",
        method=method,
        data=None if body is None else json.dumps(body).encode(),
        headers={
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
            # cloudflare blocks urllib's default user agent (error 1010)
            "User-Agent": "ima",
        },
    )
    try:
        with urlopen(request) as response:
            content: bytes = response.read()
    except HTTPError as error:
        message = f"{method} {path}: {error.code} {error.read().decode()}"
        raise RuntimeError(message) from error
    return json.loads(content) if content else {}


def config_gpu(config: str, gpu: str) -> str:
    """Returns the GPU of a run.

    Args:
        config: Config path, with an optional `runpod` section.
        gpu: GPU from the command line, empty if not given.

    Returns:
        `gpu` if given, else the config's `runpod.gpu`, else `GPU`.
    """
    values: dict = yaml.safe_load(Path(config).read_text()) or {}
    return gpu or (values.get("runpod") or {}).get("gpu") or GPU


def start_command(task: str, argv: list[str], ref: str) -> str:
    """Returns the pod's shell script.

    Clones the repository at `ref`, installs the environment, runs the task
    (with TensorBoard for `train`), then deletes the pod, also on failure.

    Args:
        task: `cache`, `train`, `calibrate` or `test`.
        argv: Task arguments (config path and overrides).
        ref: Commit to run.

    Returns:
        The script, for `bash -c`.
    """
    log = f"{VOLUME}/runs/logs/$RUNPOD_POD_ID.log"
    steps = [
        f"git clone {REPOSITORY} {CODE}",
        f"cd {CODE}",
        f"git checkout {shlex.quote(ref)}",
        f"ln -s {VOLUME}/runs runs",
        f"ln -s {VOLUME}/datasets datasets",
        "uv sync",
    ]
    if task == "train":
        port = TENSORBOARD_PORT
        tensorboard = f"uv run tensorboard --logdir runs --bind_all --port {port}"
        # own log: on the task's pipe it would keep `tee` open after the task
        tensorboard_log = f"{VOLUME}/runs/logs/$RUNPOD_POD_ID.tensorboard.log"
        steps.append(f"{{ {tensorboard} > {tensorboard_log} 2>&1 & }}")
    steps.append(shlex.join(["uv", "run", "python", "-m", MODULES[task], *argv]))
    delete = (
        'curl -fsS -X DELETE -H "Authorization: Bearer $RUNPOD_API_KEY" '
        f"{API}/pods/$RUNPOD_POD_ID"
    )
    return "\n".join(
        [
            f"mkdir -p {VOLUME}/runs/logs",
            f"( {' && '.join(steps)} ) 2>&1 | tee {log}",
            # sleeping stops a container restart from running the task again
            f"{delete} || {{ echo 'pod delete failed'; sleep infinity; }}",
        ]
    )


def pod_request(
    task: str, argv: list[str], ref: str, gpu: str, data_center: str, volume_id: str
) -> dict:
    """Returns the body of the pod creation request.

    Args:
        task: `cache`, `train`, `calibrate` or `test`.
        argv: Task arguments (config path and overrides).
        ref: Commit to run.
        gpu: RunPod GPU id (regular model/data caching uses CPU only).
        data_center: Data center of the network volume.
        volume_id: Network volume id.

    Returns:
        The request body.
    """
    body: dict = {
        "name": f"ima-{task}",
        "image": IMAGE,
        "dataCenterIds": [data_center],
        "mounts": {"network": [{"volumeId": volume_id, "path": VOLUME}]},
        "disk": DISK,
        "env": {
            "HF_TOKEN": "{{ RUNPOD_SECRET_huggingface }}",
            "HF_HUB_CACHE": f"{VOLUME}/models",
            "HF_DATASETS_CACHE": f"{VOLUME}/datasets",
            "UV_CACHE_DIR": f"{VOLUME}/uv",
            # the uv cache is on another file system than the environment
            "UV_LINK_MODE": "copy",
        },
        "entrypoint": ["bash", "-c"],
        "cmd": [start_command(task, argv, ref)],
    }
    if task == "cache" and not cache_uses_gpu(argv[0]):
        body["cpu"] = CPU
    else:
        body["gpu"] = {"id": gpu, "count": 1}
    if task == "train":
        body["ports"] = [f"{TENSORBOARD_PORT}/http"]
    return body


def main() -> None:
    """Creates the pod of a task and prints where to follow it."""
    parser = argparse.ArgumentParser(description="Runs a task on RunPod.")
    parser.add_argument("--task", required=True, choices=sorted(MODULES))
    parser.add_argument(
        "--config",
        required=True,
        help="config path, relative to the repository root (a training config "
        "for cache and train, a calibration config for calibrate, a test "
        "config for test; a distillation config for cache builds the teacher "
        "soft labels)",
    )
    parser.add_argument("--overrides", default="", help='e.g. "--lora.r 8"')
    parser.add_argument("--gpu", default="", help="instead of the config's runpod.gpu")
    parser.add_argument("--ref", default="", help="commit (default: HEAD, pushed)")
    args = parser.parse_args()
    key = os.environ["RUNPOD_API_KEY"]
    volume_id = os.environ["RUNPOD_VOLUME_ID"]
    ref = args.ref or subprocess.run(
        ["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True
    ).stdout.strip()
    data_center = api("GET", f"/network-volumes/{volume_id}", key)["dataCenter"]
    body = pod_request(
        args.task,
        [args.config, *shlex.split(args.overrides)],
        ref,
        config_gpu(args.config, args.gpu),
        data_center,
        volume_id,
    )
    pod_id = api("POST", "/pods", key, body)["id"]
    print(f"pod: {pod_id} (commit {ref}, data center {data_center})")
    print(f"log: {VOLUME}/runs/logs/{pod_id}.log on the network volume")
    if args.task == "train":
        print(f"tensorboard: https://{pod_id}-{TENSORBOARD_PORT}.proxy.runpod.net")


if __name__ == "__main__":
    main()
