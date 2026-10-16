"""Builds missing offline teacher targets for caching or student training."""

import gc
from pathlib import Path
from tempfile import TemporaryDirectory

import torch
from transformers import TrainingArguments

from src.common.config_parser import ConfigParser
from src.common.prompt import PromptBuilder
from src.data.config import DataSettings
from src.data.hub import HubRecordLoader
from src.distillation.cache import TeacherTargetCache
from src.distillation.config import DistillationSettings
from src.evaluation.evaluator import DecisionEvaluator
from src.model.loader import ModelLoader


def ensure_targets(
    settings: DistillationSettings, data: DataSettings,
) -> TeacherTargetCache:
    """Reuses valid soft labels or builds them before the student is loaded.

    Generation publishes a completed directory. The teacher and evaluator
    are released before returning, including when generation fails.
    """
    path = Path(settings.soft_labels)
    loader = HubRecordLoader(data.dataset)
    expected_splits = {"train": data.train}
    if data.validation is not None:
        expected_splits["validation"] = data.validation
    if (path / "manifest.json").is_file():
        cache = TeacherTargetCache(path)
        if cache.manifest["teacher"] != settings.teacher:
            raise ValueError("cached targets do not match the teacher")
        for name, expression in expected_splits.items():
            cache.attach(loader.load(expression), data.dataset, name,
                         expression, cache.manifest["max_options"])
        return cache
    if path.exists() and any(path.iterdir()):
        raise ValueError("target cache is incomplete; use a new directory")
    path.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=f".{path.name}-", dir=path.parent) as temp:
        model, processor = ModelLoader.load(settings.teacher)
        evaluator = None
        try:
            evaluator = DecisionEvaluator(
                model, PromptBuilder(processor, model.config.max_options),
                ConfigParser.parse(TrainingArguments, {
                    "report_to": "none", **settings.generating,
                    "output_dir": str(Path(temp) / "predict"),
                }),
            )
            splits = {}
            for name, expression in expected_splits.items():
                records = loader.load(expression)
                output = evaluator.predict(records)
                splits[name] = (expression, records, output.predictions[1])
                print(f"{name} teacher questions: {len(output.label_ids)}")
            generated = Path(temp) / "targets"
            TeacherTargetCache.write(
                generated, settings.teacher, model.config.temperature,
                data.dataset, model.config.max_options, splits,
            )
            generated.replace(path)
        finally:
            del evaluator, model, processor
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    print(f"teacher targets: {path}")
    return TeacherTargetCache(path)
