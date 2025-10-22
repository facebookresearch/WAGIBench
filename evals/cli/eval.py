# Copyright (c) Meta Platforms, Inc. and affiliates.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import inspect
import json

import os
import shlex
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from functools import cached_property
from pathlib import Path
from typing import Any, Dict, Optional

import torch
from evals.api import PredictorConfig, TaskConfig
from evals.params import from_cli, parse_args, to_cli
from evals.predictors import build_predictor, PredictorRegistry
from evals.tasks import build_task, PerplexityTaskConfig, TaskRegistry
from evals.utils import (
    filter_by_pattern,
    flatten_dict,
    format_dict,
    get_checkpoint_step,
    get_git_info,
    get_global_rank,
    get_job_info,
    get_mp_rank,
    get_random_state,
    get_version,
    hive,
    initialize_logger,
    rank_zero_info,
    setup_env,
    unroll_configs,
    write_to_json,
    write_to_tensorboard,
)


@dataclass
class EvalConfig:
    dataset_dir: str
    tasks: Optional[str] = None
    task_args: Optional[Dict[str, Any]] = None
    ppl: Optional[PerplexityTaskConfig] = None

    predictor: str = "llm_inference"
    predictor_config: Optional[PredictorConfig] = None

    temperature: float = 0.0
    top_k: int = 0
    top_p: float = 1.0
    seed: int = 42
    confidence_level: Optional[float] = None

    dump_dir: Optional[str] = None
    metric_log_dir: Optional[str] = None
    tb_log_dir: Optional[str] = None
    no_resume: Optional[bool] = False

    max_samples: Optional[int] = None
    show_progress: bool = False

    record_to_hive: bool = True

    def __post_init__(self) -> None:
        assert self.predictor in PredictorRegistry.names(), (
            f"Predictor {self.predictor} not available in {PredictorRegistry.names()}",
        )
        self.metric_log_dir = self.metric_log_dir or self.dump_dir
        if self.tb_log_dir is None and self.metric_log_dir is not None:
            self.tb_log_dir = os.path.join(self.metric_log_dir, "tb")
        assert self.temperature >= 0.0, self.temperature
        if self.temperature == 0:
            self.top_p = 0
            self.top_k = 0

    @cached_property
    def task_configs(self) -> Dict[str, TaskConfig]:
        configs: Dict[str, TaskConfig] = {"ppl": self.ppl} if self.ppl else {}
        for name in filter_by_pattern(TaskRegistry.names(), self.tasks):
            defaults = inspect.signature(TaskRegistry._REGISTRY[name]).parameters
            params = (self.task_args or {}).get(name, {})
            for fname, kwargs in unroll_configs(defaults, params, prefix=name).items():
                if "dataset_dir" in defaults:
                    kwargs["dataset_dir"] = self.dataset_dir
                configs[fname] = TaskRegistry.get_config(name, **kwargs)
        return configs


def main(cfg: EvalConfig) -> None:
    setup_env()
    torch.manual_seed(cfg.seed)

    if cfg.dump_dir is not None and get_global_rank() == 0:
        metadata = {
            "timestamp": datetime.now().strftime("%Y_%m_%d_%H_%M_%S"),
            "command": " ".join(map(shlex.quote, sys.argv)),
            "job_info": get_job_info(),
            "git_info": get_git_info(),
            "config": asdict(cfg),
            "task_configs": cfg.task_configs,
        }
        metadata_file = os.path.join(cfg.dump_dir, "metadata.jsonl")
        rank_zero_info(f"Writing configs and metadata to {metadata_file}")
        write_to_json(metadata, metadata_file, mode="a", ending="\n")

    rank_zero_info(f"Evals version {get_version()} ({Path(__file__).parent.parent})")
    rank_zero_info(f"Config: {asdict(cfg)}")
    assert cfg.task_configs, f"No tasks were found given pattern '{cfg.tasks}'"
    rank_zero_info(f"Selected tasks for execution: {list(cfg.task_configs)}")

    assert cfg.predictor_config is not None
    if cfg.predictor in {"llm_inference", "llm_inference_reward"}:
        cfg.predictor_config.max_total_len = max(  # type: ignore
            1 + getattr(t, "max_prompt_len", 0) + getattr(t, "max_gen_len", 0)
            for t in cfg.task_configs.values()
        )
    predictor = build_predictor(cfg.predictor_config)
    step = get_checkpoint_step(getattr(cfg.predictor_config, "checkpoint_dir", None))

    metrics: Dict[str, Dict[str, float]] = {}
    for name, task_config in cfg.task_configs.items():
        # Simplify name if it contains slashes, e.g., ppl.files = "/x/y/z,/a/b"
        if "/" in set(name):
            name = "_".join(s.split("/")[-1] for s in name.split(","))
        start = time.monotonic()
        rank_zero_info(f"Running evaluation on task {name}")
        if cfg.dump_dir is not None:
            result_file = os.path.join(cfg.dump_dir, "results", f"{name}.json")
            if not cfg.no_resume and os.path.exists(result_file):
                rank_zero_info(f"Loading cached evaluation results from {result_file}")
                with open(result_file) as f:
                    metrics[name] = json.load(f)["results"]

        if name not in metrics:
            task = build_task(task_config)
            result = task.run(
                predictor=predictor,
                top_p=cfg.top_p,
                top_k=cfg.top_k,
                temperature=cfg.temperature,
                random_state=get_random_state(cfg.seed),
                max_samples=cfg.max_samples,
                show_progress=cfg.show_progress,
            )
            # Sync task completion across all ranks in a mp group. This ensures each task completes on all ranks before the next task starts on any rank
            # Otherwise race conditions arise where non-rank_0 processes start the next task before rank_0 has completed the first task.
            # This has been observed to cause timeouts in pytorch distributed operations as other rank 0's are unexpectedly waiting for rank_0 to complete the first task
            if torch.distributed.is_initialized():
                torch.distributed.barrier()

            metrics[name] = {k: v.value for k, v in result.metrics.items()}

            # Compute confidence intervals
            if cfg.confidence_level is not None:
                for key, avg_metric in result.metrics.items():
                    ci_lb, ci_ub = avg_metric.compute_ci(cfg.confidence_level)
                    if ci_lb and ci_ub:
                        metrics[name][f"{key}_ci_lb_{cfg.confidence_level}"] = ci_lb
                        metrics[name][f"{key}_ci_ub_{cfg.confidence_level}"] = ci_ub

            if cfg.dump_dir is not None and get_mp_rank() == 0:
                raw_file = os.path.join(
                    cfg.dump_dir, "raw_results", str(get_global_rank()), f"{name}.json"
                )
                rank_zero_info(f"Writing raw results to {raw_file}")
                write_to_json(result.raw_results, raw_file)

                if get_global_rank() == 0:
                    result_content = {"results": metrics[name], "configs": task_config}
                    rank_zero_info(f"Writing metric results to {result_file}")
                    write_to_json(result_content, result_file, indent=4)
                    if cfg.tb_log_dir:
                        rank_zero_info(f"Writing Tensorboard logs to {cfg.tb_log_dir}")
                        task_results = flatten_dict({name: metrics[name]})
                        write_to_tensorboard(
                            task_results, cfg.tb_log_dir, step, tag="eval"
                        )

        log = format_dict(flatten_dict(metrics[name]), delimiter=" | ", decimal=6)
        rank_zero_info(f"Evaluation results on task {name}: {log}")
        rank_zero_info(f"Task {name} took {time.monotonic() - start:.2f} seconds")
        torch.cuda.empty_cache()

    results = flatten_dict(metrics)
    rank_zero_info(f"All evaluation results: {format_dict(results)}")
    if cfg.metric_log_dir and get_global_rank() == 0:
        metric_log_path = os.path.join(cfg.metric_log_dir, "metrics.eval.jsonl")
        rank_zero_info(f"Writing metric logs to {metric_log_path}")
        timestamp = {"global_step": step, "created_at": datetime.utcnow().isoformat()}
        write_to_json(timestamp | results, metric_log_path, mode="a", ending="\n")

        if cfg.record_to_hive and cfg.dump_dir:
            hive.record_to_hive(
                dirs=[os.path.join(cfg.dump_dir, "results")], files=[metadata_file]
            )


def cfg_from_cli() -> EvalConfig:
    known, unknown = to_cli(EvalConfig).parse_known_args()
    cfg: EvalConfig = from_cli(EvalConfig, vars(known), allow_incomplete=True)
    cfg.predictor_config = parse_args(
        PredictorRegistry.get_config_cls(cfg.predictor), unknown
    )
    return cfg


if __name__ == "__main__":
    cfg: EvalConfig = cfg_from_cli()
    initialize_logger()
    main(cfg)
