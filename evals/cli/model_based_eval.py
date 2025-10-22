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

# type: ignore

import gc
import inspect
import os
import shlex
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

import numpy as np
import torch

from evals.params import parse_args  # from_cli, to_cli
from evals.predictors import build_predictor
from evals.predictors.llm_inference import (
    LLMInferencePredictor,
    LLMInferencePredictorConfig,
)
from evals.predictors.llm_inference_reward import (
    LLMInferenceRewardScorer,
    RewardScorerConfig,
)
from evals.tasks import build_task, TaskConfig, TaskRegistry
from evals.tasks.base import ModelBasedEvalTaskConfig
from evals.utils import (
    filter_by_pattern,
    flatten_dict,
    format_dict,
    get_dp_rank,
    get_git_info,
    get_global_rank,
    get_mp_rank,
    get_random_state,
    initialize_logger,
    mp_rank_zero_info,
    rank_zero_info,
    setup_env,
    write_to_json,
)
from evals.utils.distributed import reinit_model_parallel


@dataclass
class SamplingConfig:
    temperature: float = 0.0
    top_k: int = 0
    top_p: float = 1.0

    def __post_init__(self):
        assert self.temperature >= 0.0, self.temperature


def release_memory(obj: Any):
    del obj
    gc.collect()
    torch.cuda.empty_cache()


@dataclass
class ModelBasedEvalConfig:
    dataset_dir: str
    tasks: str

    generator: LLMInferencePredictorConfig
    llama_judge: Optional[LLMInferencePredictorConfig] = None
    reward_judge: Optional[RewardScorerConfig] = None

    generator_sampling_config: Optional[SamplingConfig] = field(
        default_factory=lambda: SamplingConfig()
    )
    judge_sampling_config: Optional[SamplingConfig] = field(
        default_factory=lambda: SamplingConfig()
    )

    seed: int = 42
    datasets: Optional[str] = None
    dump_dir: Optional[str] = None
    max_samples: Optional[int] = None
    show_progress: bool = True


def main(cfg: ModelBasedEvalConfig):
    setup_env()
    torch.manual_seed(cfg.seed)

    if cfg.dump_dir is not None and get_global_rank() == 0:
        metadata = {
            "timestamp": datetime.now().strftime("%Y_%m_%d_%H_%M_%S"),
            "command": " ".join(map(shlex.quote, sys.argv)),
            "git_info": get_git_info(),
            "config": asdict(cfg),
            "task_configs": cfg.task_configs,
        }
        write_to_json(metadata, os.path.join(cfg.dump_dir, "metadata.json"), indent=4)

    rank_zero_info(f"Config: {asdict(cfg)}")
    names: List[str] = filter_by_pattern(TaskRegistry.names(), cfg.tasks)
    assert len(names) > 0, f"No tasks were found given the pattern '{cfg.tasks}'"
    rank_zero_info(f"Selected tasks for execution: {names}")

    task_configs: Dict[str, TaskConfig] = {}
    for name in names:
        parameters = inspect.signature(TaskRegistry._REGISTRY[name]).parameters
        kwargs: Dict[str, Any] = {}
        if "dataset_dir" in parameters:
            kwargs["dataset_dir"] = cfg.dataset_dir
        if "datasets" in parameters:
            kwargs["datasets"] = cfg.datasets
        task_configs[name] = TaskRegistry.get_config(name, **kwargs)

    assert all(
        isinstance(task, ModelBasedEvalTaskConfig) for task in task_configs.values()
    )

    if isinstance(cfg.generator, LLMInferencePredictorConfig):
        cfg.generator.max_total_len = max(  # type: ignore
            1
            + getattr(t.generator_task_config, "max_prompt_len", 0)  # type: ignore
            + getattr(t.generator_task_config, "max_gen_len", 0)
            for t in task_configs.values()
        )

    # Step-1: perform generation from the LLM to be evaluated
    generator = build_predictor(cfg.generator)
    generator_outputs: Dict[str, Any] = {}

    for name, task_config in task_configs.items():
        task = build_task(task_config)
        start = time.monotonic()
        rank_zero_info(f"Running evaluation on task {name}")
        generator_output = task.run(
            predictor=generator,
            top_p=cfg.generator_sampling_config.top_p,
            top_k=cfg.generator_sampling_config.top_k,
            output_from_generator=None,
            temperature=cfg.generator_sampling_config.temperature,
            random_state=get_random_state(cfg.seed),
            max_samples=cfg.max_samples,
            show_progress=cfg.show_progress,
        )
        generator_outputs[name] = generator_output

    # Destruct generator to free GPU MEM for the Judge and update MP
    rank_zero_info("Generation done - destructing generator to free GPUs")
    release_memory(generator)
    reinit_model_parallel()

    # Step-2: leverage another model as judge for evaluation
    if cfg.llama_judge is not None:
        assert isinstance(cfg.llama_judge, LLMInferencePredictorConfig)
        assert cfg.reward_judge is None
        cfg.llama_judge.max_total_len = max(  # type: ignore
            1
            + getattr(t.judge_task_config, "max_prompt_len", 0)  # type: ignore
            + getattr(t.judge_task_config, "max_gen_len", 0)
            for t in task_configs.values()
        )
        judge = LLMInferencePredictor.from_config(cfg.llama_judge)
    else:
        assert isinstance(cfg.reward_judge, RewardScorerConfig)
        assert cfg.reward_judge is not None
        cfg.reward_judge.max_total_len = max(  # type: ignore
            1
            + getattr(t.judge_task_config, "max_prompt_len", 0)  # type: ignore
            + getattr(t.judge_task_config, "max_gen_len", 0)
            for t in task_configs.values()
        )
        judge = LLMInferenceRewardScorer.from_config(cfg.reward_judge)

    metrics = {}
    for name, task_config in task_configs.items():
        f_name = name
        if name == "reward":
            reward_model = cfg.reward_judge.checkpoint_dir.split("/")[-1]
            f_name += f".{reward_model}.{cfg.datasets}"
        task = build_task(task_config)
        start = time.monotonic()
        rank_zero_info(f"Running evaluation on task {name}")
        result = task.judge(
            predictor=judge,
            top_p=cfg.judge_sampling_config.top_p,
            top_k=cfg.judge_sampling_config.top_k,
            output_from_generator=generator_outputs[name],
            temperature=cfg.judge_sampling_config.temperature,
            random_state=np.random.RandomState(cfg.seed),
            max_samples=cfg.max_samples,
            show_progress=cfg.show_progress,
        )
        metrics[name] = {k: v.value for k, v in result.metrics.items()}
        log = format_dict(flatten_dict(metrics[name]), delimiter=" | ", decimal=6)
        rank_zero_info(f"Evaluation results on task {name}: {log}")

        if cfg.dump_dir is not None and get_mp_rank() == 0:
            rank = str(get_dp_rank())
            raw_file = os.path.join(cfg.dump_dir, "raw_results", rank, f"{f_name}.json")
            mp_rank_zero_info(f"Writing raw results to {raw_file}")
            write_to_json(result.raw_results, raw_file)

            if get_global_rank() == 0:
                result_file = os.path.join(cfg.dump_dir, "results", f"{f_name}.json")
                rank_zero_info(f"Writing metric results to {result_file}")
                write_to_json(metrics[name], result_file, sort_keys=True, indent=4)
        rank_zero_info(f"Task {name} took {time.monotonic() - start:.2f} seconds")


if __name__ == "__main__":
    cfg: ModelBasedEvalConfig = parse_args(ModelBasedEvalConfig)
    initialize_logger()
    main(cfg)
