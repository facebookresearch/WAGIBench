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

from dataclasses import dataclass
from typing import Any, Dict, Optional

import torch

from evals.api import (
    AggregationFn,
    AverageMetric,
    Predictor,
    Task,
    TaskConfig,
    TaskResult,
)
from evals.tasks.base.choice import ChoiceTask, ChoiceTaskConfig
from evals.tasks.base.generation import GenerationTask, GenerationTaskConfig
from evals.utils import flatten_dict, format_dict, rank_zero_info
from numpy.random import RandomState


@dataclass
class GroupTaskConfig(TaskConfig):
    tasks: Dict[str, TaskConfig]
    aggregate_fn: Optional[AggregationFn] = None

    def __post_init__(self) -> None:
        self.max_prompt_len: int = max(
            getattr(t, "max_prompt_len", 0) for t in self.tasks.values()
        )
        self.max_gen_len: int = max(
            getattr(t, "max_gen_len", 0) for t in self.tasks.values()
        )


class GroupTask(Task):
    def __init__(
        self, tasks: Dict[str, Task], aggregate_fn: Optional[AggregationFn] = None
    ) -> None:
        self.tasks: Dict[str, Task] = tasks
        self.aggregate_fn = aggregate_fn

    @staticmethod
    def from_config(config: GroupTaskConfig) -> "GroupTask":
        tasks: Dict[str, Task] = {}
        for name, cfg in config.tasks.items():
            if isinstance(cfg, GenerationTaskConfig):
                tasks[name] = GenerationTask.from_config(cfg)
            elif isinstance(cfg, ChoiceTaskConfig):
                tasks[name] = ChoiceTask.from_config(cfg)
            else:
                raise NotImplementedError(f"No support for group tasks of {type(cfg)}")
        return GroupTask(tasks, config.aggregate_fn)

    def run(
        self,
        predictor: Predictor,
        random_state: Optional[RandomState] = None,
        max_samples: Optional[int] = None,
        show_progress: bool = False,
        **kwargs: Any,
    ) -> TaskResult:
        metrics: Dict[str, Dict[str, AverageMetric]] = {}
        raw_results: Dict[str, Any] = {}
        for name, task in self.tasks.items():
            rank_zero_info(f"Running evaluation on subtask {name}")
            result = task.run(
                predictor=predictor,
                random_state=random_state,
                max_samples=max_samples,
                show_progress=show_progress,
                **kwargs,
            )
            # Sync task completion across all ranks in a mp group. This ensures each task completes on all ranks before the next task starts on any rank
            # Otherwise race conditions arise where non-rank_0 processes start the next task before rank_0 has completed the first task.
            # This has been observed to cause timeouts in pytorch distributed operations as other rank 0's are unexpectedly waiting for rank_0 to complete the first task
            if torch.distributed.is_initialized():
                torch.distributed.barrier()

            metrics[name] = result.metrics
            raw_results[name] = result.raw_results
            log = format_dict(metrics[name], delimiter=" | ", decimal=6)
            rank_zero_info(f"Evaluation results on subtask {name}: {log}")
            torch.cuda.memory.empty_cache()

        if self.aggregate_fn is not None:
            metrics.update(self.aggregate_fn(metrics))
        agg_results: Dict[str, AverageMetric] = flatten_dict(metrics)
        return TaskResult(metrics=agg_results, raw_results=[raw_results])
