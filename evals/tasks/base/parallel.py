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

from collections import defaultdict
from dataclasses import dataclass
from itertools import islice
from typing import Any, Callable, Dict, List, Optional, Sequence

from evals.api import (
    Example,
    ExampleFn,
    GlobalMetricFn,
    MetricFn,
    ParallelMetricFn,
    Predictor,
    Prompt,
    TaskResult,
)
from evals.tasks.base.generation import GenerationTask, GenerationTaskConfig
from evals.utils import (
    all_gather_object,
    ExampleSelector,
    get_dp_group,
    get_dp_rank,
    get_dp_size,
    init_torch_distributed,
    load_jsonl,
    mean_reduce_dict,
)
from numpy.random import RandomState


@dataclass
class ParallelGenerationTaskConfig(GenerationTaskConfig):
    parallel_metric_fns: Optional[List[ParallelMetricFn]] = None


class ParallelGenerationTask(GenerationTask):
    def __init__(
        self,
        dataset: List[Example],
        prompt_fn: Callable[..., Prompt],
        max_gen_len: int,
        max_prompt_len: int,
        num_generations: int,
        few_shot_selector: ExampleSelector,
        preprocess_fn: Optional[ExampleFn],
        postprocess_fn: Optional[ExampleFn],
        metric_fns: Optional[Sequence[MetricFn]],
        global_metric_fns: Optional[Sequence[GlobalMetricFn]],
        parallel_metric_fns: Optional[List[ParallelMetricFn]],
    ) -> None:
        GenerationTask.__init__(
            self,
            dataset=dataset,
            prompt_fn=prompt_fn,
            max_gen_len=max_gen_len,
            max_prompt_len=max_prompt_len,
            num_generations=num_generations,
            few_shot_selector=few_shot_selector,
            preprocess_fn=preprocess_fn,
            postprocess_fn=postprocess_fn,
            metric_fns=metric_fns,
            global_metric_fns=global_metric_fns,
        )
        self.parallel_metric_fns: List[ParallelMetricFn] = parallel_metric_fns or []

    @staticmethod
    def from_config(cfg: ParallelGenerationTaskConfig) -> "ParallelGenerationTask":  # type: ignore
        dataset = load_jsonl(cfg.eval_file, get_dp_size(), get_dp_rank())
        few_shot_examples = cfg.few_shot_examples
        if cfg.num_few_shot > 0 and few_shot_examples is None:
            assert cfg.few_shot_file is not None
            few_shot_examples = load_jsonl(filename=cfg.few_shot_file)

        return ParallelGenerationTask(
            dataset=dataset,
            prompt_fn=cfg.prompt_fn,
            max_gen_len=cfg.max_gen_len,
            max_prompt_len=cfg.max_prompt_len,
            num_generations=cfg.num_generations,
            few_shot_selector=ExampleSelector(
                examples=few_shot_examples,
                num_examples=cfg.num_few_shot,
                select_strategy=cfg.few_shot_strategy,
                select_indices=cfg.few_shot_indices,
                preprocess_fn=cfg.preprocess_fn,
            ),
            preprocess_fn=cfg.preprocess_fn,
            postprocess_fn=cfg.postprocess_fn,
            metric_fns=cfg.metric_fns,
            global_metric_fns=cfg.global_metric_fns,
            parallel_metric_fns=cfg.parallel_metric_fns,
        )

    def run(  # type: ignore
        self,
        predictor: Predictor,
        random_state: Optional[RandomState] = None,
        max_samples: Optional[int] = None,
        show_progress: bool = False,
        temperature: float = 1.0,
        top_p: float = 0.6,
        top_k: int = 0,
        **kwargs: Any,
    ) -> TaskResult:
        avg_results = GenerationTask.run(
            self,
            predictor=predictor,
            random_state=random_state,
            max_samples=max_samples,
            show_progress=show_progress,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            **kwargs,
        )
        raw_results: Dict[str, List[float]] = defaultdict(list)
        if len(self.parallel_metric_fns) > 0:
            init_torch_distributed()
            object_gather_list = all_gather_object(self.dataset, group=get_dp_group())
            gathered_results = [x for obj in object_gather_list for x in obj]
            for parallel_fn in self.parallel_metric_fns:
                all_results_list = all_gather_object(
                    parallel_fn(gathered_results, show_progress),
                    group=get_dp_group(),
                )
                all_results = [x for obj in all_results_list for x in obj]
                results = islice(all_results, get_dp_rank(), None, get_dp_size())
                for x, metrics in zip(self.dataset, results):
                    x["metrics"].update(metrics)

        for x in self.dataset:
            for name, value in x["metrics"].items():
                raw_results[name].append(value)

        avg_results.metrics.update(mean_reduce_dict(raw_results, group=get_dp_group()))
        return TaskResult(metrics=avg_results.metrics, raw_results=self.dataset)
