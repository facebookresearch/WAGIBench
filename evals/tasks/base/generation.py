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
from typing import Any, Callable, Dict, List, Literal, Optional, Sequence

import torch

from evals.api import (
    Example,
    ExampleFn,
    FilterFn,
    GlobalMetricFn,
    MetricFn,
    Prediction,
    Predictor,
    Prompt,
    Task,
    TaskConfig,
    TaskResult,
)
from evals.utils import (
    ExampleSelector,
    gather_object,
    get_dp_group,
    get_dp_rank,
    get_dp_size,
    get_mp_rank,
    get_rank,
    load_jsonl,
    mean_reduce_dict,
)
from numpy.random import RandomState


@dataclass
class GenerationTaskConfig(TaskConfig):
    eval_file: str
    prompt_fn: Callable[..., Prompt]
    max_gen_len: int = 256
    max_prompt_len: int = 1024
    num_generations: int = 1
    return_logprobs: bool = False

    num_few_shot: int = 0
    few_shot_examples: Optional[List[Example]] = None
    few_shot_file: Optional[str] = None
    few_shot_strategy: Literal["first", "index", "random"] = "first"
    few_shot_indices: Optional[Sequence[int]] = None

    preprocess_fn: Optional[ExampleFn] = None
    postprocess_fn: Optional[ExampleFn] = None
    metric_fns: Optional[Sequence[MetricFn]] = None
    global_metric_fns: Optional[Sequence[GlobalMetricFn]] = None
    filter_fn: Optional[FilterFn] = None


class GenerationTask(Task):
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
        filter_fn: Optional[FilterFn] = None,
        return_logprobs: bool = False,
    ) -> None:
        self.dataset: List[Example] = dataset
        self.prompt_fn = prompt_fn
        self.max_prompt_len = max_prompt_len
        self.max_gen_len = max_gen_len
        self.num_generations = num_generations
        self.few_shot_selector = few_shot_selector
        self.return_logprobs = return_logprobs
        self.preprocess_fn: Optional[ExampleFn] = preprocess_fn
        self.postprocess_fn: Optional[ExampleFn] = postprocess_fn
        self.metric_fns: Sequence[MetricFn] = metric_fns or []
        self.global_metric_fns: Sequence[GlobalMetricFn] = global_metric_fns or []
        self.filter_fn: Optional[FilterFn] = filter_fn

    @staticmethod
    def from_config(cfg: GenerationTaskConfig) -> "GenerationTask":
        dataset = load_jsonl(cfg.eval_file, get_dp_size(), get_dp_rank())
        few_shot_examples = cfg.few_shot_examples
        if cfg.num_few_shot > 0 and few_shot_examples is None:
            assert cfg.few_shot_file is not None
            few_shot_examples = load_jsonl(filename=cfg.few_shot_file)

        return GenerationTask(
            dataset=dataset,
            prompt_fn=cfg.prompt_fn,
            max_gen_len=cfg.max_gen_len,
            max_prompt_len=cfg.max_prompt_len,
            num_generations=cfg.num_generations,
            return_logprobs=cfg.return_logprobs,
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
            filter_fn=cfg.filter_fn,
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
        if max_samples is not None:
            self.dataset = self.dataset[:max_samples]
        if self.filter_fn is not None:
            self.dataset = [x for x in self.dataset if self.filter_fn(x)]

        indices: List[Sequence[int]] = []
        prompts: List[Prompt] = []
        for x in self.dataset:
            x.update(self.preprocess_fn(x) if self.preprocess_fn else {})
            x["few_shot"] = self.few_shot_selector(random_state=random_state)
            x["prompt"] = self.prompt_fn(**x)
            indices.append(range(len(prompts), len(prompts) + self.num_generations))
            prompts.extend([x["prompt"]] * self.num_generations)

        predictions: Sequence[Prediction] = predictor(
            prompts=prompts,
            max_prompt_len=self.max_prompt_len,
            max_gen_len=self.max_gen_len,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            return_logprobs=self.return_logprobs,
            echo=False,
            show_progress=show_progress,
        )

        # Ensure predictions are complete across all ranks in a mp group before proceeding to calculate metrics. This makes the code easier to reason about in a distributed setting with minimal performance cost.
        # Although not yet observed, it is possible for issues to occur if a rank is in the process of looking for new prompts right as all predictions are completed.
        # In this case, one rank may start calculating metrics while other ranks are still looking for new prompts which can lead to unexpected behavior.
        if torch.distributed.is_initialized():
            torch.distributed.barrier()

        # Calculate metrics on MP rank 0 only
        if get_mp_rank() != 0:
            return TaskResult(metrics={})

        raw_results: Dict[str, List[float]] = defaultdict(list)
        for idx, x in enumerate(self.dataset):
            preds = [predictions[i] for i in indices[idx]]
            x["prediction_token_ids"] = [p.token_ids for p in preds]
            x["prediction_tokens"] = [p.tokens for p in preds]
            x["prediction_texts"] = [p.text for p in preds]
            if self.return_logprobs:
                x["prediction_logprobs"] = [p.logprobs for p in preds]
            x.update(self.postprocess_fn(x) if self.postprocess_fn else {})

            x["metrics"] = {k: v for fn in self.metric_fns for k, v in fn(x).items()}
            for name, value in x["metrics"].items():
                raw_results[name].append(value)

        avg_results = mean_reduce_dict(raw_results, group=get_dp_group())

        if len(self.global_metric_fns) > 0:
            object_gather_list = gather_object(self.dataset, group=get_dp_group())
            if get_rank() == 0:
                gathered_results = [x for obj in object_gather_list for x in obj]
                for ag_fn in self.global_metric_fns:
                    avg_results.update(ag_fn(gathered_results))

        return TaskResult(metrics=avg_results, raw_results=self.dataset)
