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

from evals.api import (
    Example,
    ExampleFn,
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
    get_rank,
    load_jsonl,
    mean_reduce_dict,
)
from numpy.random import RandomState

try:
    from llm_common.datatypes import Message, SampleSFT
except (ImportError, ModuleNotFoundError):
    Message = SampleSFT = None


@dataclass
class MultiTurnGenerationTaskConfig(TaskConfig):
    eval_file: str
    prompt_fn: Callable
    num_turn_fn: Callable
    max_samples: Optional[int] = None
    max_gen_len: int = 256
    max_prompt_len: int = 1024
    num_generations: int = 1
    return_logprobs: bool = False

    num_few_shot: int = 0
    few_shot_examples: Optional[List[Example]] = None
    few_shot_file: Optional[str] = None
    few_shot_strategy: Literal["first", "index", "random"] = "first"
    few_shot_indices: Optional[List[int]] = None

    preprocess_fn: Optional[ExampleFn] = None
    postprocess_fn: Optional[ExampleFn] = None
    metric_fns: Optional[List[MetricFn]] = None
    global_metric_fns: Optional[List[GlobalMetricFn]] = None


class MultiTurnGenerationTask(Task):
    def __init__(
        self,
        dataset: List[Example],
        prompt_fn: Callable[..., Prompt],
        num_turn_fn: Callable[..., int],
        max_gen_len: int,
        max_prompt_len: int,
        num_generations: int,
        few_shot_selector: ExampleSelector,
        preprocess_fn: Optional[ExampleFn],
        postprocess_fn: Optional[ExampleFn],
        metric_fns: Optional[List[MetricFn]],
        global_metric_fns: Optional[List[GlobalMetricFn]],
        return_logprobs: bool = False,
    ) -> None:
        self.dataset: List[Example] = dataset
        self.prompt_fn = prompt_fn
        self.num_turn_fn = num_turn_fn
        self.max_prompt_len = max_prompt_len
        self.max_gen_len = max_gen_len
        self.num_generations = num_generations
        self.few_shot_selector = few_shot_selector
        self.return_logprobs = return_logprobs
        self.preprocess_fn: Optional[ExampleFn] = preprocess_fn
        self.postprocess_fn: Optional[ExampleFn] = postprocess_fn
        self.metric_fns: List[MetricFn] = metric_fns or []
        self.global_metric_fns: List[GlobalMetricFn] = global_metric_fns or []

    @staticmethod
    def from_config(cfg: MultiTurnGenerationTaskConfig) -> "MultiTurnGenerationTask":
        dataset = load_jsonl(
            cfg.eval_file,
            num_shards=get_dp_size(),
            shard_idx=get_dp_rank(),
            max_samples=cfg.max_samples,
        )
        few_shot_examples = cfg.few_shot_examples
        if cfg.num_few_shot > 0 and few_shot_examples is None:
            assert cfg.few_shot_file is not None
            few_shot_examples = load_jsonl(filename=cfg.few_shot_file)

        return MultiTurnGenerationTask(
            dataset=dataset,
            prompt_fn=cfg.prompt_fn,
            num_turn_fn=cfg.num_turn_fn,
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

        raw_results: Dict[str, List[float]] = defaultdict(list)
        # Each sample may have a different number of turns, so have to compute sample one-by-one
        for x in self.dataset:
            x.update(self.preprocess_fn(x) if self.preprocess_fn else {})
            x["few_shot"] = self.few_shot_selector(random_state=random_state)
            x["num_turns"] = self.num_turn_fn(**x)
            x["num_generations"] = self.num_generations
            prev_turns = [SampleSFT(dialog=[])] * self.num_generations
            predictions: List[Prediction] = []
            # generate each turn
            for idx in range(x["num_turns"]):
                prompts = [
                    self.prompt_fn(**x, turn_idx=idx, prev_turns=t) for t in prev_turns
                ]
                new_preds = predictor(
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
                predictions.extend(new_preds)
                prev_turns = [
                    SampleSFT(
                        dialog=t.dialog + [Message.assistant(body=p.text.strip())]
                    )
                    for (t, p) in zip(prev_turns, new_preds)
                ]
            # compute metrics
            x["prediction_token_ids"] = [p.token_ids for p in predictions]
            x["prediction_tokens"] = [p.tokens for p in predictions]
            x["prediction_texts"] = [p.text for p in predictions]
            if self.return_logprobs:
                x["prediction_logprobs"] = [p.logprobs for p in predictions]
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


@dataclass
class MultiGenerationTaskConfig(TaskConfig):
    eval_file: str
    prompt_fn: List[Callable]
    max_samples: Optional[int] = None
    max_gen_len: int = 256
    max_prompt_len: int = 1024
    num_generations: int = 1
    return_logprobs: bool = False

    preprocess_fn: Optional[ExampleFn] = None
    postprocess_fn: Optional[ExampleFn] = None
    metric_fns: Optional[List[MetricFn]] = None
    global_metric_fns: Optional[List[GlobalMetricFn]] = None


class MultiGenerationTask(Task):
    """
    Generate multiple turns of answers, where prompt of a specific turn depends on the response from the previous turn.
    Please provide prompt_fn for each generated turn, where each prompt_fn has access to all the previous generated responses.
    """

    def __init__(
        self,
        dataset: List[Example],
        prompt_fn: List[Callable[..., Prompt]],
        max_gen_len: int,
        max_prompt_len: int,
        num_generations: int,
        preprocess_fn: Optional[ExampleFn],
        postprocess_fn: Optional[ExampleFn],
        metric_fns: Optional[List[MetricFn]],
        global_metric_fns: Optional[List[GlobalMetricFn]],
        return_logprobs: bool = False,
    ) -> None:
        self.dataset: List[Example] = dataset
        self.prompt_fn = prompt_fn
        self.max_prompt_len = max_prompt_len
        self.max_gen_len = max_gen_len
        self.num_generations = num_generations
        self.return_logprobs = return_logprobs
        self.preprocess_fn: Optional[ExampleFn] = preprocess_fn
        self.postprocess_fn: Optional[ExampleFn] = postprocess_fn
        self.metric_fns: List[MetricFn] = metric_fns or []
        self.global_metric_fns: List[GlobalMetricFn] = global_metric_fns or []

    @staticmethod
    def from_config(cfg: MultiGenerationTaskConfig) -> "MultiGenerationTask":
        dataset = load_jsonl(
            cfg.eval_file,
            num_shards=get_dp_size(),
            shard_idx=get_dp_rank(),
            max_samples=cfg.max_samples,
        )

        return MultiGenerationTask(
            dataset=dataset,
            prompt_fn=cfg.prompt_fn,
            max_gen_len=cfg.max_gen_len,
            max_prompt_len=cfg.max_prompt_len,
            num_generations=cfg.num_generations,
            return_logprobs=cfg.return_logprobs,
            preprocess_fn=cfg.preprocess_fn,
            postprocess_fn=cfg.postprocess_fn,
            metric_fns=cfg.metric_fns,
            global_metric_fns=cfg.global_metric_fns,
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

        raw_results: Dict[str, List[float]] = defaultdict(list)
        prev_turns = [[SampleSFT(dialog=[])] * self.num_generations] * len(self.dataset)
        all_pred: List[List[Prediction]] = (
            [[]] * self.num_generations * len(self.dataset)  # type: ignore
        )
        for p_fn in self.prompt_fn:
            prompts: List[Prompt] = []
            prompt_indices: List[List[int]] = []
            prev_index: int = 0
            for idx, x in enumerate(self.dataset):
                x.update(self.preprocess_fn(x) if self.preprocess_fn else {})
                new_prompt = [p_fn(**x, prev_turns=t) for t in prev_turns[idx]]
                prev_turns[idx] = [
                    SampleSFT(dialog=t.dialog + p.dialog)  # type: ignore
                    for (t, p) in zip(prev_turns[idx], new_prompt)
                ]
                prompts.extend(new_prompt)
                prompt_indices.append(list(range(prev_index, len(prompts))))
                prev_index = len(prompts)

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
            all_pred = [a + [p] for a, p in zip(all_pred, predictions)]
            for idx, x in enumerate(self.dataset):
                indices = prompt_indices[idx]
                prev_turns[idx] = [
                    SampleSFT(dialog=t.dialog + p.messages)
                    for (t, p) in zip(
                        prev_turns[idx], [predictions[i] for i in indices]
                    )
                ]

        for idx, x in enumerate(self.dataset):
            indices = prompt_indices[idx]
            preds = [all_pred[i] for i in indices]
            x["prediction_token_ids"] = [[t.token_ids for t in p] for p in preds]
            x["prediction_tokens"] = [[t.tokens for t in p] for p in preds]
            x["prediction_texts"] = [[t.text for t in p] for p in preds]
            x["messages"] = [[t.messages for t in p] for p in preds]
            if self.return_logprobs:
                x["prediction_logprobs"] = [[t.logprobs for t in p] for p in preds]
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
