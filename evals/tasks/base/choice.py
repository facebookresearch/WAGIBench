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

import numpy as np
import torch

from evals.api import (
    Example,
    ExampleFn,
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
    get_dp_group,
    get_dp_rank,
    get_dp_size,
    get_mp_rank,
    load_jsonl,
    mean_reduce_dict,
    text_index,
)
from numpy.random import RandomState


def nll_accuracy(x: Dict[str, Any]) -> Dict[str, float]:
    metrics: Dict[str, float] = {}
    for name, nlls in x["nlls"].items():
        pred = int(np.argmin(nlls))
        metrics[f"acc_{name}"] = float(pred == x["target"])
        metrics[f"nll_{name}"] = nlls[pred]
        metrics[f"nll_{name}_target"] = nlls[x["target"]]
        # negative log-normalized probability of target -log(P[correct]/(sum_{choice} P[choice]))
        norm_factor = torch.logsumexp(-torch.tensor(nlls), dim=0).item()
        metrics[f"nll_{name}_target_norm"] = nlls[x["target"]] + norm_factor
    return metrics


@dataclass
class ChoiceTaskConfig(TaskConfig):
    eval_file: str
    prompt_fn: Callable[..., str]

    max_prompt_len: int = 1024
    max_gen_len: int = 0
    nll_completion: bool = False

    num_few_shot: int = 0
    few_shot_examples: Optional[List[Example]] = None
    few_shot_file: Optional[str] = None
    few_shot_strategy: Literal["first", "index", "random"] = "first"
    few_shot_indices: Optional[Sequence[int]] = None

    preprocess_fn: Optional[ExampleFn] = None
    postprocess_fn: Optional[ExampleFn] = None
    metric_fns: Optional[Sequence[MetricFn]] = (nll_accuracy,)


class ChoiceTask(Task):
    def __init__(
        self,
        dataset: List[Example],
        prompt_fn: Callable[..., Prompt],
        max_prompt_len: int,
        max_gen_len: int,
        nll_completion: bool,
        few_shot_selector: ExampleSelector,
        preprocess_fn: Optional[ExampleFn],
        postprocess_fn: Optional[ExampleFn],
        metric_fns: Optional[Sequence[MetricFn]],
    ) -> None:
        self.dataset: List[Example] = dataset
        self.prompt_fn = prompt_fn
        self.max_prompt_len = max_prompt_len
        self.max_gen_len = max_gen_len

        self.few_shot_selector = few_shot_selector
        self.preprocess_fn: Optional[ExampleFn] = preprocess_fn
        self.postprocess_fn: Optional[ExampleFn] = postprocess_fn
        self.metric_fns: Sequence[MetricFn] = metric_fns or []
        self.nll_completion = nll_completion

    @staticmethod
    def from_config(cfg: ChoiceTaskConfig) -> "ChoiceTask":
        dataset = load_jsonl(cfg.eval_file, get_dp_size(), get_dp_rank())
        few_shot_examples = cfg.few_shot_examples
        if cfg.num_few_shot > 0 and few_shot_examples is None:
            assert cfg.few_shot_file is not None
            few_shot_examples = load_jsonl(filename=cfg.few_shot_file)

        return ChoiceTask(
            dataset=dataset,
            prompt_fn=cfg.prompt_fn,
            max_prompt_len=cfg.max_prompt_len,
            max_gen_len=cfg.max_gen_len,
            nll_completion=cfg.nll_completion,
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
        )

    def run(
        self,
        predictor: Predictor,
        random_state: Optional[RandomState] = None,
        max_samples: Optional[int] = None,
        show_progress: bool = False,
        **kwargs: Any,
    ) -> TaskResult:
        if max_samples is not None:
            self.dataset = self.dataset[:max_samples]

        prompts: List[Prompt] = []
        indices: List[Sequence[int]] = []
        for x in self.dataset:
            x.update(self.preprocess_fn(x) if self.preprocess_fn else {})
            assert all(key in x for key in ("choice_texts", "target"))

            x["few_shot"] = self.few_shot_selector(random_state=random_state)
            x["prompts"] = [self.prompt_fn(**x, choice_text=c) for c in x["choice_texts"]]  # type: ignore
            x["completions"] = [f"Answer: {c}" for c in x["choice_texts"]]

            prev_index = len(prompts)
            prompts += x["prompts"] + (x["completions"] if self.nll_completion else [])
            indices.append(range(prev_index, len(prompts)))

        predictions: Sequence[Prediction] = predictor(
            prompts=prompts,
            max_prompt_len=self.max_prompt_len,
            max_gen_len=self.max_gen_len,
            temperature=0.0,
            top_p=0.0,
            top_k=0,
            echo=True,
            return_logprobs=True,
            show_progress=show_progress,
        )

        # Calculate metrics on MP rank 0 only
        if get_mp_rank() != 0:
            return TaskResult(metrics={})

        raw_results: Dict[str, List[float]] = defaultdict(list)
        for idx, x in enumerate(self.dataset):
            x.update(self.postprocess_fn(x) if self.postprocess_fn else {})
            preds = [predictions[i] for i in indices[idx]]

            x["nlls"] = defaultdict(list)
            for cix, (text, pred) in enumerate(zip(x["choice_texts"], preds)):
                assert pred.tokens and pred.logprobs and pred.text_offsets, pred
                logprobs = pred.logprobs[text_index(pred.text, pred.text_offsets, text)]
                x["nlls"]["char"].append(-sum(logprobs) / len(text))
                x["nlls"]["token"].append(-sum(logprobs) / len(logprobs))
                x["nlls"]["raw"].append(-sum(logprobs))

                if self.nll_completion:
                    assert len(preds) == 2 * len(x["choice_texts"])
                    compl = preds[cix + len(x["choice_texts"])]
                    assert compl.tokens and compl.logprobs and compl.text_offsets
                    slice = text_index(compl.text, compl.text_offsets, text)
                    nll_compl = -sum(logprobs) + sum(compl.logprobs[slice])
                    x["nlls"]["completion"].append(nll_compl)

            x["metrics"] = {k: v for fn in self.metric_fns for k, v in fn(x).items()}
            for name, value in x["metrics"].items():
                raw_results[name].append(value)

        avg_results = mean_reduce_dict(raw_results, group=get_dp_group())
        return TaskResult(metrics=avg_results, raw_results=self.dataset)
