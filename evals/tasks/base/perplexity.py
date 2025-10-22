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

import json
from collections import defaultdict
from dataclasses import dataclass
from itertools import islice
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import torch

from evals.api import AverageMetric, Prediction, Predictor, Task, TaskConfig, TaskResult
from evals.utils import (
    flatten_dict,
    format_dict,
    get_dp_group,
    get_dp_rank,
    get_dp_size,
    get_mp_rank,
    mean_reduce_dict,
    rank_zero_info,
)
from numpy.random import RandomState


@dataclass
class PerplexityTaskConfig(TaskConfig):
    files: str
    max_prompt_len: int = 2048
    num_prompts: int = 1600
    window_stride: Optional[int] = None
    split_by_eos: bool = True

    def __post_init__(self) -> None:
        for file in self.files.split(","):
            assert Path(file).exists(), f"{file} does not exist"
        self.max_prompt_len += 1  # due to input shifting


class PerplexityTask(Task):
    def __init__(
        self,
        files: List[str],
        max_prompt_len: int,
        num_prompts: int,
        window_stride: Optional[int],
        split_by_eos: bool = True,
    ) -> None:
        self.files = files
        self.max_prompt_len: int = max_prompt_len
        self.num_prompts = num_prompts
        self.window_stride = window_stride
        self.split_by_eos = split_by_eos

    @staticmethod
    def from_config(cfg: PerplexityTaskConfig) -> "PerplexityTask":
        return PerplexityTask(
            cfg.files.split(","),
            cfg.max_prompt_len,
            cfg.num_prompts,
            cfg.window_stride,
            cfg.split_by_eos,
        )

    def run(
        self,
        predictor: Predictor,
        random_state: Optional[RandomState] = None,
        max_samples: Optional[int] = None,
        show_progress: bool = False,
        **kwargs: Any,
    ) -> TaskResult:
        metrics: Dict[str, Dict[str, AverageMetric]] = {}
        raw_results: Dict[str, List[Any]] = defaultdict(list)
        for eval_file in self.files:
            prompts = self.tokenize_prompt(eval_file, predictor)
            rank_zero_info(
                f"Calculating perplexity using {len(prompts)} prompts of length "
                f"{self.max_prompt_len - 1} from {eval_file}"
            )
            prompts = list(islice(prompts, get_dp_rank(), None, get_dp_size()))
            predictions: Sequence[Prediction] = predictor(
                prompts=prompts,
                max_prompt_len=self.max_prompt_len,
                max_gen_len=0,
                temperature=0.0,
                top_p=0,
                top_k=0,
                echo=True,
                return_logprobs=True,
                show_progress=show_progress,
            )

            # Calculate metrics on MP rank 0 only
            if get_mp_rank() != 0:
                continue
            nll_tokens = [[-x for x in pred.logprobs] for pred in predictions]  # type: ignore
            nll_dict = {"nll_token": [nll for seq in nll_tokens for nll in seq]}
            results = mean_reduce_dict(nll_dict, group=get_dp_group())
            results["ppl"] = AverageMetric(
                avg=results["nll_token"].value,
                count=results["nll_token"].count,
                square=results["nll_token"].square,
                avg_ci_fn=lambda x: float(np.exp(x)),
            )
            metrics[eval_file] = results
            if prompts and nll_tokens:
                raw_results[eval_file] = [
                    dict(zip(("prompt", "nll_token"), v))
                    for v in zip(prompts, nll_tokens)
                ]
            log = format_dict(metrics[eval_file], delimiter=" | ", decimal=6)
            rank_zero_info(f"Evaluation results on file {eval_file}: {log}")
            torch.cuda.memory.empty_cache()
        return TaskResult(metrics=flatten_dict(metrics), raw_results=[raw_results])

    def tokenize_prompt(self, eval_file: str, predictor: Predictor) -> List[List[int]]:
        assert hasattr(predictor, "tokenizer"), "predictor doesn't have a tokenizer"
        buffer: List[int] = []
        prompts: List[List[int]] = []
        index: int = 0
        done = False
        with open(eval_file, "r") as file:
            for line in file:
                data = json.loads(line)
                text = data["text" if "text" in data else "content"]
                buffer.extend(predictor.encode(text, max_len=1e9))  # type: ignore
                buffer.append(predictor.tokenizer.eos_id)  # type: ignore

                while len(buffer) - index >= self.max_prompt_len:
                    prompts.append(buffer[index : index + self.max_prompt_len])
                    index += self.window_stride or self.max_prompt_len
                    if len(prompts) >= self.num_prompts:
                        done = True
                        break
                if done:
                    break

        if not self.split_by_eos:
            return prompts

        out = []
        for prompt in prompts:
            i = 0
            for j in range(len(prompt)):
                if prompt[j] == predictor.tokenizer.eos_id or j == len(prompt) - 1:
                    out.append(prompt[i : j + 2])
                    i = j + 1  # repeat last token
        return out
