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

import abc
from dataclasses import dataclass
from enum import Enum
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    runtime_checkable,
    Sequence,
    Tuple,
    Union,
)

try:
    from llm_common.datatypes import Message as LLMCommonMessage, SampleSFT
except ImportError:
    SampleSFT = LLMCommonMessage = None

from numpy.random import RandomState
from typing_extensions import Protocol


@dataclass
class Dialog:  # DEPRECATED
    # Keep it to prevent breaking other dependencies
    messages: List[LLMCommonMessage]

    def __repr__(self) -> str:
        return "\n\n".join(str(msg) for msg in self.messages)

    def to_sample_sft(self) -> SampleSFT:
        return SampleSFT(dialog=self.messages)


PromptForDialog = Union[str, Dialog, List[int], SampleSFT]


class Role(Enum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"


@dataclass
class Message:
    role: Role
    content: str

    def __repr__(self) -> str:
        return f"{self.role.value.upper()}: {self.content}"

    def to_dict(self) -> Dict[str, str]:
        return {"role": self.role.value, "content": self.content}


Prompt = Union[str, List[Message], List[int]]


@dataclass
class Prediction:
    text: str
    tokens: Optional[List[str]] = None
    logprobs: Optional[List[float]] = None
    text_offsets: Optional[List[int]] = None
    token_ids: Optional[List[int]] = None
    messages: Optional[List[Message]] = None
    output_dict: Optional[Dict[str, Any]] = None


@runtime_checkable
class Predictor(Protocol):
    @staticmethod
    @abc.abstractmethod
    def from_config(config: "PredictorConfig") -> "Predictor": ...

    @abc.abstractmethod
    def __call__(
        self,
        prompts: Sequence[Prompt],
        max_prompt_len: int,
        max_gen_len: int,
        temperature: float,
        top_p: float,
        top_k: int,
        echo: bool,
        return_logprobs: bool,
        show_progress: bool,
    ) -> Sequence[Prediction]: ...


@dataclass
class AverageMetric:
    """
    Average metric with confidence interval.

    avg is the mean of a list of values
    count is the length of this list
    square is the mean of the squares of the values
    avg_ci_fn is a function applied to the bounds of the confidence interval
    raw_values: TODO
    """

    avg: float
    count: int
    square: float
    avg_ci_fn: Optional[Callable] = None

    @property
    def value(self):
        return self.avg_ci_fn(self.avg) if self.avg_ci_fn else self.avg

    def update(self, value: float, count: int, square: Optional[float] = None) -> None:
        self.avg = (self.avg * self.count + value * count) / (self.count + count)
        if square is None:
            assert count == 1
            square = value**2
        self.square = (self.square * self.count + square * count) / (self.count + count)
        self.count += count

    def compute_ci(
        self, confidence_level: float = 0.95
    ) -> Tuple[Optional[float], Optional[float]]:
        """
        returns bounds of confidence interval: ci_lb ('lower_bound') and ci_ub ('upper_bound').
        Confidence interval is computed with error margins:
        z * s / sqrt(n), where:
        - P(-z <= X <= z) = confidence_level and X follows a t student low with self.count - 1 parameters.
        - s is the unbiased std estimate: (1/(n-1) sum((xi - mean(xi) ** 2))) ** 0.5

        example: first 100 integers as metric_values and confidence_level = 0.95:
        >>> avg_m = AverageMetric(0, 0, 0)
        >>> for i in range(100):
        >>>     avg_m.update(value=i, count=1)
        >>> avg_m.compute_ci() # mean is 49.5, std is 29.0115, self.count = 100, z = 1.98
        >>> (43.743, 55.257)
        """
        from scipy.stats import t

        if self.count < 2:
            return None, None

        std = (self.count / (self.count - 1) * (self.square - (self.avg) ** 2)) ** 0.5
        scale = std / (self.count**0.5)
        lb, ub = t.interval(confidence_level, self.count - 1, loc=self.avg, scale=scale)
        if self.avg_ci_fn:
            lb, ub = self.avg_ci_fn(lb), self.avg_ci_fn(ub)
        return (lb, ub)


@dataclass
class TaskResult:
    metrics: Dict[str, AverageMetric]
    raw_results: Optional[List[Dict[str, Any]]] = None


class Task(abc.ABC):
    @abc.abstractmethod
    def run(
        self,
        predictor: Predictor,
        random_state: Optional[RandomState] = None,
        max_samples: Optional[int] = None,
        show_progress: bool = False,
        **kwargs: Any,
    ) -> TaskResult: ...


@dataclass
class TaskConfig:
    pass


@dataclass
class PredictorConfig:
    pass


Example = Dict[str, Any]
ExampleFn = Callable[[Example], Example]
MetricFn = Callable[[Example], Dict[str, float]]
GlobalMetricFn = Callable[[List[Example]], Dict[str, AverageMetric]]
ParallelMetricFn = Callable[[List[Example], bool], List[Dict[str, float]]]
AggregationFn = Callable[
    [Dict[str, Dict[str, AverageMetric]]], Dict[str, Dict[str, AverageMetric]]
]
FilterFn = Callable[[Example], bool]
