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
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from evals.api import (
    AverageMetric,
    Example,
    ExampleFn,
    MetricFn,
    Prediction as _Prediction,
    Predictor,
    Task,
    TaskConfig,
    TaskResult,
)
from evals.utils import (
    gather_object,
    get_dp_group,
    get_dp_rank,
    get_dp_size,
    get_mp_rank,
    mean_reduce_dict,
    rank_zero_info,
)

from numpy.random import RandomState


@dataclass
class Prediction(_Prediction):
    text: str = ""
    score: Optional[float] = None
    output_dict: Optional[Dict[str, Any]] = None


@dataclass
class ImageReasoningTaskConfig(TaskConfig):
    load_data_fn: Callable[..., List[Example]]
    max_gen_len: int = 20
    sweep_temperature: Optional[bool] = False
    min_temperature: Optional[float] = 0.1
    max_temperature: Optional[float] = 1.0
    num_temperature: Optional[int] = 10
    preprocess_fn: Optional[ExampleFn] = None
    postprocess_fn: Optional[ExampleFn] = None
    metric_fns: Optional[List[MetricFn]] = None
    global_metric_fns: Optional[
        List[Callable[[List[Example]], Dict[str, Tuple[AverageMetric, List[float]]]]]
    ] = None


@dataclass
class VideoReasoningTaskConfig(ImageReasoningTaskConfig):
    jsonl_dataset_path: str = ""
    dataset_root_dir: str = ""
    max_gen_len: int = 20

    def to_json_dict(self) -> Dict[str, Any]:
        """
        Convert the config to a JSON dictionary.
        Useful when one wishes to write the config to a JSON file as
        functools.partial or other callables doenot support JSON serialization.

        Returns:
            A dictionary representation of the config.
        """
        return {
            "jsonl_dataset_path": self.jsonl_dataset_path,
            "dataset_root_dir": self.dataset_root_dir,
            "max_gen_len": self.max_gen_len,
            "load_data_fn": str(self.load_data_fn),
            "sweep_temperature": self.sweep_temperature,
            "min_temperature": self.min_temperature,
            "max_temperature": self.max_temperature,
            "num_temperature": self.num_temperature,
            "postprocess_fn": str(self.postprocess_fn),
            "metric_fns": str(self.metric_fns),
            "global_metric_fns": str(self.global_metric_fns),
        }


class ImageReasoningTask(Task):
    def __init__(
        self,
        annotations: List[Example],
        max_gen_len: int,
        sweep_temperature: Optional[bool] = False,
        min_temperature: Optional[float] = 0.1,
        max_temperature: Optional[float] = 1.0,
        num_temperature: Optional[int] = 10,
        preprocess_fn: Optional[ExampleFn] = None,
        postprocess_fn: Optional[ExampleFn] = None,
        metric_fns: Optional[List[MetricFn]] = None,
        global_metric_fns: Optional[
            List[
                Callable[[List[Example]], Dict[str, Tuple[AverageMetric, List[float]]]]
            ]
        ] = None,
    ) -> None:
        """
        Each dataset will be registered separately, and at that time
        `name`, `dataset_dir`, `build_dataset_fn` will be passes in.
        TODO: only support zero-shot
        Args:
            load_data_fn (Callable): to load lost of dictionaries
                TODO: let it to take some arguments, which will be used for k-shot
            metric_fns: a list of metrics can run distributedly per data point
                Input: a dict with "prediction" (str) and "targets" (string list)
                Output: a dict whose keys are metric names
            global_metric_fns: a list of metrics requiring intermediate aggregation
                Input: a list of dict with "prediction" (str) and "targets" (string list)
                Output: a dict whose keys are metric names
        """
        self.annotations = annotations
        self.max_gen_len = max_gen_len
        self.sweep_temperature = sweep_temperature
        self.min_temperature = min_temperature
        self.max_temperature = max_temperature
        self.num_temperature = num_temperature
        self.preprocess_fn: Optional[ExampleFn] = preprocess_fn
        self.postprocess_fn: Optional[ExampleFn] = postprocess_fn
        self.metric_fns: List[MetricFn] = metric_fns or []
        self.global_metric_fns: List[
            Callable[[List[Example]], Dict[str, Tuple[AverageMetric, List[float]]]]
        ] = (global_metric_fns or [])

    @staticmethod
    def from_config(config: ImageReasoningTaskConfig) -> "ImageReasoningTask":
        annotations = config.load_data_fn()
        annotations = annotations[get_dp_rank() : len(annotations) : get_dp_size()]

        return ImageReasoningTask(
            annotations=annotations,
            max_gen_len=config.max_gen_len,
            preprocess_fn=config.preprocess_fn,
            postprocess_fn=config.postprocess_fn,
            metric_fns=config.metric_fns,
            global_metric_fns=config.global_metric_fns,
            sweep_temperature=config.sweep_temperature,
            min_temperature=config.min_temperature,
            max_temperature=config.max_temperature,
            num_temperature=config.num_temperature,
        )

    def run(
        self,
        predictor: Predictor,
        random_state: Optional[RandomState] = None,
        max_samples: Optional[int] = None,
        show_progress: bool = False,
        **kwargs: Any,
    ) -> TaskResult:
        """
        Results will be gathered to global rank 0. Otherwise, it will be empty.
        """
        if max_samples is not None:
            annotations = self.annotations[:max_samples]
        else:
            annotations = self.annotations

        for example in annotations:
            example.update(
                self.preprocess_fn(
                    example,
                )
                if self.preprocess_fn
                else {}
            )

        if self.sweep_temperature:
            temp_to_predictions = {}

            for temp in np.linspace(
                self.min_temperature, self.max_temperature, self.num_temperature
            ):
                kwargs["generate_kwargs"].update({"temperature": temp})
                rank_zero_info(f"Running predictor with {kwargs}")
                temp_to_predictions[f"generation_kwargs_temp_{temp:.{1}f}"] = predictor(  # type: ignore
                    annotations=annotations,
                    max_gen_len=self.max_gen_len,
                    show_progress=show_progress,
                    **kwargs,
                )
            predictions = [
                Prediction(
                    output_dict=dict(
                        zip(
                            temp_to_predictions,
                            pred_across_sweep,
                        )
                    )
                )
                for pred_across_sweep in zip(*temp_to_predictions.values())
            ]

        else:
            rank_zero_info(f"Running predictor with {kwargs}")
            predictions: List[Prediction] = predictor(  # type: ignore
                annotations=annotations,
                max_gen_len=self.max_gen_len,
                show_progress=show_progress,
                **kwargs,
            )
        rank_zero_info("Generation is done.")
        return self.parse_predictions(predictions, annotations)

    def parse_predictions(
        self,
        predictions: List[Prediction],
        annotations: List[Example],
    ) -> TaskResult:
        # Calculate metrics on MP rank 0 only
        if get_mp_rank() != 0:
            return TaskResult(metrics={})

        raw_results: Dict[str, List[float]] = defaultdict(list)
        for pred, example in zip(predictions, annotations):  # type: ignore
            if getattr(pred, "score", None) is not None:
                example["prediction_score"] = pred.score
            if getattr(pred, "text", None) is not None:
                example["prediction"] = pred.text
            if getattr(pred, "output_dict", None) is not None:
                example["output_dict"] = pred.output_dict
            example.update(self.postprocess_fn(example) if self.postprocess_fn else {})

            example["metrics"] = {}
            for fn in self.metric_fns:
                example["metrics"].update(fn(example))
            for name, value in example["metrics"].items():
                raw_results[name].append(value)

        avg_results = mean_reduce_dict(raw_results, group=get_dp_group())
        object_gather_list = gather_object(annotations, group=get_dp_group())
        object_gather_list = [obj for obj in object_gather_list if obj is not None]
        gathered_results = [x for obj in object_gather_list for x in obj]
        if get_dp_rank() == 0:
            for global_fn in self.global_metric_fns:
                global_metric_dict = global_fn(gathered_results)
                for metric_k, (_avg, _raw) in global_metric_dict.items():
                    for result, raw_val in zip(gathered_results, _raw):
                        result["metrics"][metric_k] = raw_val
                    avg_results[metric_k] = _avg
        else:
            avg_results = {}
            gathered_results = []

        assert len(set(pred["id"] for pred in gathered_results)) == len(
            gathered_results
        )
        return TaskResult(metrics=avg_results, raw_results=gathered_results)


@dataclass
class AndroidAgentTaskConfig(TaskConfig):
    load_data_fn: Callable[..., List[Example]]
    max_gen_len: int = 20
    sweep_temperature: Optional[bool] = False
    min_temperature: Optional[float] = 0.1
    max_temperature: Optional[float] = 1.0
    num_temperature: Optional[int] = 10
    action_history_length: int = 0
    image_history_length: int = 0
    preprocess_fn: Optional[Callable] = None
    postprocess_fn: Optional[ExampleFn] = None
    metric_fns: Optional[List[MetricFn]] = None
    global_metric_fns: Optional[
        List[Callable[[List[Example]], Dict[str, Tuple[AverageMetric, List[float]]]]]
    ] = None


class AndroidAgentTask(Task):
    def __init__(
        self,
        annotations: List[Example],
        max_gen_len: int,
        sweep_temperature: Optional[bool] = False,
        min_temperature: Optional[float] = 0.1,
        max_temperature: Optional[float] = 1.0,
        num_temperature: Optional[int] = 10,
        action_history_length: Optional[int] = 0,
        image_history_length: Optional[int] = 0,
        preprocess_fn: Optional[Callable] = None,
        postprocess_fn: Optional[ExampleFn] = None,
        metric_fns: Optional[List[MetricFn]] = None,
        global_metric_fns: Optional[
            List[
                Callable[[List[Example]], Dict[str, Tuple[AverageMetric, List[float]]]]
            ]
        ] = None,
    ) -> None:
        """
        Each dataset will be registered separately, and at that time
        `name`, `dataset_dir`, `build_dataset_fn` will be passes in.
        TODO: only support zero-shot
        Args:
            load_data_fn (Callable): to load lost of dictionaries
                TODO: let it to take some arguments, which will be used for k-shot
            metric_fns: a list of metrics can run distributedly per data point
                Input: a dict with "prediction" (str) and "targets" (string list)
                Output: a dict whose keys are metric names
            global_metric_fns: a list of metrics requiring intermediate aggregation
                Input: a list of dict with "prediction" (str) and "targets" (string list)
                Output: a dict whose keys are metric names
        """
        self.annotations = annotations
        self.max_gen_len = max_gen_len
        self.sweep_temperature = sweep_temperature
        self.min_temperature = min_temperature
        self.max_temperature = max_temperature
        self.num_temperature = num_temperature
        self.action_history_length = action_history_length
        self.image_history_length = image_history_length
        self.preprocess_fn: Optional[Callable] = preprocess_fn
        self.postprocess_fn: Optional[ExampleFn] = postprocess_fn
        self.metric_fns: List[MetricFn] = metric_fns or []
        self.global_metric_fns: List[
            Callable[[List[Example]], Dict[str, Tuple[AverageMetric, List[float]]]]
        ] = (global_metric_fns or [])

    @staticmethod
    def from_config(config: AndroidAgentTaskConfig) -> "AndroidAgentTask":
        annotations = config.load_data_fn()
        annotations = annotations[get_dp_rank() : len(annotations) : get_dp_size()]

        return AndroidAgentTask(
            annotations=annotations,
            max_gen_len=config.max_gen_len,
            action_history_length=config.action_history_length,
            image_history_length=config.image_history_length,
            preprocess_fn=config.preprocess_fn,
            postprocess_fn=config.postprocess_fn,
            metric_fns=config.metric_fns,
            global_metric_fns=config.global_metric_fns,
            sweep_temperature=config.sweep_temperature,
            min_temperature=config.min_temperature,
            max_temperature=config.max_temperature,
            num_temperature=config.num_temperature,
        )

    def run(
        self,
        predictor: Predictor,
        random_state: Optional[RandomState] = None,
        max_samples: Optional[int] = None,
        show_progress: bool = False,
        **kwargs: Any,
    ) -> TaskResult:
        """
        Results will be gathered to global rank 0. Otherwise, it will be empty.
        """
        if max_samples is not None:
            annotations = self.annotations[:max_samples]
        else:
            annotations = self.annotations

        for example in annotations:
            example.update(
                self.preprocess_fn(
                    example,
                    self.action_history_length,
                    self.image_history_length,
                )
                if self.preprocess_fn
                else {}
            )

        if self.sweep_temperature:
            temp_to_predictions = {}

            for temp in np.linspace(
                self.min_temperature, self.max_temperature, self.num_temperature
            ):
                kwargs["generate_kwargs"].update({"temperature": temp})
                rank_zero_info(f"Running predictor with {kwargs}")
                temp_to_predictions[f"generation_kwargs_temp_{temp:.{1}f}"] = predictor(  # type: ignore
                    annotations=annotations,
                    max_gen_len=self.max_gen_len,
                    show_progress=show_progress,
                    **kwargs,
                )
            predictions = [
                Prediction(
                    output_dict=dict(
                        zip(
                            temp_to_predictions,
                            pred_across_sweep,
                        )
                    )
                )
                for pred_across_sweep in zip(*temp_to_predictions.values())
            ]

        else:
            rank_zero_info(f"Running predictor with {kwargs}")
            predictions: List[Prediction] = predictor(  # type: ignore
                annotations=annotations,
                max_gen_len=self.max_gen_len,
                show_progress=show_progress,
                **kwargs,
            )

        rank_zero_info("Generation is done.")
        return self.parse_predictions(predictions, annotations)

    def parse_predictions(
        self,
        predictions: List[Prediction],
        annotations: List[Example],
    ) -> TaskResult:
        # Calculate metrics on MP rank 0 only
        if get_mp_rank() != 0:
            return TaskResult(metrics={})

        raw_results: Dict[str, List[float]] = defaultdict(list)
        for pred, example in zip(predictions, annotations):  # type: ignore
            if getattr(pred, "score", None) is not None:
                example["prediction_score"] = pred.score
            if getattr(pred, "text", None) is not None:
                example["prediction"] = pred.text
            if getattr(pred, "output_dict", None) is not None:
                example["output_dict"] = pred.output_dict
            example.update(self.postprocess_fn(example) if self.postprocess_fn else {})

            example["metrics"] = {}
            for fn in self.metric_fns:
                example["metrics"].update(fn(example))
            for name, value in example["metrics"].items():
                raw_results[name].append(value)

        avg_results = mean_reduce_dict(raw_results, group=get_dp_group())
        object_gather_list = gather_object(annotations, group=get_dp_group())
        object_gather_list = [obj for obj in object_gather_list if obj is not None]
        gathered_results = [x for obj in object_gather_list for x in obj]
        if get_dp_rank() == 0:
            for global_fn in self.global_metric_fns:
                global_metric_dict = global_fn(gathered_results)
                for metric_k, (_avg, _raw) in global_metric_dict.items():
                    for result, raw_val in zip(gathered_results, _raw):
                        result["metrics"][metric_k] = raw_val
                    avg_results[metric_k] = _avg
        else:
            avg_results = {}
            gathered_results = []

        assert len(set(pred["id"] for pred in gathered_results)) == len(
            gathered_results
        )
        return TaskResult(metrics=avg_results, raw_results=gathered_results)


class VideoReasoningTask(ImageReasoningTask):
    """Task to do reasoning over videos instead of images"""

    ...
