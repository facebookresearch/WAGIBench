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
from typing import Any, Dict, List, Optional, Sequence

from evals.api import (
    Example,
    GlobalMetricFn,
    MetricFn,
    Predictor,
    Task,
    TaskConfig,
    TaskResult,
)
from evals.tasks.base.generation import GenerationTask, GenerationTaskConfig
from evals.utils import (
    all_gather_object,
    ExampleSelector,
    gather_object,
    get_dp_group,
    get_dp_rank,
    get_dp_size,
    get_rank,
    load_jsonl,
    mean_reduce_dict,
    rank_zero_info,
)
from numpy.random import RandomState


@dataclass
class ModelBasedEvalTaskConfig(TaskConfig):
    generator_task_config: GenerationTaskConfig
    judge_task_config: GenerationTaskConfig
    metric_fns: Optional[Sequence[MetricFn]] = None
    global_metric_fns: Optional[Sequence[GlobalMetricFn]] = None


class ModelBasedEvalTask(Task):
    def __init__(
        self,
        generator_task_config: GenerationTaskConfig,
        judge_task_config: GenerationTaskConfig,
        metric_fns: Optional[Sequence[MetricFn]],
        global_metric_fns: Optional[Sequence[GlobalMetricFn]],
    ) -> None:
        self.generator_task_config = generator_task_config
        self.judge_task_config = judge_task_config
        self.metric_fns: Sequence[MetricFn] = metric_fns or []
        self.global_metric_fns: Sequence[GlobalMetricFn] = global_metric_fns or []

    @staticmethod
    def from_config(cfg: ModelBasedEvalTaskConfig) -> "ModelBasedEvalTask":
        return ModelBasedEvalTask(
            cfg.generator_task_config,
            cfg.judge_task_config,
            cfg.metric_fns,
            cfg.global_metric_fns,
        )

    @staticmethod
    def from_iterator(
        cfg: GenerationTaskConfig, dataset: List[Example]
    ) -> "GenerationTask":
        # We don't load data from a JSONL file to prepare the inputs
        # for generator, we use the generator outputs as the inputs
        # This method is meant to be used by the Judge
        few_shot_examples = cfg.few_shot_examples
        if cfg.num_few_shot > 0 and few_shot_examples is None:
            assert cfg.few_shot_file is not None
            few_shot_examples = load_jsonl(filename=cfg.few_shot_file)

        return GenerationTask(
            dataset=dataset,  # type: ignore
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
        )

    def run(
        self,
        predictor: Predictor,
        random_state: Optional[RandomState] = None,
        max_samples: Optional[int] = None,
        show_progress: bool = False,
        temperature: float = 1.0,
        top_p: float = 0.6,
        top_k: int = 0,
        **kwargs: Any,
    ) -> Any:
        rank_zero_info("Running generator...")
        generator_task = GenerationTask.from_config(self.generator_task_config)
        generator_outputs = generator_task.run(
            predictor=predictor,
            top_p=top_p,
            top_k=top_k,
            temperature=temperature,
            random_state=random_state,
            max_samples=max_samples,
            show_progress=show_progress,
            **kwargs,
        )
        assert generator_outputs.raw_results is not None
        object_gather_list = all_gather_object(
            generator_outputs.raw_results, group=get_dp_group()
        )
        return [x for obj in object_gather_list for x in obj]

    def judge(
        self,
        output_from_generator: List[Example],
        predictor: Predictor,
        random_state: Optional[RandomState] = None,
        max_samples: Optional[int] = None,
        show_progress: bool = False,
        **kwargs: Any,
    ) -> Any:
        from evals.predictors.llm_inference import LLMInferencePredictor
        from evals.predictors.llm_inference_reward import LLMInferenceRewardScorer

        rank_zero_info("Running Judge...")
        if isinstance(predictor, LLMInferencePredictor):
            judge_task = ModelBasedEvalTask.from_iterator(
                self.judge_task_config, output_from_generator
            )
            judge_outputs = judge_task.run(
                predictor=predictor,
                random_state=random_state,
                max_samples=max_samples,
                show_progress=show_progress,
                **kwargs,
            )
        elif isinstance(predictor, LLMInferenceRewardScorer):
            raw_results: Dict[str, List[float]] = defaultdict(list)
            prompts = [
                self.judge_task_config.prompt_fn(ex) for ex in output_from_generator
            ]
            prompts = list(islice(prompts, get_dp_rank(), None, get_dp_size()))
            rewards = predictor(
                prompts, max_prompt_len=self.judge_task_config.max_prompt_len
            )
            results = [
                ex | {"reward": rewards[i]}
                for i, ex in enumerate(output_from_generator)
            ]
            raw_results["reward"] = rewards
            avg_results = mean_reduce_dict(raw_results, group=get_dp_group())
            if len(self.global_metric_fns) > 0:
                object_gather_list = gather_object(self.dataset, group=get_dp_group())
                if get_rank() == 0:
                    gathered_results = [x for obj in object_gather_list for x in obj]
                    for ag_fn in self.global_metric_fns:
                        avg_results.update(ag_fn(gathered_results))
            judge_outputs = TaskResult(metrics=avg_results, raw_results=results)
        else:
            raise NotImplementedError
        return judge_outputs
