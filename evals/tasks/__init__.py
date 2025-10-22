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

# flake8: noqa
import importlib
from functools import partial
from itertools import product
from pathlib import Path
from typing import AbstractSet, Any, Callable, Dict, Iterable, Optional, Tuple, Union

from evals.api import Task, TaskConfig
from evals.tasks.base.choice import ChoiceTask, ChoiceTaskConfig
from evals.tasks.base.generation import GenerationTask, GenerationTaskConfig
from evals.tasks.base.group import GroupTask, GroupTaskConfig
from evals.tasks.base.model_based import ModelBasedEvalTask, ModelBasedEvalTaskConfig
from evals.tasks.base.multiturn import MultiTurnGenerationTaskConfig
from evals.tasks.base.parallel import (
    ParallelGenerationTask,
    ParallelGenerationTaskConfig,
)


class TaskRegistry:
    _REGISTRY: Dict[str, Callable[..., TaskConfig]] = {}

    @staticmethod
    def names() -> AbstractSet[str]:
        return TaskRegistry._REGISTRY.keys()

    @staticmethod
    def register(name: str, callable: Callable[..., TaskConfig]) -> None:
        if name in TaskRegistry._REGISTRY:
            raise ValueError(f"Task {name} already exists.")
        TaskRegistry._REGISTRY[name] = callable

    @staticmethod
    def get_config(name: str, **kwargs: Any) -> TaskConfig:
        if name not in TaskRegistry._REGISTRY:
            raise ValueError(f"No task registered under the name {name}")
        return TaskRegistry._REGISTRY[name](**kwargs)

    @staticmethod
    def reset() -> None:
        TaskRegistry._REGISTRY = {}


def register_task(
    name: str,
    parameters: Optional[Dict[Union[str, Tuple[str, ...]], Iterable[Any]]] = None,
) -> Callable[[Callable[..., TaskConfig]], Callable[..., TaskConfig]]:
    """Register the task name with the decorated task configuration callable."""

    def register(callable: Callable[..., TaskConfig]) -> Callable[..., TaskConfig]:
        if parameters is None:
            TaskRegistry.register(name, callable)
        else:
            for values in product(*parameters.values()):
                param_dict: Dict[str, Any] = {}
                for keys, value in zip(parameters.keys(), values):
                    if isinstance(keys, tuple):
                        param_dict.update(zip(keys, value))
                    else:
                        param_dict[keys] = value
                task_name = name.format(**param_dict)
                TaskRegistry.register(task_name, partial(callable, **param_dict))
        return callable

    return register


def build_task(config: TaskConfig) -> Task:
    config_cls_name = config.__class__.__name__
    try:
        module = __import__(config.__class__.__module__, fromlist=[config_cls_name])
        cls_name = config.__class__.__name__.replace("Config", "")
        return getattr(module, cls_name).from_config(config)
    except ImportError:
        raise ValueError("No task class found for {config_cls_name}")


# Recursively import all python modules except those starting with an underscore
base_dir = Path(__file__).parent
for file in base_dir.rglob("*.py"):
    if not file.name.startswith("_") and "multimodal" not in str(file):
        relative_path = file.relative_to(base_dir).with_suffix("")
        module_path = ".".join(relative_path.parts)
        importlib.import_module(f"evals.tasks.{module_path}")
