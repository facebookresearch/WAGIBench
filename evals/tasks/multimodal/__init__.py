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
import os
from typing import AbstractSet, Any, Callable, Dict

from evals.api import Task, TaskConfig
from evals.tasks.multimodal.base import (
    AndroidAgentTask,
    AndroidAgentTaskConfig,
    ImageReasoningTask,
    ImageReasoningTaskConfig,
    VideoReasoningTask,
    VideoReasoningTaskConfig,
)


class MultimodalTaskRegistry:
    _REGISTRY: Dict[str, Callable[..., TaskConfig]] = {}

    @staticmethod
    def names() -> AbstractSet[str]:
        return MultimodalTaskRegistry._REGISTRY.keys()

    @staticmethod
    def register(name: str, callable: Callable[..., TaskConfig]) -> None:
        if name in MultimodalTaskRegistry._REGISTRY:
            raise ValueError(f"Task {name} already exists.")
        MultimodalTaskRegistry._REGISTRY[name] = callable

    @staticmethod
    def get_config(name: str, **kwargs: Any) -> TaskConfig:
        if name not in MultimodalTaskRegistry._REGISTRY:
            raise ValueError(f"No task registered under the name {name}")
        return MultimodalTaskRegistry._REGISTRY[name](**kwargs)


def build_multimodal_task(config: TaskConfig) -> Task:
    if isinstance(
        config, VideoReasoningTaskConfig
    ):  # video reasoning task config is a ImageReasoningTaskConfig - change this?
        return VideoReasoningTask.from_config(config)
    elif isinstance(config, ImageReasoningTaskConfig):
        return ImageReasoningTask.from_config(config)
    elif isinstance(config, AndroidAgentTaskConfig):
        return AndroidAgentTask.from_config(config)
    else:
        raise ValueError(f"Unable to build task for config {type(config)}")


for file in os.listdir(os.path.dirname(__file__)):
    if file.endswith(".py") and not file.startswith("_"):
        module = file[: file.find(".py")]
        importlib.import_module(f"evals.tasks.multimodal.{module}")
