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

from typing import AbstractSet, Dict, Type

from evals.api import Predictor, PredictorConfig


_PREDICTOR_CONFIG_MAP = {}


class PredictorRegistry:
    _REGISTRY: Dict[str, str] = {}

    @classmethod
    def names(cls) -> AbstractSet[str]:
        return cls._REGISTRY.keys()

    @classmethod
    def register(cls, name: str, module_path: str) -> None:
        if name in cls._REGISTRY:
            raise ValueError(f"Predictor {name} already exists.")
        cls._REGISTRY[name] = module_path

    @classmethod
    def get_config_cls(cls, name: str) -> Type[PredictorConfig]:
        if name not in cls._REGISTRY:
            raise ValueError(f"No predictor registered under the name {name}")

        module_path, config_cls_name = cls._REGISTRY[name].rsplit(".", 1)
        module = __import__(module_path, fromlist=[config_cls_name])
        return getattr(module, config_cls_name)


def build_predictor(config: PredictorConfig) -> Predictor:
    config_cls_name = config.__class__.__name__
    try:
        module = __import__(config.__class__.__module__, fromlist=[config_cls_name])
        cls_name = config.__class__.__name__.replace("Config", "")
        return getattr(module, cls_name).from_config(config)
    except ImportError as err:
        raise ValueError(f"Import error found for config {config_cls_name}: {err}")


for name, config_cls_name in _PREDICTOR_CONFIG_MAP.items():
    PredictorRegistry.register(name, f"evals.predictors.{name}.{config_cls_name}")
