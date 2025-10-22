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
from typing import Any, Dict

from evals.predictors import build_predictor, PredictorRegistry
from evals.predictors.multimodal.api import MultiModalPredictor


_MULTIMODAL_PREDICTOR_CONFIG_MAP = {
    "maestro_ob2_judge": "MaestroOB2JudgeHuggingFacePredictorConfig",
    "maestro_ob2_qwen": "MaestroOB2QwenPredictorConfig",
}


class MultimodalPredictorRegistry(PredictorRegistry):
    _REGISTRY: Dict[str, Any] = {}


for name, config_cls_name in _MULTIMODAL_PREDICTOR_CONFIG_MAP.items():
    name_in = name.split("-")[0]
    MultimodalPredictorRegistry.register(
        name, f"evals.predictors.multimodal.{name_in}.{config_cls_name}"
    )
