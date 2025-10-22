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

import os
from abc import abstractmethod
from dataclasses import dataclass
from enum import Enum

from typing import Optional, Tuple, Union

import numpy as np
import torch
from evals.predictors.multimodal.api import MultiModalPredictor
from evals.predictors.multimodal.utils import TGenerationInputs
from evals.utils.common import set_env_for_transformers_version
from evals.utils.distributed import init_torch_distributed

from transformers import AutoModelForCausalLM, AutoProcessor, AutoTokenizer


class ConversationMode(str, Enum):
    LLAMA_2 = "llama_2"


@dataclass(frozen=True)
class MaestroHuggingFacePredictorConfig:
    model_path: str
    number_of_samples: int = 1
    sample_fps: Optional[float] = 1.0
    use_preprocess: bool = False
    device: Union[str, torch.device] = "cuda"
    judge_path: str = ""

    def __post_init__(self) -> None:
        """Ensure that the model path exists and that CUDA is available if using a GPU."""
        if self.device == "cuda" and not torch.cuda.is_available():
            raise ValueError("CUDA is not available")


class MaestroHuggingFacePredictor(
    MultiModalPredictor,
):
    def __init__(
        self,
        model: AutoModelForCausalLM,
        tokenizer: AutoTokenizer,
        dtype: Union[torch.dtype, str],
        device: str = "cuda",
        use_preprocess: bool = False,
        batch_size: int = 1,
        number_of_samples: int = 1,
        sample_fps: Optional[float] = None,
        image_processor: Optional[AutoProcessor] = None,
    ) -> None:
        """
        Run a HuggingFace model on data from the Maestro OB2 dataset.
        Args:
            model: The model to use for prediction.
            tokenizer: The tokenizer to use for prediction.
            dtype: The dtype to use for prediction.
            device: The device to use for prediction.
            batch_size: The batch size to use for prediction.
            number_of_samples: The number of subdivisions in the video. ELA typically uses 4, CogVLM uses 24.

        Returns:
            None
        """
        super().__init__()
        self.number_of_samples: int = number_of_samples
        self.sample_fps = sample_fps
        self.dtype: torch.dtype = dtype
        self.model: AutoModelForCausalLM = model
        self.tokenizer: AutoTokenizer = tokenizer
        self.image_processor: Optional[AutoProcessor] = image_processor
        self.device: Union[str, torch.device] = device
        self.batch_size: int = batch_size

    def get_tokenizer_and_model(  # type: ignore[misc]
        predictor,
        model_path: str,
        dtype: torch.dtype = torch.float16,
        device: Union[str, torch.device] = "cuda",
        config: Optional[MaestroHuggingFacePredictorConfig] = None,
    ) -> Tuple[AutoTokenizer, AutoModelForCausalLM]:
        """
        Load the model and tokenizer which are both pretrained.

        Args:
            model_path: The path to the model.
            model_type: The type of model to load.
            dtype: The dtype to use for the model.
            device: The device to use for the model.

        Returns:
            A tuple containing the model and tokenizer.
        """
        init_torch_distributed()

        # NOTE: This is a hack to get around the fact that the model is not properly leverageing device_map.
        with set_env_for_transformers_version():
            # Call the specific predictor's get_tokenizer_and_model method
            assets = predictor._get_tokenizer_and_model(
                model_path, dtype, device, config
            )

        return (*assets,)

    @abstractmethod
    def input_transform(
        self,
        query: str,
        video_batch: Optional[np.ndarray] = None,
        max_gen_len: int = 20,
    ) -> TGenerationInputs:
        """
        Build the input for the model.

        Args:
            query: The query to use for prediction.
            video_batch: The video batch to use for prediction.
            max_gen_len: The maximum length of the generated text.

        Returns:
            The input for the model.
        """
        pass
