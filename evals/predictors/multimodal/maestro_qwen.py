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

from dataclasses import dataclass
from typing import Optional, Tuple, Union

import numpy as np
import torch

from evals.predictors.multimodal.maestro_huggingface import (
    MaestroHuggingFacePredictor,
    MaestroHuggingFacePredictorConfig,
)
from evals.predictors.multimodal.utils import TGenerationInputs
from evals.utils.distributed import get_local_rank, init_torch_distributed
from qwen_vl_utils import process_vision_info
from torchvision import transforms

from transformers import (
    AutoModelForCausalLM,
    AutoProcessor,
    AutoTokenizer,
    Qwen2_5_VLForConditionalGeneration,
)


@dataclass(frozen=True)
class MaestroQwenPredictorConfig(MaestroHuggingFacePredictorConfig):
    """
    Configuration for the MaestroQwenPredictorConfig.
    """

    input_size: int = 448
    device_map: str = "auto"
    sample_fps: Optional[float] = None
    mode: str = "default"


class MaestroQwenPredictor(
    MaestroHuggingFacePredictor,
):
    def __init__(self, **kwargs):
        config = kwargs.pop("config")
        mode = kwargs.pop("mode")
        super().__init__(**kwargs)

    @staticmethod
    def _get_tokenizer_and_model(
        model_path: str,
        dtype: torch.dtype = torch.float16,
        device: Union[str, torch.device] = "cuda",
        config: Optional[MaestroQwenPredictorConfig] = None,
    ) -> Tuple[AutoModelForCausalLM, AutoTokenizer, AutoProcessor]:
        """
        Load the model and tokenizer which are both pretrained.

        Args:
            model_path: The path to the model.
            model_type: The type of model to load. Example: ela_v1.
            dtype: The dtype to use for the model.

        Returns:
            The model and tokenizer.
        """
        init_torch_distributed()

        if config.device_map == "local_gpu":
            device_map = f"cuda:{get_local_rank()}"
        elif config.device_map == "auto":
            device_map = "auto"
        else:
            raise ValueError(f"Unknown device map: {config.device_map}")

        tokenizer = None
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype="auto",
            device_map=device_map,
        ).eval()
        image_processor = AutoProcessor.from_pretrained(model_path)

        return tokenizer, model, image_processor

    def input_transform(
        self,
        query: str,
        video_batch: np.ndarray,
        max_gen_len: int = 20,
    ) -> TGenerationInputs:
        """
        Build the input for the model.

        Args:
            query: The query to use for the prompt.
            video: The video to use for the prompt. Use batch size of 1 for this method.
            max_gen_len: The maximum length of generation.

        Returns:
            The input to feed into the model's geneate method.
        """
        if len(video_batch) != 1:
            raise ValueError("Batch size must be 1 for this method.")

        to_pil_image = transforms.ToPILImage()
        video_batch = video_batch.squeeze(
            dim=0
        )  # Remove the batch dimension as we're using a batch size of 1.

        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "video",
                        "video": [
                            to_pil_image(_image.permute(2, 0, 1))
                            for _image in video_batch
                        ],
                    },
                    {"type": "text", "text": query},
                ],
            }
        ]

        text = self.image_processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        image_inputs, video_inputs, video_kwargs = process_vision_info(
            messages, return_video_kwargs=True
        )

        inputs = self.image_processor(
            text=[text],
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
            **video_kwargs,
        )
        inputs = inputs.to("cuda")

        return inputs, {"max_new_tokens": max_gen_len}
