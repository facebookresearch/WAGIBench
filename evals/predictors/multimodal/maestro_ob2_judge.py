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

from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Dict, Final, Optional, Tuple, Union

import numpy as np
import torch
from evals.api import Example, Prediction  # Example is dict[str, Any]

from evals.predictors.multimodal.maestro_huggingface import (
    MaestroHuggingFacePredictor,
    MaestroHuggingFacePredictorConfig,
)
from evals.predictors.multimodal.maestro_ob2_mixin import (
    MaestroOB2HuggingFacePredictorMixin,
)
from evals.predictors.multimodal.utils import TGenerationInputs
from evals.tasks.multimodal.maestro_ob2 import post_process_llm_outputs
from evals.utils.common import set_env_for_transformers_version
from evals.utils.distributed import get_local_rank, init_torch_distributed
from evals.utils.prompts import CONTEXT_JUDGE_PROMPT, REFERENCE_JUDGE_PROMPT_V3

from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline


@dataclass(frozen=True)
class MaestroOB2JudgeHuggingFacePredictorConfig(MaestroHuggingFacePredictorConfig):
    """
    Configuration for the MaestroOB2JudgeHuggingFacePredictor.
    """

    device_map: str = "auto"
    max_num_tiles: int = -1  # -1 means no tile resizing
    sample_fps: Optional[float] = None
    mode: str = "default"
    generate_kwargs: Dict[str, Any] = field(default_factory=dict)


class MaestroOB2JudgeHuggingFacePredictor(
    MaestroOB2HuggingFacePredictorMixin,
    MaestroHuggingFacePredictor,
):
    context_judge_prompt: Final[str] = CONTEXT_JUDGE_PROMPT
    reference_judge_prompt: Final[str] = REFERENCE_JUDGE_PROMPT_V3

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)

    @staticmethod
    def from_config(
        config: MaestroOB2JudgeHuggingFacePredictorConfig,
    ) -> "MaestroOB2JudgeHuggingFacePredictor":
        """
        Construct a MaestroOB2JudgeHuggingFacePredictor from a config.

        Args:
            config: The config to use for construction.
        Returns:
            The constructed HuggingFacePredictor.
        """
        model, _ = MaestroOB2JudgeHuggingFacePredictor.get_tokenizer_and_model(
            MaestroOB2JudgeHuggingFacePredictor,
            model_path=config.model_path,
            dtype=torch.float16,
            config=config,
        )

        return MaestroOB2JudgeHuggingFacePredictor(
            model=model,
            tokenizer=None,
            dtype=torch.float16,
            device=config.device,
            number_of_samples=-1,
            sample_fps=-1,
            image_processor=None,
        )

    def input_transform(
        self, query: str, video_batch: np.ndarray, max_gen_len: int = 20
    ):
        pass

    def preprocess_data(self, example: Example) -> Example:
        example = deepcopy(example)
        return example

    @staticmethod
    def _get_tokenizer_and_model(
        model_path: str,
        dtype: torch.dtype = torch.float16,
        device: Union[str, torch.device] = "cuda",
        config: Optional[MaestroHuggingFacePredictorConfig] = None,
    ) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
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
        device_map = "auto"
        if config is not None and hasattr(config, "device_map"):
            if config.device_map == "local_gpu":
                device_map = f"cuda:{get_local_rank()}"

        # Initialize the judge model
        tokenizer = None
        with set_env_for_transformers_version():
            model = pipeline(
                "text-generation",
                model=model_path,
                model_kwargs={"torch_dtype": torch.bfloat16},
                device_map=device_map,
            )

        return model, tokenizer

    def predict_batch(
        self, batch: Example, max_gen_len: int, generate_kwargs: dict[str, Any]
    ) -> list[Prediction]:
        if generate_kwargs is not None and "judge_input" in generate_kwargs:
            judge_input = generate_kwargs["judge_input"]
        else:
            judge_input = "context_reference"

        # Get reference
        gt_option = batch["y"]
        gt_option_str = json.dumps(gt_option, indent=2)

        # Get contexts
        if "world_states" in judge_input:
            contexts = []
            for x in batch["x"]["world_states"]:
                if "Summary" in x:
                    description = "\n".join(x["Summary"])
                else:
                    description = ""
                contexts.append(description)
            contexts = "\n".join(contexts)
        else:
            contexts = []
            for x in batch["x"]["cues"]:
                description = "\n".join(x["description"])
                contexts.append(description)
            contexts = "\n".join(contexts)

        # Loop over generations from different models
        output_dict = {}
        for model_name, generation in batch["generations"].items():
            with torch.no_grad():
                # Judge the response
                if judge_input in {"context_reference", "world_states_reference"}:
                    query = self.judge_prompt.format(
                        contexts=contexts,
                        gt=gt_option_str,
                        prediction=generation,
                    )
                elif judge_input in {"context", "world_states"}:
                    query = self.context_judge_prompt.format(
                        contexts=contexts,
                        prediction=generation,
                    )
                elif judge_input == "reference":
                    query = self.reference_judge_prompt.format(
                        gt=gt_option_str,
                        prediction=generation,
                    )
                else:
                    raise ValueError(f"Unknown judge input: {judge_input}")

                messages = [
                    {
                        "role": "system",
                        "content": "You are a very intelligent and helpful assistant.",
                    },
                    {"role": "user", "content": query},
                ]
                outputs = self.model(
                    messages,
                    max_new_tokens=1024,
                )
                judge_response = outputs[0]["generated_text"][-1]["content"]
                judge_cot, judge_response = post_process_llm_outputs(
                    judge_response, "think"
                )

            output_dict[model_name] = {
                "response": generation,
                "judge_response": judge_response,
                "judge_cot": judge_cot,
                "max_scale": self.max_scale,
            }

        return [Prediction(text="", output_dict=output_dict)]
