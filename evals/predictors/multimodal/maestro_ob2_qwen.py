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

from dataclasses import dataclass, field
from typing import Any, Dict

import numpy as np
import torch

import yaml
from evals.api import Example, Prediction  # Example is dict[str, Any]

from evals.predictors.multimodal.maestro_ob2_mixin import (
    MaestroOB2HuggingFacePredictorMixin,
)
from evals.predictors.multimodal.maestro_qwen import (
    MaestroQwenPredictor,
    MaestroQwenPredictorConfig,
)
from evals.predictors.multimodal.maestro_video_mixin import (
    convert_to_free_form_text_representation,
    get_world_state_history,
)


@dataclass(frozen=True)
class MaestroOB2QwenPredictorConfig(MaestroQwenPredictorConfig):
    """
    Configuration for the MaestroOB2QwenPredictor.
    """

    number_of_samples: int = 32
    generate_kwargs: Dict[str, Any] = field(default_factory=dict)


class MaestroOB2QwenPredictor(
    MaestroOB2HuggingFacePredictorMixin,
    MaestroQwenPredictor,
):
    def __init__(self, **kwargs: Any) -> None:
        self.mode = kwargs.get("mode", "default")
        self.sample_fps = kwargs.get("sample_fps", None)
        if self.mode == "chunked":
            # Load caption prompt
            caption_prompt_path = "scripts/prompts/world_state/caption.yaml"
            with open(caption_prompt_path, "r") as f:
                data = yaml.safe_load(f)
            self.caption_prompt = data["prompt"]

        super().__init__(**kwargs)

    @staticmethod
    def from_config(
        config: MaestroOB2QwenPredictorConfig,
    ) -> "MaestroQwenPredictor":
        """
        Construct a MaestroOB2HuggingFacePredictor from a config.

        Args:
            config: The config to use for construction.
        Returns:
            The constructed HuggingFacePredictor.
        """
        tokenizer, model, image_processor = (
            MaestroQwenPredictor.get_tokenizer_and_model(
                MaestroQwenPredictor,
                model_path=config.model_path,
                dtype=torch.float16,
                config=config,
            )
        )

        return MaestroOB2QwenPredictor(
            model=model,
            tokenizer=tokenizer,
            dtype=torch.float16,
            device=config.device,
            number_of_samples=config.number_of_samples,
            sample_fps=config.sample_fps,
            image_processor=image_processor,
            mode=config.mode,
            judge_path=config.judge_path,
            config=config,
        )

    def _inference(
        self,
        query: str,
        video_batch: np.ndarray,
        max_gen_len: int,
    ) -> str:
        max_gen_len = 40960
        inputs, gen_kwargs = self.input_transform(
            query=query,
            video_batch=video_batch,
            max_gen_len=max_gen_len,
        )

        with torch.no_grad():
            generated_ids = self.model.generate(**inputs, max_new_tokens=max_gen_len)
            generated_ids_trimmed = [
                out_ids[len(in_ids) :]
                for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            response = self.image_processor.batch_decode(
                generated_ids_trimmed,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )[0]

        return response

    def predict_batch_default(
        self, batch: Example, max_gen_len: int, generate_kwargs: dict[str, Any]
    ) -> list[Prediction]:
        """
        Predict the answer for a multiple choice question about the video provided in batch.

        Args:
            batch: The example(s) to predict.
            max_gen_len: The maximum length of generation.
            generate_kwargs: Additional arguments to be passed to the generator - unused.

        Returns:
            The predictions about the videos in the batch.
        """
        # Use batch size 1
        example = batch["x"]
        video_batch = example["frames"]
        options = example["mcq_set"]
        query = self.build_prompt(options, example, video_batch, kwargs=generate_kwargs)
        response = self._inference(query, video_batch, max_gen_len)
        return [Prediction(text=response, output_dict={})]

    def predict_batch_chunked(
        self, batch: Example, max_gen_len: int, generate_kwargs: dict[str, Any]
    ) -> list[Prediction]:
        """
        Predict the answer for a multiple choice question about the video provided in batch.

        Args:
            batch: The example(s) to predict.
            max_gen_len: The maximum length of generation.
            generate_kwargs: Additional arguments to be passed to the generator - unused.

        Returns:
            The predictions about the videos in the batch.
        """
        # Use batch size 1
        example = batch["x"]
        video_batch = example["frames"]
        options = example["mcq_set"]

        chunk_frames = int(self.chunk_secs * self.sample_fps)
        N = video_batch.shape[1]
        video_duration_secs = N / self.sample_fps

        # Loop over chunks to caption the video
        captions = []
        for start_frames in range(0, N, chunk_frames):
            video_chunk = video_batch[
                :, start_frames : start_frames + chunk_frames, ...
            ]
            response = self._inference(self.caption_prompt, video_chunk, 2048)
            captions.append(response)

        # Clear the cache
        torch.cuda.empty_cache()

        # Process the captions into world state history
        world_states = get_world_state_history(
            video_duration_secs, captions, self.chunk_secs
        )
        world_states = convert_to_free_form_text_representation(world_states)
        output_dict = {"world_states": world_states}

        # Create the QA prompt
        prompt = self.system_prompt
        mcq_query = (
            f"A: {options[0]} \nB: {options[1]} \nC: {options[2]} \nD: {options[3]} \n"
        )
        prompt = f"{prompt}To assist answering the question, detailed world states for video segments are listed below:\n\n{world_states}"
        prompt = prompt + "Below are the options for the question:\n\n" + mcq_query
        prompt = (
            prompt
            + "\nNow let's answer the question. Your response should contain only the option letter A, B, C, or D. Only respond with one letter. Do not repeat the option.\n"
        )

        # Generate the response
        frame_id_list = np.linspace(0, N - 1, self.number_of_samples, dtype=int)
        video_chunk = video_batch[:, frame_id_list, ...]
        response = self._inference(prompt, video_chunk, 200)
        return [Prediction(text=response, output_dict=output_dict)]
