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

import itertools

import json
from copy import deepcopy
from typing import Any, Dict, Final, List, Optional, Union

import numpy as np

import torch

from evals.api import Example, Prediction  # Example is dict[str, Any]
from evals.predictors.multimodal.maestro_video_mixin import (
    convert_to_free_form_text_representation,
    get_world_state_history,
    MaestroVideoMixin,
)
from evals.tasks.multimodal.maestro_ob2 import post_process_llm_outputs
from evals.utils.common import set_env_for_transformers_version
from evals.utils.distributed import get_local_rank
from evals.utils.prompts import (
    CONTEXT_REFERENCE_JUDGE_PROMPT_V3,
    OB2_DIGITAL_ACTION_GENERATION_PROMPT,
    OB2_MCQ_NOTE_PROMPT,
    OB2_MCQ_PROMPT,
    REFORMAT_PROMPT,
)

from transformers import pipeline


SUPPORTED_APPS = {
    "Calendar": "events",
    "Messaging": "conversations",
    "Search": "searches",
    "Videos": "videos",
    "Notes": "notes",
    "Maps": "maps",
    "Music": "music",
}

SUPPORTED_APPS_KEYS = {
    "Calendar": "key_events",
    "Messaging": "key_conversations",
    "Search": "key_searches",
    "Videos": "key_videos",
    "Notes": "key_notes",
    "Maps": "maps",
    "Music": "music",
}


def flatten(lst):
    for item in lst:
        if isinstance(item, (list, tuple)):  # notice the (list, tuple)
            yield from flatten(item)
        else:
            yield item


class MaestroOB2HuggingFacePredictorMixin(MaestroVideoMixin):
    mcq_prompt: Final[str] = OB2_MCQ_PROMPT
    mcq_note_prompt: Final[str] = OB2_MCQ_NOTE_PROMPT
    generative_prompt: Final[str] = OB2_DIGITAL_ACTION_GENERATION_PROMPT
    judge_prompt: Final[str] = CONTEXT_REFERENCE_JUDGE_PROMPT_V3
    reformat_prompt: Final[str] = REFORMAT_PROMPT
    max_scale: Final[int] = 2

    def __init__(self, **kwargs: Any) -> None:
        device_map = kwargs.pop("device_map", "auto")
        judge_path = kwargs.pop("judge_path", "")

        if self.mode == "generative":
            if device_map == "local_gpu":
                device_map = f"cuda:{get_local_rank()}"

            assert (
                len(judge_path) > 0
            ), "judge_path must be provided for generative mode"

            # Initialize the judge model
            with set_env_for_transformers_version():
                self.judge_model = pipeline(
                    "text-generation",
                    model=judge_path,
                    model_kwargs={"torch_dtype": torch.bfloat16},
                    device_map=device_map,
                )

        super().__init__(**kwargs)

    def preprocess_data(self, example: Example) -> Example:
        """
        Transform for each batch loaded from Image Dataset.

        Args:
            example: The example to preprocess.

        Returns:
            The preprocessed example.
        """
        example = deepcopy(example)
        video, _frame_ids, video_fps = self.load_video(
            video_path=example["video_path"],
            clip_start_sec=example["x"]["context_window"]["start_s"],
            clip_end_sec=example["x"]["context_window"]["end_s"],
        )
        example["x"]["frames"] = video
        return example

    def preprocess_transcription(
        self,
        transcription: List[Dict[str, Any]],
        start_s: Optional[float] = None,
        end_s: Optional[float] = None,
    ) -> str:
        """
        Preprocess the transcription for the model.

        Args:
            transcription: The transcription to preprocess.

        Returns:
            The preprocessed transcription.
        """
        utterances = []
        for utterance in transcription:
            utterance_start_s, utterance_end_s = None, None
            if "start_time" in utterance:
                utterance_start_s = utterance["start_time"].item()
            if "end_time" in utterance:
                utterance_end_s = utterance["end_time"].item()

            if (
                start_s is not None
                and end_s is not None
                and utterance_start_s is not None
                and utterance_end_s is not None
            ):
                if start_s <= utterance_start_s and utterance_end_s <= end_s:
                    valid_utterance = True
                else:
                    valid_utterance = False
            else:
                valid_utterance = True

            if valid_utterance:
                speaker = int(utterance["label"])
                if isinstance(utterance["transcript"], str):
                    text = utterance["transcript"]
                elif isinstance(utterance["transcript"], list):
                    text = utterance["transcript"][0]
                else:
                    raise ValueError(
                        f"Unknown type for transcript: {type(utterance['transcript'])}"
                    )
                phrase = f"Speaker {speaker}: {text}"
                utterances.append(phrase)

        transcript = "\n".join(utterances)

        return transcript

    def build_prompt(
        self,
        options: List[Union[Dict[str, str], str]],
        example: Dict[str, Any],
        video_batch: np.ndarray,
        mode: str = "mcq",
        kwargs: Dict[str, Any] = {},
    ) -> str:
        """
        Build a prompt for the model.

        Args:
            options: The options to use for the prompt.

        Returns:
            The prompt to use.
        """
        if mode == "mcq":
            prompt = self.mcq_prompt
            mcq_options = f"A: {options[0]} \nB: {options[1]} \nC: {options[2]} \nD: {options[3]} \n"
        elif mode == "generative":
            prompt = self.generative_prompt
        else:
            raise NotImplementedError(f"{mode} is not supported")

        # Get the context window
        start_s = example["context_window"]["start_s"].item()
        end_s = example["context_window"]["end_s"].item()

        if kwargs is not None and kwargs.get("vision_state_oracle", False):
            vision_context = ""
            for cue in example["cues"]:
                if any("vision" in x for x in itertools.chain(*cue["modality"])):
                    vision_context = (
                        vision_context + "\n".join(cue["description"]) + "\n"
                    )
            if vision_context:
                prompt = (
                    prompt + f"The vision context for the video:\n" + vision_context
                )

        # Add the transcription to the prompt
        if example["transcription"]:
            transcript = self.preprocess_transcription(
                example["transcription"], start_s, end_s
            )
            if len(transcript) > 0:
                prompt = (  # @TODO template etc this
                    prompt
                    + f"\nTo facilitate the task, the transcription for the video is provided as follows. One of the speakers may be wearing smart glasses. Feel free to ignore the transcription if it is not relelvant. The transcription is:\n"
                    + transcript
                    + "\n"
                )

        # Add longitudinal history to the prompt
        if example["longitudinal_history"]:
            longitudinal_history = ""
            if "world_state_ctx" in example:
                if example["world_state_ctx"] != [""]:
                    longitudinal_history = f"\nHere is a description of the user's current context: {example['world_state_ctx']}"
            longitudinal_history += "\nHere are a list of the user's past actions with corresponding text context to help you in this task:\n"
            history_present = False
            for hist in example["longitudinal_history"]:
                if "transcription" in hist:
                    hist["world_state"]["transcription"] = (
                        self.preprocess_transcription(
                            hist["transcription"],
                            hist["context_window"]["start_s"].item(),
                            hist["context_window"]["end_s"].item(),
                        )
                    )
                h = json.dumps(hist["world_state"], indent=2)
                h = json.dumps(h, indent=2)
                longitudinal_history += f"\n{h}\n"
                history_present = True
            if history_present:
                prompt += longitudinal_history

        # Add the digital state to the prompt
        if example["digital_state"]:
            digital_context = ""
            app_states = example["digital_state"]
            digital_context += f"\nThe current datetime is: {app_states['datetimings']['current_datetime']}\n"
            digital_key_only = app_states.get("digital_key_only", False)
            if digital_key_only:
                for app_name, val in SUPPORTED_APPS_KEYS.items():
                    if app_name in app_states and app_states[app_name]["context"]:
                        digital_context += f"\nCurrent state of the user's {app_name} app:\n{app_states[app_name]['state'][val]}\n"
            else:
                for app_name, val in SUPPORTED_APPS.items():
                    if app_name in app_states:
                        digital_context += f"\nCurrent state of the user's {app_name} app:\n{app_states[app_name]['state'][val]}\n"
            prompt += digital_context

        if kwargs is not None and kwargs.get("audio_state_oracle", False):
            audio_context = ""
            for cue in example["cues"]:
                if any("audio" in x for x in itertools.chain(*cue["modality"])):
                    audio_context = audio_context + "\n".join(cue["description"]) + "\n"
            if audio_context:
                prompt = (
                    prompt
                    + f"The conversation context for the video:\n"
                    + audio_context
                )

        if kwargs is not None and kwargs.get("digital_state_oracle", False):
            digital_context = ""
            for cue in example["cues"]:
                if any("digital" in x for x in itertools.chain(*cue["modality"])):
                    digital_context = (
                        digital_context + "\n".join(cue["description"]) + "\n"
                    )
            if digital_context:
                prompt = (
                    prompt
                    + f"The digital state context for the video:\n"
                    + digital_context
                )

        if mode == "mcq":
            prompt = prompt + "\n" + mcq_options
        elif mode == "generative":
            pass
        else:
            raise NotImplementedError(f"{mode} is not supported")

        caption_video = False if kwargs is None else kwargs.get("caption_video", False)

        if caption_video:
            caption_query = "Let's do this step by step. First, please describe the video and make sure to include all details that are relevant to answer the question."

            caption_query = prompt + "\n" + caption_query

            caption_max_gen_len = 3000

            caption_inputs, caption_gen_kwargs = self.input_transform(
                query=caption_query,
                video_batch=video_batch,
                max_gen_len=caption_max_gen_len,
            )
            with torch.inference_mode():
                outputs = self.model.generate(**caption_inputs, **caption_gen_kwargs)
                response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

            if "llama3v" in str(self).lower():
                splitor = "assistant\n\n"
            elif "cogvlm" in str(self).lower():
                splitor = " Answer:"
            else:
                raise NotImplementedError(
                    f"{type(self)} is not supported for captioning"
                )

            caption = response.split(splitor)[1]

            prompt = f"{prompt}\n\nThe video caption is listed below:\n{caption}\n"

        if mode == "mcq":
            prompt = prompt + "\n\n" + self.mcq_note_prompt
        elif mode == "generative":
            pass
        else:
            raise NotImplementedError(f"{mode} is not supported")

        return prompt

    def _inference(
        self,
        query: str,
        video_batch: np.ndarray,
        max_gen_len: int,
    ) -> str:
        inputs, gen_kwargs = self.input_transform(
            query=query,
            video_batch=video_batch,
            max_gen_len=max_gen_len,
        )

        with torch.no_grad():
            outputs = self.model.generate(**inputs, **gen_kwargs)
            if "input_ids" in inputs:
                outputs = outputs[:, inputs["input_ids"].shape[1] :]
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

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

        # Process the captions into world state history
        world_states = get_world_state_history(
            video_duration_secs, captions, self.chunk_secs
        )
        world_states = convert_to_free_form_text_representation(world_states)
        output_dict = {"world_states": world_states}

        # Create the QA prompt
        prompt = self.mcq_prompt
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

    def predict_batch_generative(
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
        gt_option = batch["y"]
        video_batch = example["frames"]

        if "mcq_set" in example:
            options = example["mcq_set"]
            assert (
                gt_option.nelement() == 1
            ), "Only single sample inference is supported"
            gt_option = gt_option.item()
            gt_option = options[gt_option]
            if "structured_goal" in gt_option:
                gt_option = gt_option["structured_goal"]
        else:
            options = None

        gt_option_str = json.dumps(gt_option, indent=2)
        contexts = []
        contexts_for_logging = []
        for x in example["cues"]:
            modality = ",".join(list(flatten(x["modality"])))
            description = "\n".join(x["description"])
            contexts.append(description)
            contexts_for_logging.append(f"[{modality}] {description}")
        contexts = "\n".join(contexts)
        contexts_for_logging = "\n".join(contexts_for_logging)

        # Forward pass
        query = self.build_prompt(
            options,
            example,
            video_batch,
            mode="generative",
            kwargs=generate_kwargs,
        )
        response = self._inference(query, video_batch, max_gen_len)

        with torch.no_grad():
            # Reformat the response
            query = self.reformat_prompt.format(prediction=response)
            messages = [
                {
                    "role": "system",
                    "content": "You are a very intelligent and helpful assistant.",
                },
                {"role": "user", "content": query},
            ]
            outputs = self.judge_model(
                messages,
                max_new_tokens=1024,
            )
            reformat_response = outputs[0]["generated_text"][-1]["content"]
            _, reformat_response = post_process_llm_outputs(reformat_response, "think")
            reformat_response, _ = post_process_llm_outputs(reformat_response, "answer")

            # Judge the response
            query = self.judge_prompt.format(
                contexts=contexts,
                gt=gt_option_str,
                prediction=reformat_response,
                # max_scale=self.max_scale,
            )
            messages = [
                {
                    "role": "system",
                    "content": "You are a very intelligent and helpful assistant.",
                },
                {"role": "user", "content": query},
            ]
            outputs = self.judge_model(
                messages,
                max_new_tokens=1024,
            )
            judge_response = outputs[0]["generated_text"][-1]["content"]
            judge_cot, judge_response = post_process_llm_outputs(
                judge_response, "think"
            )

        output_dict = {
            "gt": gt_option_str,
            "contexts": contexts_for_logging,
            "response": response,
            "reformat_response": reformat_response,
            "judge_response": judge_response,
            "judge_cot": judge_cot,
            "max_scale": self.max_scale,
        }

        return [Prediction(text=response, output_dict=output_dict)]

    def predict_batch(
        self, batch: Example, max_gen_len: int, generate_kwargs: dict[str, Any]
    ) -> list[Prediction]:
        if self.mode == "default":
            return self.predict_batch_default(batch, max_gen_len, generate_kwargs)
        elif self.mode == "chunked":
            return self.predict_batch_chunked(batch, max_gen_len, generate_kwargs)
        elif self.mode == "generative":
            return self.predict_batch_generative(batch, max_gen_len, generate_kwargs)
        else:
            raise NotImplemented(f"Unknown mode {self.mode}")
