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
import logging
import os
import re

from functools import partial
from typing import Any, Dict, Final, List, Optional

from evals.tasks.multimodal import MultimodalTaskRegistry, VideoReasoningTaskConfig
from evals.utils import evaluate

logger: logging.Logger = logging.getLogger()


OPTIONS: Final[Dict[str, int]] = (
    {  # Map multiple choice answers to ints. OB2 specifies answers as ints.
        "A": 0,
        "B": 1,
        "C": 2,
        "D": 3,
    }
)
CONTEXT_WINDOW_LEN_BUCKET: Final[Dict[str, List[int]]] = {
    "short": [0, 30],
    "medium": [30, 60],
    "long": [60, 999999999],
}
LLM_JUDGE_SCORE_MAP: Final[Dict[float, float]] = {
    2.0: 1.0,
    1.0: 0.5,
    0.0: 0.0,
}

START_THINKING_TAG = "<think>"
END_THINKING_TAG = "</think>"


def load_ob2_data(
    jsonl_dataset_path: str,
    dataset_dir: str,
    transcription_dataset_path: Optional[str] = None,
    caption_directory: Optional[str] = None,
    digital_state_path: Optional[str] = None,
    longitudinal_history_path: Optional[str] = None,
    digital_key_only: Optional[str] = "0",
    longitudinal_positive_only: Optional[str] = "0",
) -> List[Dict[str, Any]]:
    """
    Load a JSONL file containing the OB2 data.

    Args:
        jsonl_filepath: Path to the JSONL file of the OB2 dataset.
        transcription_dataset_path: Path to the JSONL file of the transcription dataset.
        dataset_dir: Path to the root directory of the dataset.
        caption_directory: Path to the captions directory.
        digital_state_path: Path to the JSON file with digital states.
        longitudinal_history_path: Path to the JSON file with longitudinal history.
        digital_key_only: A string 0/1 flag to only use the key components of digital state and avoid all noise.
        longitudinal_positive_only: A string 0/1 flag to only use the positive examples of longitudinal history.

    Returns:
        A list of dictionaries containing the OB2 data.
    """
    print("Loading data from", jsonl_dataset_path)
    print("Loading data from", dataset_dir)
    if digital_state_path:
        print("Loading digital states from", digital_state_path)

    if longitudinal_history_path:
        print("Loading longitudinal history from", longitudinal_history_path)

    new_data = []
    with open(jsonl_dataset_path, "r") as f:
        data = json.load(f)

    if transcription_dataset_path:
        with open(transcription_dataset_path, "r") as f:
            transcriptions = json.load(f)
    else:
        transcriptions = {}

    if digital_state_path:
        with open(digital_state_path, "r") as f:
            digital_states = json.load(f)
        # Re-index them by client_tag + scenario_shortname
        digital_states = {
            v["meta"]["scenario"]["shortname"] + "_" + v["meta"]["client_tag"]: v
            for (k, v) in digital_states.items()
        }
    else:
        digital_states = {}

    if longitudinal_history_path:
        with open(longitudinal_history_path, "r") as f:
            longitudinal_history = json.load(f)
        # Re-index them by client_tag + scenario_shortname
        longitudinal_history = {
            f"{v['meta']['client_tag']}_{v['meta']['scenario']['shortname']}": v
            for v in longitudinal_history
        }
        lgl_dict = {
            k: v["longitudinal_history"] for k, v in longitudinal_history.items()
        }
    else:
        lgl_dict = {}

    success_ctr, success_ctr_lgl, setup_lgl_ctr = 0, 0, 0
    anchor_world_state = 0
    ctr = 0
    for row_id, row in enumerate(data):
        row["id"] = row_id
        ctr += 1
        if "client_tag" in row["x"]:
            video_file_name = f'{row["x"]["client_tag"]}.mp4'
            row["video_path"] = os.path.join(
                dataset_dir,
                video_file_name,
            )
        elif "video_filename" in row["x"]:
            video_file_name = f'{row["x"]["video_filename"]}'
            row["video_path"] = os.path.join(dataset_dir, video_file_name)
            if video_file_name in transcriptions:
                row["x"]["transcription"] = transcriptions[video_file_name][
                    "transcription"
                ]
            else:
                row["x"][
                    "transcription"
                ] = False  # Torch collate function hates None so use False to indicate no transcription.
        else:
            raise ValueError("No video path found in row")

        if caption_directory:
            caption_name = os.path.splitext(os.path.basename(row["video_path"]))[0]
            row["caption_path"] = os.path.join(
                caption_directory, f"{caption_name}.json"
            )

        row["x"][
            "digital_state"
        ] = False  # Torch collate function hates None so use False to indicate no digital state.
        if digital_state_path:
            idx = str(
                row["meta"]["scenario"]["shortname"] + "_" + row["meta"]["client_tag"]
            )
            if idx in digital_states:
                success_ctr += 1
                row["x"]["digital_state"] = digital_states[idx]["x"].get(
                    "app_states", False
                )
                row["x"]["digital_state"]["digital_key_only"] = (
                    False if digital_key_only == "0" else True
                )

        row["x"]["longitudinal_history"] = False
        if longitudinal_history_path:
            idx = f"{row['meta']['client_tag']}_{row['meta']['scenario']['shortname']}"
            if idx in lgl_dict:
                success_ctr_lgl += 1
                setup_exists = False
                for hist_item in lgl_dict[idx]:
                    h_fn = f"{hist_item['client_tag']}.mp4"
                    if hist_item["history_type"] == "setup":
                        setup_exists = True
                    if h_fn in transcriptions:
                        hist_item["transcription"] = transcriptions[h_fn][
                            "transcription"
                        ]
                if setup_exists:
                    setup_lgl_ctr += 1
                row["x"]["longitudinal_positive_only"] = (
                    True if longitudinal_positive_only == "1" else False
                )

                if longitudinal_positive_only == "1":
                    row["x"]["longitudinal_history"] = [
                        hist_item
                        for hist_item in lgl_dict[idx]
                        if hist_item["history_type"] == "setup"
                    ]
                else:
                    row["x"]["longitudinal_history"] = lgl_dict[idx]
                # pyre-ignore
                row["x"]["world_state_ctx"] = longitudinal_history[idx]["x"][
                    "world_state_ctx"
                ]
                if row["x"]["world_state_ctx"] != "":
                    anchor_world_state += 1

        # mask out the context window as the videos are already clipped
        row["x"]["context_window"]["start_s"] = -1.0
        row["x"]["context_window"]["end_s"] = -1.0

        new_data.append(row)

    if digital_state_path:
        print(f"Success counter (Digital) = {success_ctr}/{ctr}\n")

    if longitudinal_history_path:
        print(f"Success counter (Longitudinal) = {success_ctr_lgl}/{ctr}\n")
        print(f"Setup counter (Longitudinal) = {setup_lgl_ctr}/{success_ctr_lgl}\n")
        print(f"Anchors with world state = {anchor_world_state}/{success_ctr_lgl}")

    return new_data


def split_think_and_response(input_string: str) -> tuple[str, str]:
    """
    Split the response into think and response.

    Args:
        response: The response to split.

    Returns:
        The think and response.
    """
    input_string = input_string.strip()

    # Find the positions of the start and end tags
    start_index = input_string.find(START_THINKING_TAG) + len(START_THINKING_TAG)
    end_index = input_string.find(END_THINKING_TAG)
    # Extract the content between the tags
    think = input_string[start_index:end_index]
    # Extract the rest of the string after the </think> tag
    if end_index == -1:
        print("No enough tokens to finish thinking, consider increasing max_gen_len")
        response = ""
    else:
        response = input_string[end_index + len(END_THINKING_TAG) :].strip()

    return think, response


def post_process_llm_outputs(response, tag, force_strip_tag=True):
    start_tag = f"<{tag}>"
    end_tag = f"</{tag}>"
    # pattern = rf"{start_tag}(.*?){end_tag}"
    pattern = rf"(?:{start_tag}|{end_tag})([^<>]+)(?:{end_tag}|{start_tag})"
    target = re.findall(pattern, response, re.S)
    if len(target) > 0:
        target = target[0].strip()
        remainder = re.sub(pattern, "", response, flags=re.S).strip()
    elif end_tag in response:
        splits = response.split(end_tag)
        target = splits[0].strip()
        remainder = splits[-1].strip()
    elif start_tag in response:
        splits = response.split(start_tag)
        target = splits[-1].strip()
        remainder = splits[0].strip()
    else:
        target = response.strip()
        remainder = ""
    if force_strip_tag:
        target = target.replace(start_tag, "").replace(end_tag, "").strip()
        remainder = remainder.replace(start_tag, "").replace(end_tag, "").strip()
    return target, remainder


def parse_judge_score(judge_response, tag):
    try:
        judge_response, _ = post_process_llm_outputs(judge_response, tag)
        judge_score = float(judge_response)
        followed_instructions = True
    except Exception as e:
        logger.error(f"Unparsable judge response: {judge_response}")
        judge_score = 0.0
        followed_instructions = False
    return judge_score, followed_instructions


def post_process(model_prediction_and_annotation: Dict[str, Any]) -> Dict[str, Any]:
    """Process what comes out of the model.

    Args:
        model_prediction_and_annotation: The output from the predictor. This dict likely has the rest of the original JSONL row within it from the OB2 dataset and the video frames.

    Returns:
        with whitespace and some special characters removed, and the rest of the dataset row sans video frames.
    """
    model_prediction_and_annotation["raw_prediction"] = model_prediction_and_annotation[
        "prediction"
    ]
    prediction = model_prediction_and_annotation["prediction"]
    if (
        "output_dict" in model_prediction_and_annotation
        and len(model_prediction_and_annotation["output_dict"]) > 0
        and "judge_response" in model_prediction_and_annotation["output_dict"]
    ):
        # Generative OB2
        output_dict = model_prediction_and_annotation["output_dict"]
        judge_score, judge_followed_instructions = parse_judge_score(
            output_dict["judge_response"], tag="score"
        )
        output_dict["judge_score"] = judge_score
        output_dict["judge_followed_instructions"] = judge_followed_instructions
    else:
        # MCQ OB2
        if prediction.startswith(START_THINKING_TAG):
            think, response = split_think_and_response(prediction)
            print("think: ", think, flush=True)
            print("response: ", response, flush=True)
            prediction = response
        else:
            prediction, cot = post_process_llm_outputs(prediction, "answer")
            prediction = (  # Remove the junk output by the model - probably need to add more
                prediction.replace("Answer", "")
                .split("{")[0]
                .split(",")[0]
                .replace(" ", "")
                .replace("(", "")
                .replace(")", "")
                .replace(".", "")
                .replace(":", "")
                .replace("'", "")
                .replace('"', "")
                .strip()
            )
    return {**model_prediction_and_annotation, "prediction": prediction}


def evaluate_ob2(
    prediction: str,
    y: str,
    x: Dict[str, Any],
    meta: Dict[str, Any],
    output_dict: Optional[Dict[str, Any]] = None,
    options: Dict[str, int] = OPTIONS,
) -> Dict[str, float]:
    """
    See if the model correctly chose the correct multiple choice answer for OB2. Mark "correct" as True if so, False otherwise.

    Args:
        prediction: The letter, i.e A,B,C, or D which the model predicted as the correct answer for the MCQ.
        options: A map of MCQ choices as lettters mapping to a number. OB2 specifies the correct answer as 0,1,2,3 instead of A,B,C,D so we have to map back.

    Returns:
        A dict of the form: {"correct": True/False} indicating if the model answered correctly or not.
    """
    outputs = {}
    if output_dict is not None and "judge_score" in output_dict:
        followed_instructions = output_dict["judge_followed_instructions"]
        correct = LLM_JUDGE_SCORE_MAP[float(output_dict["judge_score"])]
        outputs.update(
            {"correct": correct, "followed_instructions": followed_instructions}
        )
    else:
        followed_instructions = True
        if prediction in options:
            answer_index = options[prediction]
            correct = answer_index == int(
                y
            )  # Target is specified as int not a choice ie A,B,C, or D
        else:
            logger.error(f"Unparsable Prediction: {prediction}")
            correct = False
            followed_instructions = False
        outputs.update(
            {"correct": correct, "followed_instructions": followed_instructions}
        )

    ## Fine-grained bucketing

    # Bucketing cue modality
    cue_modality = set()
    for cue in x["cues"]:
        for modality in cue["modality"]:
            if modality:
                outputs[f"correct_cueModality_{modality}"] = correct
                cue_modality.add(modality)
    if "vision" in cue_modality and len(cue_modality) == 1:
        outputs[f"correct_cueModality_visionOnly"] = correct
    if "vision" in cue_modality and "audio" in cue_modality and len(cue_modality) == 2:
        outputs[f"correct_cueModality_visionAudio"] = correct
    if (
        "vision" in cue_modality
        and "digital" in cue_modality
        and len(cue_modality) == 2
    ):
        outputs[f"correct_cueModality_visionDigital"] = correct
    if (
        "vision" in cue_modality
        and "audio" in cue_modality
        and "digital" in cue_modality
        and len(cue_modality) == 3
    ):
        outputs[f"correct_cueModality_visionAudioDigital"] = correct

    # Bucketing goal type
    if "structured_goal" in meta and "type" in meta["structured_goal"]:
        goal_type = meta["structured_goal"]["type"]
        if goal_type:
            outputs[f"correct_goalType_{goal_type}"] = correct

    # Bucketing context window length
    context_window_s = x["context_window"]["end_s"] - x["context_window"]["start_s"]
    for bucket, bucket_range in CONTEXT_WINDOW_LEN_BUCKET.items():
        if bucket_range[0] <= context_window_s < bucket_range[1]:
            outputs[f"correct_contextWindowLength_{bucket}"] = correct
            break

    return outputs


def get_task_config(
    dataset_dir: str,
    jsonl_dataset_path: str,
    transcription_dataset_path: Optional[str] = None,
    caption_directory: Optional[str] = None,
    digital_state_path: Optional[str] = None,
    longitudinal_history_path: Optional[str] = None,
    max_gen_len: int = 20,
    model_type: str = "cogvlm",
    digital_key_only: Optional[str] = "0",
    longitudinal_positive_only: Optional[str] = "0",
) -> VideoReasoningTaskConfig:
    """
    Builder for a VideoReasoningTaskConfig.

    Args:
        jsonl_dataset_path: Path to the JSONL dataset containing the data used for this eval. This file contains annotations, video paths, etc.
        max_gen_len: The maximum number of tokens to generate.

    Returns:
        A configuration for the OB2 video reasoning task.
    """
    load_data_fn = partial(
        load_ob2_data,
        jsonl_dataset_path=jsonl_dataset_path,
        transcription_dataset_path=transcription_dataset_path,
        caption_directory=caption_directory,
        dataset_dir=dataset_dir,
        digital_state_path=digital_state_path,
        longitudinal_history_path=longitudinal_history_path,
        digital_key_only=digital_key_only,
        longitudinal_positive_only=longitudinal_positive_only,
    )
    return VideoReasoningTaskConfig(
        load_data_fn=load_data_fn,
        max_gen_len=max_gen_len,
        postprocess_fn=post_process,
        metric_fns=[
            evaluate(
                evaluate_ob2,  # Takes model predictions as input
                inputs=(
                    "prediction",
                    "y",
                    "x",
                    "meta",
                    "output_dict",
                ),  # Apply evaluate_ob2 to values under these keys
                outputs=(
                    # "correct",
                ),  # Put outputs under these keys
            )
        ],
    )


# Register the task
MultimodalTaskRegistry.register(
    "maestro_ob2-maestro_ob2_qwen",
    get_task_config,
)
