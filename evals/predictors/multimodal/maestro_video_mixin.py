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

import io
import json
import math
import re

from typing import Optional, Tuple, Union

import numpy as np
from decord import bridge, cpu, VideoReader
from iopath.common.file_io import g_pathmgr


class MaestroVideoMixin:
    """
    Provides a method to load a video from a path.
    """

    mode: str = "default"  # default, chunked
    chunk_secs: float = 60.0  # 1-min chunk
    # chunk_secs: float = 10.0  # 10 seconds chunk

    def load_video(
        self,
        video_path: str,
        clip_start_sec: Optional[float] = None,
        clip_end_sec: Optional[float] = None,
    ) -> Tuple[np.ndarray, np.ndarray, Union[int, float]]:
        """
        Return a numpy array of a video.
        Sample number_of_samples frames from the video between clip_start_sec and clip_end_sec to
        create a batch to be fed into a model.

        Args:
            video_path: The location of the video.
            clip_start_sec: The start time of the clip.
            clip_end_sec: The ending time of the clip.
            number_of_splits: The number of partitions to split the frames of the video into.

        Returns:
            A numpy array of the clip in BTHWC order, the id of the frames in the clip, and the FPS of the video.
        """
        bridge.set_bridge("torch")
        with g_pathmgr.open(video_path, "rb") as f:
            decord_vr = VideoReader(f, ctx=cpu(0))
        frame_id_list = None
        total_frames = len(decord_vr)
        video_fps = decord_vr.get_avg_fps()
        if clip_start_sec is None or clip_start_sec < 0.0:
            clip_start_sec = 0.0
        if clip_end_sec is None or clip_end_sec < 0.0:
            clip_end_sec = total_frames / video_fps

        start_frame = int(clip_start_sec * video_fps)
        end_frame = (
            min(total_frames, int(clip_end_sec * decord_vr.get_avg_fps()))
            if clip_end_sec is not None
            else total_frames
        )

        if hasattr(self, "sample_fps") and self.sample_fps is not None:
            # frame_id_list = [
            #     i for i in range(0, len(decord_vr), round(video_fps / self.sample_fps))
            # ]
            frame_id_list = [
                i
                for i in range(
                    start_frame, end_frame, round(video_fps / self.sample_fps)
                )
            ]
        else:
            frame_id_list = np.linspace(
                start_frame, end_frame - 1, self.number_of_samples, dtype=int
            )

        video_data = decord_vr.get_batch(frame_id_list)  # BHWC
        return video_data, frame_id_list, video_fps


### Utils
def get_world_state_history(duration, captions, chunk_secs, min_duration=30.0):
    """
    This function will return a list of dictionaries where each dictionary will contain the world state for 1-minute of the video arranged chronologically.
    It will also include the timestamp in the format HH:MM:SS - HH:MM:SS.
    """
    # Create world state history as a list of dictionaries.
    world_state_history = []

    for index, caption in enumerate(captions):
        caption = read_and_combine_json_outputs(caption)  # List with a single dict
        if len(caption) > 0:
            start_time = index * chunk_secs
            end_time = min((index + 1) * chunk_secs, duration)
            if end_time - start_time < min_duration:
                continue
            start_time = convert_seconds_to_hms(start_time)
            end_time = convert_seconds_to_hms(end_time)
            caption[0]["time stamp"] = f"{start_time} - {end_time}"
            world_state_history.append(caption[0])

    return world_state_history


def read_and_combine_json_outputs(file_content):
    """
    LLM outputs from GPT-type models have many issues even when returned in JSON format.
    Some examples include using <'> instead of <"> for dictionary keys, missing <,> in dictionary lists etc.
    This function is written to handle most of these errors and return a list of dictionary output.
    A temporary file created for debugging purposes.
    """
    combined_list = []
    current_block = ""

    # Use this for frequency comparison questions.
    file_content = file_content.replace("json", "")
    file_content = file_content.replace("`", "")
    cleaned_content = re.sub(r"},\s*", "}, ", file_content)
    cleaned_content = f"[{cleaned_content}]"

    # Write the cleaned content back to a file
    buffer = io.StringIO()
    buffer.write(cleaned_content)
    buffer.seek(0)
    lines = buffer.readlines()
    buffer.close()

    # Go through each line
    for line in lines:
        line = re.sub(r"}\s+{", "},\n{", line)
        stripped_line = line.strip()
        if stripped_line:  # Add non-empty lines to the current block
            current_block += line
        elif current_block:  # Process the current block when an empty line is found
            try:
                processed_block = preprocess_json_block(current_block)
                json_array = json.loads(processed_block)
                combined_list.extend(json_array)
                current_block = ""  # Reset the block
            except json.JSONDecodeError as e:
                print(f"{e}: content: {file_content}")
                current_block = ""  # Reset the block on error

    # Process any remaining block after the loop
    if current_block:
        try:
            processed_block = preprocess_json_block(current_block)
            json_array = json.loads(processed_block)
            combined_list.extend(json_array)
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON: {e}, content: {file_content}")

    return combined_list


def convert_seconds_to_hms(seconds):
    """
    Converts a time in seconds (including fractional seconds) to a format of hours:minutes:seconds.
    Each part will be a maximum of two integers, and fractional seconds will be rounded down.
    """
    # Convert seconds to an integer to remove fractional parts
    # Use math.floor to round down the seconds
    total_seconds = math.floor(seconds)

    # Calculate hours, minutes, and seconds
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    remaining_seconds = total_seconds % 60

    return f"{hours:02}:{minutes:02}:{remaining_seconds:02}"


def preprocess_json_block(block):
    # This function preprocesses a JSON block to ensure it's in a valid format for json.loads()
    # It focuses on escaping newline characters within the strings, not outside them.
    # This simplistic approach may need refinement based on your JSON structure.
    block = block.strip("`")
    block = block.replace("json", "")
    block = block.replace("\n", "")
    return block


def convert_to_free_form_text_representation(world_state_history):
    """
    This function will form the textual representation for the entire video to be used for QA.
    It gives a good structured representation of the entire video.
    JSON types of representations are good for outputs, but free-form/ semi-structured should be better for input.
    """
    free_form_text_representation = ""
    for i in world_state_history:
        x = ""
        start_time, end_time = (
            i["time stamp"].split("-")[0].strip(),
            i["time stamp"].split("-")[1].strip(),
        )
        x += f"**Timestamp**: {start_time} - {end_time}\n"

        for key in i:
            if key != "time stamp":
                x += f"**{key}**: {i[key]}\n"
        free_form_text_representation += f"{x}\n"

    return free_form_text_representation
