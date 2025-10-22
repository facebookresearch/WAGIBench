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

from typing import Any, Dict, List

from evals.utils import jinja_format

try:
    from llm_common.datatypes import Message, MessageV2, SampleSFT
    from llm_common.tokenizers.datatypes import Roles
except ImportError:
    Message = MessageV2 = SampleSFT = Roles = None


def jinja_dialog_format(
    template: Dict[str, str],
    skip_validation: bool = True,
    append_gen_prefix: bool = False,
    **kwargs: Any,
) -> SampleSFT:
    messages: List[Message] = []
    # System
    if "system_prompt" in template:
        system_content = jinja_format(
            template=template["system_prompt"],
            skip_validation=skip_validation,
            **kwargs,
        )
        messages.append(Message.system(system_content))
    # Few shots examples
    for example in kwargs["few_shot"]:
        prompt_content = jinja_format(
            template=template["prompt"], skip_validation=skip_validation, **example
        )
        answer_content = jinja_format(
            template=template["answer"], skip_validation=skip_validation, **example
        )
        messages.append(Message.user(body=prompt_content))
        messages.append(Message.assistant(body=answer_content))
        messages.append(Message.assistant_eot())
    # Question
    prompt_content = jinja_format(
        template=template["prompt"], skip_validation=skip_validation, **kwargs
    )
    messages.append(Message.user(body=prompt_content))
    # Generation prefix
    if append_gen_prefix:
        assert "gen_prefix" in template, "gen_prefix NOT defined in template"
        gen_prefix = jinja_format(
            template=template["gen_prefix"], skip_validation=skip_validation, **kwargs
        )
        messages.append(Message.assistant(body=gen_prefix))
    return SampleSFT(dialog=messages)


def get_dialog(kwargs: Any) -> List[Message]:
    """
    Messages should be compatible with finetune_v3 tokenizers
    """
    messages: List[Message] = []
    for msg in kwargs["dialog"]:
        assert msg["source"] in [
            Roles.system,
            Roles.user,
            Roles.assistant,
            Roles.ipython,
        ], "Unknown source"
        if msg["source"] == Roles.system:
            messages.append(Message.system(body=msg["body"]))
        elif msg["source"] == Roles.user:
            messages.append(Message.user(body=msg["body"]))
        elif msg["source"] == Roles.assistant:
            messages.append(
                Message(
                    source=msg["source"],
                    destination=msg["destination"],
                    body=msg["body"],
                    eot=msg["eot"],
                )
            )
        elif msg["source"] == Roles.ipython:
            messages.append(Message.ipython_return(body=msg["body"]))
    return messages


def get_dialog_v2(kwargs: Any) -> List[MessageV2]:
    """
    MessageV2 should be compatible with TiktokenFinetuneV5 tokenizers
    """
    messages: List[MessageV2] = []
    for i, msg in enumerate(kwargs["dialog"]):
        assert msg["source"] in [
            Roles.system,
            Roles.user,
            Roles.assistant,
            Roles.ipython,
        ], "Unknown source"
        if msg["source"] == Roles.system:
            messages.append(MessageV2.system(body=msg["body"].strip()))
        elif msg["source"] == Roles.user:
            messages.append(MessageV2.user(body=msg["body"].strip()))
        elif msg["source"] == Roles.assistant:
            if msg["body"] is None and msg["eot"]:
                continue
            eot = msg["eot"]
            if i < len(kwargs["dialog"]) - 1 and kwargs["dialog"][i + 1]["eot"]:
                eot = True
            messages.append(
                MessageV2(
                    source=msg["source"],
                    body=msg["body"].strip(),
                    eot=eot,
                )
            )
        elif msg["source"] == Roles.ipython:
            messages.append(MessageV2.ipython_return(body=msg["body"].strip()))
    return messages


def remove_dialog_headers(s: str) -> str:
    return (
        s.split("<|start_header_id|>assistant<|end_header_id|>\n\n")[-1]
        .split("<|eot_id|>")[0]
        .strip()
    )
