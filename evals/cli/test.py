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

from autogen import AssistantAgent, GroupChat, GroupChatManager, UserProxyAgent
from evals.predictors.multimodal.llama3_agent import Llama3Agent
from evals.predictors.multimodal.llama3v_agent import Llama3VAgent

if __name__ == "__main__":
    config = [
        {
            "model": "/checkpoint/maestro/models/Meta-Llama-3.1-8B-Instruct-hf",
            "model_client_cls": "Llama3Client",
        },
        {
            "model": "/checkpoint/maestro/models/Meta-Llama-3.2-11B-Vision-Instruct",
            "model_client_cls": "Llama3VClient",
        },
    ]

    bob = Llama3Agent(
        "bob",
        system_message="A chatbot named bob",
        llm_config={"config_list": config},
    )

    user_proxy = UserProxyAgent(
        "user_proxy",
        code_execution_config=False,  # no code execution
    )

    user_proxy.initiate_chat(bob, message="hi, how's the weather today?")
