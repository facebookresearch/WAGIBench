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
from typing import List, Optional

import torch

from evals.api import PredictorConfig
from evals.params import from_cli, parse_args, to_cli
from evals.predictors import build_predictor, PredictorRegistry
from evals.utils import (
    broadcast_object_list,
    get_dp_size,
    get_mp_group,
    get_mp_rank,
    initialize_logger,
    setup_env,
)
from llm_common.datatypes import Message, SampleSFT


@dataclass
class InteractiveConfig:
    seed: int = 42
    predictor: str = "llm_inference"
    predictor_config: Optional[PredictorConfig] = None
    max_prompt_len: int = 4096
    max_gen_len: int = 1024
    temperature: float = 0.0
    top_k: int = 0
    top_p: float = 1.0

    def __post_init__(self) -> None:
        assert self.predictor in PredictorRegistry.names(), (
            f"Predictor {self.predictor} not available in {PredictorRegistry.names()}",
        )
        assert self.temperature >= 0.0, self.temperature
        if self.temperature == 0:
            self.top_p = 0
            self.top_k = 0


def main(cfg: InteractiveConfig) -> None:
    torch.manual_seed(cfg.seed)

    assert cfg.predictor_config is not None
    if cfg.predictor == "llm_inference":
        cfg.predictor_config.max_total_len = 4096  # type: ignore
    predictor = build_predictor(cfg.predictor_config)
    assert get_dp_size() == 1, "data parallelism not supported for interactive mode"

    messages = []
    if get_mp_rank() == 0:
        system_prompt = input(format_prompt("SYSTEM: ")).strip()
        if system_prompt:
            messages.append(Message(source="system", body=system_prompt))

    while True:
        if get_mp_rank() == 0:
            user_input = input(format_prompt("\nUSER: ")).strip()
            messages.append(Message(source="user", body=user_input))
        object_list: List[List[Message]] = [messages]
        broadcast_object_list(object_list, src=0, group=get_mp_group())
        predictions = predictor(
            prompts=[SampleSFT(object_list[0])],
            max_prompt_len=cfg.max_prompt_len,
            max_gen_len=cfg.max_gen_len,
            temperature=cfg.temperature,
            top_p=cfg.top_p,
            top_k=cfg.top_k,
            echo=False,
            return_logprobs=False,
            show_progress=False,
        )
        if get_mp_rank() == 0:
            assistant_msg = predictions[0].text.strip()
            print(format_prompt("\nASSISTANT: ") + assistant_msg)
            messages.append(Message(source="assistant", body=assistant_msg))


def cfg_from_cli() -> InteractiveConfig:
    known, unknown = to_cli(InteractiveConfig).parse_known_args()  # type: ignore
    cfg: InteractiveConfig = from_cli(
        InteractiveConfig, vars(known), allow_incomplete=True
    )
    cfg.predictor_config = parse_args(
        PredictorRegistry.get_config_cls(cfg.predictor), unknown
    )
    return cfg


def format_prompt(prompt: str) -> str:
    return f"\033[32m{prompt}\033[0;1m"


if __name__ == "__main__":
    cfg: InteractiveConfig = cfg_from_cli()
    initialize_logger()
    setup_env()
    main(cfg)
