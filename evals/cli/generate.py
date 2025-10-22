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

import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Union

import torch

from evals.api import Message, PredictorConfig, Prompt, Role
from evals.params import from_cli, parse_args, to_cli
from evals.predictors import build_predictor, PredictorRegistry
from evals.utils import (
    gather_object,
    get_dp_group,
    get_dp_rank,
    get_dp_size,
    get_global_rank,
    initialize_logger,
    load_jsonl,
    rank_zero_info,
    setup_env,
    write_to_jsonl,
)


@dataclass
class GenConfig:
    input_path: str
    output_path: str
    seed: int = 42
    max_samples: Optional[int] = None
    show_progress: bool = True

    predictor: str = "llm_inference"
    predictor_config: Optional[PredictorConfig] = None
    max_prompt_len: int = 4096
    max_gen_len: int = 1024
    temperature: float = 0.0
    top_k: int = 0
    top_p: float = 1.0

    keys: str = "dialog_history,messages"

    def __post_init__(self) -> None:
        assert self.predictor in PredictorRegistry.names(), (
            f"Predictor {self.predictor} not available in {PredictorRegistry.names()}",
        )
        assert self.temperature >= 0.0, self.temperature
        if self.temperature == 0:
            self.top_p = 0
            self.top_k = 0

    @property
    def prompt_keys(self) -> List[Union[str, int]]:
        return [int(k) if k.isdigit() else k for k in self.keys.split(",")]


def main(cfg: GenConfig) -> None:
    torch.manual_seed(cfg.seed)

    assert cfg.predictor_config is not None
    if cfg.predictor == "llm_inference":
        cfg.predictor_config.max_total_len = 4096  # type: ignore
    predictor = build_predictor(cfg.predictor_config)
    gen_model: str = getattr(
        predictor,
        "model_name",
        (
            Path(cfg.predictor_config.checkpoint_dir).name
            if hasattr(cfg.predictor_config, "checkpoint_dir")
            else cfg.predictor
        ),
    )

    start = time.monotonic()
    rank_zero_info(f"Run generations using inputs from {cfg.input_path}")
    dataset = load_jsonl(
        cfg.input_path, get_dp_size(), get_dp_rank(), max_samples=cfg.max_samples
    )
    prompts: List[Prompt] = []
    for x in dataset:
        if "generations" not in x:
            x["generations"] = []
        msgs = x
        for k in cfg.prompt_keys:
            msgs = msgs[k]  # type: ignore
        prompt = [
            Message(
                role=Role.USER if msg["role"] == "user" else Role.ASSISTANT,  # type: ignore
                content=msg["content"],  # type: ignore
            )
            for msg in msgs
        ]
        prompts.append(prompt)
    predictions = predictor(
        prompts=prompts,
        max_prompt_len=cfg.max_prompt_len,
        max_gen_len=cfg.max_gen_len,
        temperature=cfg.temperature,
        top_p=cfg.top_p,
        top_k=cfg.top_k,
        echo=False,
        return_logprobs=False,
        show_progress=cfg.show_progress,
    )
    for x, prediction in zip(dataset, predictions):
        x["generations"].append(
            {
                "gen_model": gen_model,
                "gen_params": {
                    "max_prompt_len": cfg.max_prompt_len,
                    "max_gen_len": cfg.max_gen_len,
                    "temperature": cfg.temperature,
                    "top_k": cfg.top_k,
                    "top_p": cfg.top_p,
                },
                "content": prediction.text,
            }
        )
    gathered_results = gather_object(dataset, group=get_dp_group())
    results = [x for res in gathered_results for x in res]
    if get_global_rank() == 0:
        rank_zero_info(f"Writing generations to {cfg.output_path}")
        write_to_jsonl(results, cfg.output_path)
        rank_zero_info(f"Took {time.monotonic() - start:.2f} seconds")


def cfg_from_cli() -> GenConfig:
    known, unknown = to_cli(GenConfig).parse_known_args()  # type: ignore
    cfg: GenConfig = from_cli(GenConfig, vars(known), allow_incomplete=True)
    cfg.predictor_config = parse_args(
        PredictorRegistry.get_config_cls(cfg.predictor), unknown
    )
    return cfg


if __name__ == "__main__":
    cfg: GenConfig = cfg_from_cli()
    initialize_logger()
    setup_env()
    main(cfg)
