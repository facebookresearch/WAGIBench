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

import atexit
import bisect

import contextlib
import getpass
import json
import logging
import os
import pathlib
import re
import shutil
import socket
import string
import subprocess
import sys
import tempfile
from collections import Counter, defaultdict
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import asdict, is_dataclass
from enum import Enum
from inspect import Parameter
from itertools import accumulate, islice, product
from logging import LogRecord
from pathlib import Path
from typing import (
    Any,
    Callable,
    DefaultDict,
    Dict,
    Iterable,
    Iterator,
    List,
    Literal,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    TypeVar,
    Union,
)

import numpy as np
import torch

import transformers

from evals.api import AverageMetric, Example, ExampleFn, Message, Role
from evals.utils.distributed import (
    get_dp_rank,
    get_global_rank,
    mp_rank_zero_info,
    mp_rank_zero_warn,
    rank_zero_debug,
)
from jinja2 import Environment, meta, Template
from numpy.random import RandomState
from packaging.version import parse
from tqdm import tqdm

T = TypeVar("T")


@contextmanager
def set_env_for_transformers_version():
    """
    Set the environment variable for transformers version 4.51.0.
    This is needed to fix the issue where the model is not properly loaded to all GPUs.

    Returns:
        None
    """
    world_size = None
    try:
        if is_tranformers_version_greater_than_4_51(transformers):
            if "WORLD_SIZE" in os.environ:
                world_size = os.environ["WORLD_SIZE"]
                del os.environ["WORLD_SIZE"]
        yield None

    finally:
        if world_size is not None:
            os.environ["WORLD_SIZE"] = world_size


def is_tranformers_version_greater_than_4_51(transformers_module):
    """
    Check if the version of transformers is greater than 4.51.
    This is useful as there is a bug in the version 4.51.0 that causes the model to be loaded
    to one gpu instead of all of them.

    Returns:
        bool: True if the version of transformers is greater than 4.51.0, False otherwise.
    """
    return parse(transformers_module.__version__) >= parse("4.51.0")


def load_jsonl(
    filename: str,
    num_shards: int = 1,
    shard_idx: int = 0,
    max_samples: Optional[int] = None,
) -> List[Example]:
    with open_file(filename, "r", encoding="utf-8") as file:
        iterator = islice(file, shard_idx, max_samples, num_shards)
        items: List[Example] = [json.loads(line) for line in iterator]
    return items


def batchify(iterable: Iterable[T], batch_size: int) -> Iterator[List[T]]:
    it = iter(iterable)
    while batch := list(islice(it, batch_size)):
        yield batch


def apply_functions(
    fns: Optional[List[Callable[[Example], Example]]], example: Example
) -> Example:
    if fns is None:
        return example
    for fn in fns:
        example.update(fn(example))
    return example


def evaluate(
    fn: Callable[..., Any],
    outputs: Union[str, Sequence[str]],
    inputs: Union[str, Sequence[str]] = ("prediction", "targets"),
    **kwargs: Any,
) -> Callable[..., Any]:
    def wrapper(x: Example) -> Example:
        outputs = fn(*(x[k] for k in input_keys), **kwargs)
        if isinstance(outputs, Sequence):
            return dict(zip(output_keys, outputs))
        elif isinstance(outputs, Dict):
            if len(output_keys) > 0:
                missed_outputs = [k for k in outputs.keys() if k not in output_keys]
                if len(missed_outputs) > 0:
                    mp_rank_zero_warn(
                        f"Missed outputs: {missed_outputs}.\n"
                        f"Expected: {output_keys}, got: {outputs.keys()}."
                    )
                return {k: v for k, v in outputs.items() if k in output_keys}
            else:
                return outputs
        else:
            return dict(zip(output_keys, (outputs,)))

    input_keys = (inputs,) if isinstance(inputs, str) else inputs
    output_keys = (outputs,) if isinstance(outputs, str) else outputs
    return wrapper


class ExampleSelector:
    def __init__(
        self,
        examples: Optional[List[Example]] = None,
        num_examples: int = 0,
        select_strategy: Literal["first", "index", "random"] = "first",
        select_indices: Optional[Sequence[int]] = None,
        preprocess_fn: Optional[ExampleFn] = None,
    ) -> None:
        self.examples: List[Example] = examples or []
        self.num_examples: int = num_examples or 0
        assert self.examples is not None and len(self.examples) >= self.num_examples
        self.select_strategy = select_strategy
        self.select_indices = select_indices
        self.preprocess_fn: Optional[ExampleFn] = preprocess_fn

    def __call__(
        self, random_state: Optional[np.random.RandomState] = None
    ) -> List[Example]:
        if self.num_examples == 0:
            return []
        if self.select_strategy == "first":
            examples = self.examples[: self.num_examples]
        elif self.select_strategy == "index":
            assert self.select_indices is not None
            examples = [self.examples[idx] for idx in self.select_indices]
            assert self.num_examples <= len(examples)
            examples = examples[: self.num_examples]
        elif self.select_strategy == "random":
            assert random_state is not None
            indices = random_state.choice(
                len(self.examples), self.num_examples, replace=False
            )
            examples = [self.examples[idx] for idx in indices]

        outputs = deepcopy(examples)
        for x in outputs:
            x.update(self.preprocess_fn(x) if self.preprocess_fn else {})
        return outputs


def jinja_format(template: str, skip_validation: bool = True, **kwargs: Any) -> str:
    if not skip_validation:
        variables = meta.find_undeclared_variables(Environment().parse(template))
        if not all(k in kwargs for k in variables):
            raise ValueError(
                f"Expected: {variables}, got: {sorted(kwargs)}.\n"
                f"Template:\n{template}"
            )
        kwargs = {k: kwargs[k] for k in variables}
    return Template(template).render(**kwargs)


def string_format(template: str, skip_validation: bool = True, **kwargs: Any) -> str:
    if not skip_validation:
        variables = [k[1] for k in string.Formatter().parse(template) if k[1]]
        if not all(k in kwargs for k in variables):
            raise ValueError(
                f"Expected: {variables}, got: {sorted(kwargs)}.\n"
                f"Template:\n{template}"
            )
        #  `Dict[Optional[str], typing.Any]`.
        kwargs = {k: kwargs[k] for k in variables}
    return template.format(**kwargs)


def truncate(tokens: List[int], max_length: int, side: str = "left") -> List[int]:
    assert side in ("left", "right")
    return tokens[:max_length] if side == "right" else tokens[-max_length:]


def unroll_chat(messages: List[Message]) -> List[List[Message]]:
    """Always start with system prompt if there is any"""
    # TODO: only works with one system prompt at the moment
    assert messages[0].role == Role.SYSTEM or not any(
        msg.role == Role.SYSTEM for msg in messages
    ), "'unroll_chat' only supports when there is one system prompt at the beginning"
    system_prompt = messages[0] if messages[0].role == Role.SYSTEM else None
    dialogs: List[List[Message]] = []
    reversed_messages = []
    for msg in reversed(messages):
        reversed_messages.append(msg)
        if msg.role == Role.USER:
            dialogs.append(reversed_messages[::-1])
            if system_prompt is not None:
                dialogs.append([system_prompt] + reversed_messages[::-1])
    return dialogs[::-1]


def unroll_msg(content: str) -> List[str]:
    """Returns all list of last words"""
    unroll_msgs: List[str] = [content]
    while True:
        split = content.split(maxsplit=1)
        if len(split) == 2:
            content = split[1]
            unroll_msgs.append(content)
        else:
            if split[0] != content:
                unroll_msgs.append(split[0])
            break
    return unroll_msgs


def truncate_chat(
    messages: List[Message],
    max_prompt_len: int,
    chat_tokenizer_fn: Callable[[List[Message]], List[int]],
) -> List[Message]:
    """
    Left-truncates a dialog so it is the longest sequence of Messages that starts
    with a user message and whose tokenization is below `max_prompt_len`
    TODO: support system role
    """
    assert messages[-1].role == Role.USER
    unrolled_dialogs = sorted(unroll_chat(messages), key=lambda d: -len(messages))
    init_dialog_token_len = 0
    for i, dialog in enumerate(unrolled_dialogs):
        dialog_token_len = len(chat_tokenizer_fn(dialog))
        if i == 0:
            init_dialog_token_len = dialog_token_len
        if dialog_token_len > max_prompt_len:
            continue
        if i > 0:
            mp_rank_zero_info(
                f"Keeping only the last {len(messages)} messages "
                f"and truncating the prompt length from {init_dialog_token_len} to "
                f"{dialog_token_len} < {max_prompt_len} tokens."
            )
        return dialog
    mp_rank_zero_info(
        "Keeping only the last user message "
        f"and truncating the prompt length from {dialog_token_len}"
        f" so it is smaller than to {max_prompt_len} tokens."
    )
    unrolled_last_msg = sorted(unroll_msg(messages[-1].content), key=lambda d: -len(d))
    for i, content in enumerate(unrolled_last_msg):
        dialog_token_len = len(
            chat_tokenizer_fn([Message(role=Role.USER, content=content)])
        )
        if i == 0:
            init_dialog_token_len = dialog_token_len
        if dialog_token_len > max_prompt_len:
            continue
        if i > 0:
            mp_rank_zero_info(
                f"Keeping only the last {len(content.split())} words "
                f"and truncating the prompt length from {init_dialog_token_len} to "
                f"{dialog_token_len} < {max_prompt_len} tokens."
            )
        return dialog
    raise ValueError("max_prompt_len is too small for chat")


def remove_articles(text: str) -> str:
    return re.sub(r"\b(a|an|the)\b", " ", text)


def white_space_fix(text: str) -> str:
    return " ".join(text.split())


def remove_punc(text: str) -> str:
    exclude = set(string.punctuation)
    return "".join(ch for ch in text if ch not in exclude)


def normalize_answer(text: str) -> str:
    return white_space_fix(remove_articles(remove_punc(text.lower())))


def first_answer(text: str, markers: Sequence[str] = ("Q:", "A:")) -> str:
    for marker in markers:
        text = text.split(marker)[0]
    return text


def first_letter(text: str) -> str:
    text = text.strip()
    return text[0] if text else ""


def filter_string_numericals(s: str) -> str:
    match = re.findall(r"[-0-9.,/]+", s)
    match = [x for x in match if re.search(r"\d", x)]
    if len(match) == 0:
        return ""
    return match[-1].strip(".").replace(",", "")


def find_majority(texts: List[str]) -> str:
    first_two_votes = Counter(texts).most_common(2)
    majority = first_two_votes[0][0]
    if majority == "" and len(first_two_votes) > 1:
        majority = first_two_votes[1][0]
    return majority


def filter_by_pattern(names: Iterable[str], pattern: Optional[str]) -> List[str]:
    outputs: List[str] = []
    if pattern is not None:
        for p in pattern.split(","):
            p = p.strip().replace(".", "\\.").replace("*", ".*")
            outputs.extend(filter(re.compile(f"^{p}$").match, names))
    return outputs


def get_regex_mcq_patterns() -> List[str]:
    """
    Returns a list of regex patterns to extract answers from MCQ questions.
    """
    patterns = [
        r"correct answer is .*?\(([A-Z])\)",
        r"correct answer is \((.*?)\)",
        r"(?<=correct answer is )[A-Z,\s]+",
        r"(?<=correct answer is \()[A-Z,\s]+(?=\))",
        r"(?<=correct answer is )[A-Z,\s,and]+",
        r"(?<=correct answer is )[A-Z]",
        r"correct answer is\s+(?:([A-Z])|\(([A-Z,]+)\))(?:,\s*\(([A-Z,]+)\))*",
        r"(?:correct answer is\s+)([A-Z](?:,\s*[A-Z])*)(?:\.|\s|$)|(?:\(([A-Z])\),?\s*)+",
        r"correct answer is\s+([A-Z]|(?:\((?:[A-Z],?)+\)))+\.?",
        r"r'\b[A-Z]\b'",
        r"option (\w)",
        r"answer is\s+(?:([A-Z])|\(([A-Z,]+)\))(?:,\s*\(([A-Z,]+)\))*",
        r"(?<=answer is )[\d\.\$\frac{}\/]+(?=\s|\.|$)",
        r"(?:answer is\s+)([A-Z](?:,\s*[A-Z])*)(?:\.|\s|$)|(?:\(([A-Z])\),?\s*)+",
        r"answer is\s+([A-Z]|(?:\((?:[A-Z],?)+\)))+\.?" r"r'\b[A-Z]\b'",
        r"answer is\s+(?:([A-Z])|\(([A-Z,]+)\))(?:,\s*\(([A-Z,]+)\))*",
        r"(?<=answer is )[\d\.\$\frac{}\/]+(?=\s|\.|$)",
        r"correct answer is ((?:\d,)*\d)",
        r"\b([A-Za-z])\)",
        r"\b[A-Z]",
    ]
    return patterns


def get_regex_generative_patterns() -> List[str]:
    patterns = [
        r"(?i)correct answer is\s*([^\.$]+(?:\.\d+)?=*\d*)(?=\.$)",
        r"(?<=correct answer is \$)[^\$]+(?=\$)",
        r"(?i)answer is\s*([^\.$]+(?:\.\d+)?=*\d*)(?=\.$)",
        r"(?<=answer is )[\d\.\$\frac{}\/]+(?=\s|\.|$)",
        r"(?<=is \$)[^\$]+(?=\$)",
        r"(?i)is\s*([^\.$]+(?:\.\d+)?=*\d*)(?=\.$)",
        r"(?<=\\boxed\{)[^\}]+(?=\})",
    ]
    return patterns


def check_pattern(pattern: str, s: str) -> bool:
    match = re.search(pattern, s)
    return bool(match)


def unroll_configs(
    defaults: Mapping[str, Parameter], params: Mapping[str, Any], prefix: str
) -> Dict[str, Dict[str, Any]]:
    def unroll(x: Mapping[str, Any]) -> Sequence[Dict[str, Any]]:
        x = [product([k], v) if isinstance(v, list) else [(k, v)] for k, v in x.items()]
        return [dict(item) for item in product(*x)]

    configs: Dict[str, Dict[str, Any]] = {}
    defaults = {k: v.default for k, v in defaults.items()}
    for kwargs in unroll(params):
        assert kwargs.keys() <= set(defaults.keys()), (kwargs, defaults)
        # Avoid using same name for different task variants
        overrides = {k: v for k, v in kwargs.items() if defaults[k] != v}
        suffix = ".".join(f"{k}_{v}" for k, v in overrides.items())
        name = f"{prefix}{'.' + suffix if suffix else ''}"
        configs[name] = {**defaults, **kwargs}
    return configs


def get_token_offsets(tokenizer: Any, text: str) -> Tuple[List[str], List[int]]:
    from sentencepiece import SentencePieceProcessor  # type: ignore

    if not isinstance(tokenizer, SentencePieceProcessor):
        from tiktoken import Encoding  # type: ignore

        assert isinstance(tokenizer, Encoding)
        token_bytes = tokenizer.decode_tokens_bytes(
            tokenizer.encode(text, allowed_special="all")
        )
        text_len, offsets = 0, []
        for token in token_bytes:
            offsets.append(max(0, text_len - (0x80 <= token[0] < 0xC0)))
            text_len += sum(1 for c in token if not 0x80 <= c < 0xC0)
        tokens = [text[s:e] for s, e in zip(offsets, offsets[1:] + [None])]
        return tokens, offsets
    elif hasattr(tokenizer, "encode_as_immutable_proto"):
        pieces = tokenizer.encode_as_immutable_proto(text).pieces
        tokens = [p.surface for p in pieces]
        offsets = [p.begin for p in pieces]
    else:
        from sentencepiece import sentencepiece_pb2  # type: ignore

        spt = sentencepiece_pb2.SentencePieceText()
        spt.ParseFromString(tokenizer.encode_as_serialized_proto(text))
        tokens = [p.surface for p in spt.pieces]
        offsets = list(accumulate((len(t) for t in tokens), initial=0))[:-1]
    return tokens, offsets


def text_index(
    full_text: str, offsets: List[int], text: str, align: str = "right"
) -> slice:
    assert align in ("left", "right")
    start_index = full_text.rfind(text)
    if start_index == -1:
        mp_rank_zero_warn(f"Text '{text}' not found in '{full_text}'")
        return slice(0, 1)
    end_index = start_index + len(text)
    text_start = bisect.bisect_right(offsets, start_index) - 1
    text_end = bisect.bisect_right(offsets, end_index)
    if align == "left":
        return slice(text_start, text_end)
    return slice(text_start - len(offsets) or None, text_end - len(offsets) or None)


def get_torch_dtype(dtype: str) -> torch.dtype:
    return {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[dtype]


class TqdmLoggingHandler(logging.StreamHandler):
    def emit(self, record: LogRecord) -> None:
        """Avoid tqdm progress bar interruption by logger's output to console"""
        try:
            msg = self.format(record)
            tqdm.write(msg)
            self.flush()
        except RecursionError:
            raise
        except Exception:
            self.handleError(record)


def initialize_logger() -> None:
    logger = logging.getLogger()
    logger.setLevel(os.environ.get("LOGLEVEL", "INFO"))
    formatter = logging.Formatter(
        f"[%(asctime)s] [rank {get_global_rank()}] [%(levelname)s] %(message)s"
    )
    # stdout: everything
    stdout_handler = TqdmLoggingHandler(sys.stdout)
    stdout_handler.setLevel(os.environ.get("LOGLEVEL", "INFO"))
    stdout_handler.setFormatter(formatter)
    # stderr: warnings / errors and above
    stderr_handler = logging.StreamHandler(sys.stderr)
    stderr_handler.setLevel(logging.WARNING)
    stderr_handler.setFormatter(formatter)

    if logger.hasHandlers():
        logger.handlers.clear()
    logger.handlers.append(stdout_handler)
    logger.handlers.append(stderr_handler)


def setup_env() -> None:
    triton_cache_dir = tempfile.mkdtemp(prefix="/dev/shm/")
    tiktoken_cache_dir = tempfile.mkdtemp(prefix="/dev/shm/")
    atexit.register(shutil.rmtree, triton_cache_dir, ignore_errors=True)
    atexit.register(shutil.rmtree, tiktoken_cache_dir, ignore_errors=True)
    env_vars = {
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "TORCH_NCCL_ASYNC_ERROR_HANDLING": "1",
        "NCCL_IB_TIMEOUT": "22",
        "TRITON_CACHE_DIR": triton_cache_dir,
        "TIKTOKEN_CACHE_DIR": tiktoken_cache_dir,
    }
    for name, value in env_vars.items():
        if os.environ.get(name) != value:
            os.environ[name] = value
            rank_zero_debug(f"WARNING: Setting {name} to {value}")


def get_job_info() -> Dict[str, Optional[str]]:
    info: Dict[str, Optional[str]] = {
        "hostname": socket.gethostname(),
        "job_id": os.getenv("MAST_HPC_JOB_NAME", os.getenv("SLURM_JOB_ID")),
        "job_hosts": os.getenv(
            "MAST_HPC_TASK_GROUP_HOSTNAMES", os.getenv("SLURM_NODELIST")
        ),
    }
    if os.getenv("MAST_HPC_JOB_NAME"):
        info["mast_url"] = f"https://www.internalfb.com/mast/job/{info['job_id']}"
        info["packages"] = os.getenv("MAST_APPLICATION_PACKAGES")
        info["dump_dir"] = os.getenv("DUMP_DIR")
    return info


def get_git_info() -> Dict[str, Any]:
    repo: str = str(pathlib.Path(__file__).resolve().parent.parent.parent)

    def get_cmd_result(cmd: str, default: str) -> str:
        with contextlib.suppress(Exception):
            result = subprocess.check_output(
                cmd.split(), cwd=repo, stderr=subprocess.DEVNULL
            )
            return result.decode().strip()
        return default

    rev = get_cmd_result("git rev-parse HEAD", "unknown")
    branch = get_cmd_result("git rev-parse --abbrev-ref HEAD", "unknown")
    return {
        "git_repo": repo,
        "commit": rev,
        "branch": branch,
        "user": getpass.getuser(),
    }


def get_version() -> str:
    info = get_git_info()
    if info["commit"] != "unknown":
        with open(Path(info["git_repo"]) / "version.txt", "r") as f:
            version = f.readline().strip()
        version += f"+git{info['commit'][:7]}"
        return version
    try:
        from evals.version import __version__
    except ModuleNotFoundError:
        __version__ = "unknown"
    return __version__


def get_gpu_info() -> str:
    mem_stats = torch.cuda.memory_stats()
    usage = {
        "active": mem_stats["active_bytes.all.current"] / 1024**3,
        "allocated": mem_stats["allocated_bytes.all.current"] / 1024**3,
        "reserved": mem_stats["reserved_bytes.all.current"] / 1024**3,
    }
    return ", ".join(f"{k}: {v:.2f}GB" for k, v in usage.items())


def get_random_state(
    seed: int,
    include_data_parallel: bool = True,
    include_job_array: bool = True,
) -> RandomState:
    """
    Construct a random state using a base seed, and optionally, data parallel
    rank and job array task ID.

    Args:
        seed (int): Primary seed value.
        data_parallel_seed (bool): If set, random states are different across DP groups.
        job_array_seed (bool): If set, random states are different across job arrays.
    """
    arr: List[int] = [seed]
    if include_data_parallel:
        arr.append(get_dp_rank())
    if include_job_array:
        arr.append(int(os.environ.get("SLURM_ARRAY_TASK_ID", 0)))
    return RandomState(tuple(arr))


def aggregate_results(
    task_results: Dict[str, Dict[str, AverageMetric]]
) -> Dict[str, Dict[str, AverageMetric]]:
    results: DefaultDict[str, AverageMetric] = defaultdict(
        lambda: AverageMetric(0, 0, 0)
    )
    for metrics in task_results.values():
        for key, metric in metrics.items():
            results[key].update(metric.value, metric.count, metric.square)
    return {"average": dict(results)}


def aggregate_fn_macro_avg(
    task_results: Dict[str, Dict[str, AverageMetric]],
    domains: Dict[str, str],
) -> Dict[str, Dict[str, AverageMetric]]:
    results: DefaultDict[str, DefaultDict[str, AverageMetric]] = defaultdict(
        lambda: defaultdict(lambda: AverageMetric(0, 0, 0))
    )
    for task_name, task_metrics in task_results.items():
        category = next(v for k, v in domains.items() if k in task_name)
        category = category.replace(" ", "_").lower()
        for key, metric in task_metrics.items():
            # average scores over subtasks
            results["macro_avg"][key].update(metric.value, 1)
            # average scores weighted by subtask sizes
            for group in (category, "micro_avg"):
                results[group][key].update(metric.value, metric.count, metric.square)

    return {k: dict(v) for k, v in results.items()}


def flatten_dict(data: Dict[str, Any], prefix: str = "") -> Dict[str, Any]:
    flat_dict = {}
    for key, value in data.items():
        if isinstance(value, dict):
            flat_dict.update(flatten_dict(value, f"{prefix}{key}/"))
        else:
            flat_dict[f"{prefix}{key}"] = value
    return flat_dict


def get_group_by_metrics(
    group_by: List[str], metrics: List[str], results: List[Dict[str, Any]]
) -> Dict[str, AverageMetric]:
    """
    Computes average group by scores to be stored as metrics. To be called via global_metric_fns.
    Metric can be found in results.json, keyed by `group_by_{group_by_key}_metric_{metric_name}_avg`

    :param results: list of all example results
    :param group_by: list of keys present in each example to group by
    :param metrics: list of metrics present in each example, computed in a previous step via metric_fns, to group by
    """
    agg_results: Dict[str, AverageMetric] = defaultdict(lambda: AverageMetric(0, 0, 0))
    for result in results:
        group_by_str = "_".join([f"{group}_{result[group]}" for group in group_by])
        for metric in metrics:
            group_by_key = f"metric_{metric}_group_by_{group_by_str}"
            agg_results[group_by_key].count += 1
            agg_results[group_by_key].avg += result["metrics"][metric]

    for key in agg_results:
        agg_results[key].avg /= agg_results[key].count
    return dict(agg_results)


def get_checkpoint_step(
    path: Optional[str] = None,
    pattern: str = r"checkpoint_(?P<step>\d+)",
) -> int:
    if path is None:
        return -1
    match = re.search(pattern, Path(path).name)
    return int(match.group("step")) if match else -1


def format_dict(
    data: Union[Dict[str, float], Dict[str, AverageMetric]],
    decimal: int = 6,
    delimiter: str = " | ",
) -> str:
    return delimiter.join(
        f"{k.lower().replace(' ', '_')}: {v.value if isinstance(v, AverageMetric) else v:.{decimal}f}"
        for k, v in data.items()
    )


def _default_json_encoder(o: Any) -> Any:
    if is_dataclass(o):
        return asdict(o)
    if isinstance(o, Enum):
        return o.value
    if callable(o):
        return repr(o)
    if isinstance(o, np.int64):
        return int(o)
    return json.JSONEncoder().default(o)


def write_to_json(
    obj: object,
    path: str,
    mode: str = "w",
    ending: Optional[str] = None,
    **kwargs: Any,
) -> None:
    with open_file(path, mode=mode) as fp:
        json.dump(obj, fp, default=_default_json_encoder, **kwargs)
        if ending is not None:
            fp.write(ending)


def write_to_jsonl(
    items: List[Dict[str, Any]],
    path: str,
    mode: str = "w",
    **kwargs: Any,
) -> None:
    with open_file(path, mode=mode) as fp:
        for item in items:
            fp.write(json.dumps(item, default=_default_json_encoder, **kwargs) + "\n")


@contextmanager
def open_file(path: str, mode: str = "r", **kwargs: Any) -> Any:
    if mode == "w" or mode == "a":
        os.makedirs(os.path.dirname(path), exist_ok=True)
    file = open(path, mode=mode, **kwargs)

    try:
        yield file
    finally:
        file.close()


def get_local_path(path: str, recursive: bool = True) -> Path:
    try:
        from iopath.common.file_io import g_pathmgr

        return Path(g_pathmgr.get_local_path(path, recursive=recursive))
    except (ModuleNotFoundError, ValueError):
        return Path(path)


def write_to_tensorboard(
    payload: Mapping[str, Union[str, int, float, bool]],
    log_dir: str,
    step: int,
    max_queue: int = 1000,
    tag: Optional[str] = None,
    **kwargs: Any,
) -> None:
    from torch.utils.tensorboard import SummaryWriter

    writer = SummaryWriter(log_dir, max_queue=max_queue, **kwargs)
    for key, value in payload.items():
        key = key if tag is None else f"{tag}/{key}"
        writer.add_scalar(key, value, global_step=step)
    writer.close()
