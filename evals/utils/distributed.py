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

import importlib
import logging
import os
import random
import subprocess
from collections import defaultdict
from datetime import timedelta
from logging import Logger
from types import ModuleType
from typing import Any, Dict, List, Optional, Union

import torch
import torch.distributed as dist
from evals.api import AverageMetric
from fairscale.nn.model_parallel import destroy_model_parallel

_LOGGER: Logger = logging.getLogger()


def get_mpu_module() -> ModuleType:
    default_module_name = "fairscale.nn.model_parallel.initialize"
    module_name = os.environ.get("MPU_MODULE", default_module_name)
    try:
        return importlib.import_module(module_name)
    except Exception as e:
        rank_zero_warn(f"Error when importing {module_name}: {e}")
        return importlib.import_module(default_module_name)


def init_torch_distributed(backend: str = "cpu:gloo,cuda:nccl") -> None:
    if dist.is_initialized():
        return
    os.environ["RANK"] = str(get_global_rank())
    os.environ["WORLD_SIZE"] = str(get_world_size())
    os.environ["MASTER_ADDR"] = get_master_addr()
    os.environ["MASTER_PORT"] = str(
        get_master_port(job_id=int(os.environ.get("SLURM_JOB_ID", -1)))
    )
    local_rank = get_local_rank()
    if "nccl" in backend:
        torch.cuda.set_device(local_rank)
    dist.init_process_group(backend=backend, timeout=timedelta(hours=10))


def init_model_parallel(model_parallel_size: int) -> None:
    mpu = get_mpu_module()
    if not dist.is_initialized():
        init_torch_distributed()

    if not mpu.model_parallel_is_initialized():
        mpu.initialize_model_parallel(model_parallel_size)


def reinit_model_parallel() -> None:
    # Importing llm_inference here as it might not work with XLFormers environment
    from llm_inference.models import nccl

    destroy_model_parallel()
    nccl.get_mp_group.cache_clear()
    nccl.get_mp_src_rank.cache_clear()
    nccl.get_mp_rank.cache_clear()
    nccl.get_mp_world_size.cache_clear()


def is_torch_run() -> bool:
    return os.environ.get("TORCHELASTIC_RUN_ID") is not None


def is_slurm_job() -> bool:
    return "SLURM_JOB_ID" in os.environ


def get_global_rank() -> int:
    if dist.is_initialized():
        return dist.get_rank()
    if is_torch_run():
        return int(os.environ["RANK"])
    if is_slurm_job():
        return int(os.environ["SLURM_PROCID"])
    return 0


def get_local_rank() -> int:
    if is_torch_run():
        return int(os.environ["LOCAL_RANK"])
    if is_slurm_job():
        return int(os.environ["SLURM_LOCALID"])
    return 0


def get_world_size() -> int:
    if dist.is_initialized():
        return dist.get_world_size()
    if is_torch_run():
        return int(os.environ["WORLD_SIZE"])
    if is_slurm_job():
        return int(os.environ["SLURM_NTASKS"])
    return 1


def get_master_addr() -> str:
    if is_torch_run():
        return os.environ["MASTER_ADDR"]
    if is_slurm_job():
        hostnames = subprocess.check_output(
            ["scontrol", "show", "hostnames", os.environ["SLURM_JOB_NODELIST"]]
        )
        return hostnames.split()[0].decode("utf-8")
    return "127.0.0.1"


def get_master_port(job_id: int) -> Optional[int]:
    if is_torch_run():
        return int(os.environ["MASTER_PORT"])
    if is_slurm_job():
        MIN_MASTER_PORT, MAX_MASTER_PORT = (20000, 60000)
        rng = random.Random(job_id)
        return rng.randint(MIN_MASTER_PORT, MAX_MASTER_PORT)
    return None


def get_rank() -> int:
    if dist.is_initialized():
        return dist.get_rank()
    return 0


def get_dp_rank() -> int:
    mpu = get_mpu_module()
    if mpu.model_parallel_is_initialized():
        return mpu.get_data_parallel_rank()
    return get_global_rank()


def get_dp_size() -> int:
    mpu = get_mpu_module()
    if mpu.model_parallel_is_initialized():
        return mpu.get_data_parallel_world_size()
    return get_world_size()


def get_mp_rank() -> int:
    mpu = get_mpu_module()
    if mpu.model_parallel_is_initialized():
        return mpu.get_model_parallel_rank()
    return 0


def get_mp_size() -> int:
    mpu = get_mpu_module()
    if mpu.model_parallel_is_initialized():
        return mpu.get_model_parallel_world_size()
    return 1


def get_dp_group() -> Optional[dist.ProcessGroup]:
    mpu = get_mpu_module()
    if mpu.model_parallel_is_initialized():
        return mpu.get_data_parallel_group()
    if dist.is_initialized():
        return dist.group.WORLD
    return None


def get_mp_group() -> Optional[dist.ProcessGroup]:
    mpu = get_mpu_module()
    if mpu.model_parallel_is_initialized():
        return mpu.get_model_parallel_group()
    return None


def rank_zero_debug(*args: Any, logger: Optional[Logger] = None, **kwargs: Any) -> None:
    if get_global_rank() != 0:
        return
    logger = logger or _LOGGER
    logger.debug(*args, **kwargs)


def rank_zero_info(*args: Any, logger: Optional[Logger] = None, **kwargs: Any) -> None:
    if get_global_rank() != 0:
        return
    logger = logger or _LOGGER
    logger.info(*args, **kwargs)


def rank_zero_warn(*args: Any, logger: Optional[Logger] = None, **kwargs: Any) -> None:
    if get_global_rank() != 0:
        return
    logger = logger or _LOGGER
    logger.warning(*args, **kwargs)


def rank_zero_error(*args: Any, logger: Optional[Logger] = None, **kwargs: Any) -> None:
    if get_global_rank() != 0:
        return
    logger = logger or _LOGGER
    logger.error(*args, **kwargs)


def mp_rank_zero_warn(
    *args: Any, logger: Optional[Logger] = None, **kwargs: Any
) -> None:
    if get_mp_rank() != 0:
        return
    logger = logger or _LOGGER
    logger.warning(*args, **kwargs)


def mp_rank_zero_debug(
    *args: Any, logger: Optional[Logger] = None, **kwargs: Any
) -> None:
    if get_mp_rank() != 0:
        return
    logger = logger or _LOGGER
    logger.debug(*args, **kwargs)


def mp_rank_zero_info(
    *args: Any, logger: Optional[Logger] = None, **kwargs: Any
) -> None:
    if get_mp_rank() != 0:
        return
    logger = logger or _LOGGER
    logger.info(*args, **kwargs)


def rank_zero_print(*args: Any, **kwargs: Any) -> None:
    if get_global_rank() != 0:
        return
    print(*args, **kwargs)


def all_reduce(
    tensor: torch.Tensor, op: str = "sum", group: Optional[dist.ProcessGroup] = None
) -> torch.Tensor:
    """All-reduces single scalar value if torch distributed is in use."""
    if not dist.is_available() or not dist.is_initialized():
        return tensor

    dop = None
    if op == "sum" or op == "mean":
        dop = dist.ReduceOp.SUM
    elif op == "min":
        dop = dist.ReduceOp.MIN
    elif op == "max":
        dop = dist.ReduceOp.MAX
    elif op == "product":
        dop = dist.ReduceOp.PRODUCT

    dist.all_reduce(tensor, op=dop, group=group)
    if op == "mean":
        tensor /= dist.get_world_size(group)
    return tensor


def mean_reduce_dict(
    data: Dict[str, List[float]],
    device: Union[str, torch.device] = "cuda",
    group: Optional[dist.ProcessGroup] = None,
    dst: int = 0,
) -> Dict[str, AverageMetric]:
    stats: Dict[str, List[float]] = {}
    for k, v in data.items():
        v = [x for x in v if x == x]  # ignore nan
        stats[k] = [sum(v), len(v), sum(x * x for x in v)]

    gather_stats = gather_object(stats, group=group, dst=dst)
    avg_results: Dict[str, AverageMetric] = defaultdict(lambda: AverageMetric(0, 0, 0))
    for results in gather_stats:
        if results is not None:
            for k, (sum_v, len_v, sum_v2) in results.items():
                avg_results[k].update(
                    float(sum_v) / max(len_v, 1),
                    int(len_v),
                    float(sum_v2) / max(len_v, 1),
                )
    return avg_results


def gather_object(
    obj: Any, dst: int = 0, group: Optional[dist.ProcessGroup] = None
) -> List[Any]:
    if not dist.is_initialized() or dist.get_world_size(group) == 1:
        return [obj]
    import torch.distributed.distributed_c10d as c10d

    global_dst = c10d.get_global_rank(group or c10d.GroupMember.WORLD, dst)
    output_list = [None for _ in range(dist.get_world_size(group))]
    results = output_list if get_global_rank() == global_dst else None
    dist.gather_object(obj, results, dst=global_dst, group=group)
    return output_list


def all_gather_object(obj: Any, group: Optional[dist.ProcessGroup] = None) -> List[Any]:
    if not dist.is_initialized() or dist.get_world_size(group) == 1:
        return [obj]
    output_list = [None for _ in range(dist.get_world_size(group))]
    dist.all_gather_object(output_list, obj, group=group)
    return output_list


def broadcast_object_list(
    object_list: List[Any], src: int = 0, group: Optional[dist.ProcessGroup] = None
) -> None:
    if not dist.is_initialized() or dist.get_world_size(group) == 1:
        return
    import torch.distributed.distributed_c10d as c10d

    global_src = c10d.get_global_rank(group or c10d.GroupMember.WORLD, src)
    dist.broadcast_object_list(object_list, src=global_src, group=group)
