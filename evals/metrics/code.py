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

import ast
import contextlib
import faulthandler
import io
import os
import platform
import signal
import sys
import tempfile
from concurrent.futures import as_completed, ProcessPoolExecutor
from pathlib import Path
from types import FrameType
from typing import Any, Iterator, Optional, Sequence, Union

import numpy as np
import torch.multiprocessing as mp

from evals.utils import rank_zero_info


def is_valid_code(code: str, test: str, timeout: Optional[float] = 5.0) -> bool:
    script: str = f"{code}\n\n{test}"
    result: mp.Queue = mp.Queue()
    p = mp.Process(target=execute, args=(script, result, timeout))
    p.start()
    p.join(timeout=timeout + 1 if timeout is not None else None)
    if p.is_alive():
        p.kill()
    if result.empty():
        result.put("timed out")
    return result.get() == "passed"


def is_valid_python_syntax(code: str) -> bool:
    """simple python code syntax check"""
    try:
        ast.parse(str(code))
    except SyntaxError:
        return False
    return True


def is_correct_output(
    code: str,
    input_test_cases: Sequence[str],
    output_test_cases: Sequence[str],
    timeout_per_testcase: Optional[float],
    timeout_total: Optional[float],
) -> bool:
    result: mp.Queue = mp.Queue()
    p = mp.Process(
        target=execute_with_stdio,
        args=(code, result, input_test_cases, timeout_per_testcase),
    )

    p.start()
    timeout = (
        timeout_per_testcase * len(input_test_cases) + 1
        if timeout_per_testcase is not None
        else None
    )
    if timeout and timeout_total:
        timeout = min(timeout, timeout_total)
    elif not timeout and timeout_total:
        timeout = timeout_total
    p.join(timeout=timeout)
    if p.is_alive():
        # Gracefully kill process allowing for cleanup by capturing the sigterm signal
        p.terminate()
        p.join(timeout=1)
        # Forcefully kill the process if it still seems to be alive. This has been observed to happen when mp=4 with 8 gpus so dpp=2. The suspicion the two dpp ranks are creating processes that are interfering with each other.
        # TODO this should be handled by synchronously locking each process as it runs (even on different ranks), or only running metrics on a single gpu in each node
        if p.is_alive():
            p.kill()

    try:
        # result.get can hang even with an item in the queue. There is a deadlock issue that can unexpectedly occur when the process is killed
        # We handle it for now by wrapping it in a timeout
        with time_limit(1):
            outputs = result.get()
    except TimeoutException:
        outputs = ["example collectively timed out"] * len(input_test_cases)

    pass_count = 0
    for i, output in enumerate(outputs):
        if output == output_test_cases[i]:
            pass_count += 1

    num_cases = len(output_test_cases) if len(output_test_cases) > 0 else 1
    return int(pass_count / num_cases) == 1


def estimate_pass_at_k(inputs: Sequence[str], valid: Sequence[bool], k_: int) -> float:
    c, n = sum(valid), len(inputs)
    if k_ == 1:
        return float(c / n)
    if n - c < k_:
        return 1.0
    return 1.0 - float(np.prod(1.0 - k_ / np.arange(n - c + 1, n + 1)))


def pass_at_k(inputs: Sequence[str], test: str, k: Sequence[int]) -> Sequence[float]:
    valid = [is_valid_code(input, test) for input in inputs]
    return [estimate_pass_at_k(inputs, valid, k_) for k_ in k]


def pass_at_k_with_input_output_test_cases(
    inputs: Sequence[str],
    input_test_cases: Sequence[str],
    output_test_cases: Sequence[str],
    timeout_per_testcase: Optional[float] = 0.1,
    timeout_total: Optional[float] = 2.0,  # Timeout across all test cases
    k: Sequence[int] = [1],
) -> Sequence[float]:
    assert len(input_test_cases) == len(output_test_cases)

    # Creating processes can be slow so is only beneficial if we require a high degree of parallelism
    # At 100 generations, results in a 2x speed-up. At 1000 generations, results in a 12x speed-up
    n_workers = max(1, mp.cpu_count() // 2) if len(inputs) >= 100 else 1
    try:
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            futures = [
                executor.submit(
                    is_correct_output,
                    code,
                    input_test_cases,
                    output_test_cases,
                    timeout_per_testcase,
                    timeout_total,
                )
                for code in inputs
            ]
            valid = [future.result() for future in as_completed(futures)]
    except BaseException as e:
        rank_zero_info(
            f"Unexpected exception occurred trying to check is_correct_output - treating all as incorrect generations: {e}"
        )
        valid = [False] * len(inputs)

    return [estimate_pass_at_k(inputs, valid, k_) for k_ in k]


def is_compilable(inputs: Sequence[str], k: Sequence[int]) -> Sequence[float]:
    def estimate_compilable_at_k(k_: int) -> float:
        c, n = sum(is_compilable_list), len(inputs)
        if k_ == 1:
            return float(c / n)
        if n - c < k_:
            return 1.0
        return 1.0 - float(np.prod(1.0 - k_ / np.arange(n - c + 1, n + 1)))

    is_compilable_list = []
    for input in inputs:
        is_compilable = True
        try:
            compile(input, "<string>", "exec")
        except BaseException as e:
            is_compilable = False
        is_compilable_list.append(is_compilable)

    return [estimate_pass_at_k(inputs, is_compilable_list, k_) for k_ in k]


@contextlib.contextmanager
def chdir(root: Union[str, Path]) -> Iterator[None]:
    if root == ".":
        yield
        return
    cwd = os.getcwd()
    os.chdir(root)
    try:
        yield
    except BaseException as exc:
        raise exc
    finally:
        os.chdir(cwd)


@contextlib.contextmanager
def create_tempdir() -> Iterator[str]:
    with tempfile.TemporaryDirectory() as dirname:
        with chdir(dirname):
            yield dirname


def reliability_guard(maximum_memory_bytes: Optional[int] = None) -> None:
    """
    This disables various destructive functions and prevents the generated code
    from interfering with the test (e.g. fork bomb, killing other processes,
    removing filesystem files, etc.)
    WARNING
    This function is NOT a security sandbox. Untrusted code, including, model-
    generated code, should not be blindly executed outside of one. See the
    Codex paper for more information about OpenAI's code sandbox, and proceed
    with caution.
    """

    if maximum_memory_bytes is not None:
        import resource

        resource.setrlimit(
            resource.RLIMIT_AS, (maximum_memory_bytes, maximum_memory_bytes)
        )
        resource.setrlimit(
            resource.RLIMIT_DATA, (maximum_memory_bytes, maximum_memory_bytes)
        )
        if not platform.uname().system == "Darwin":
            resource.setrlimit(
                resource.RLIMIT_STACK, (maximum_memory_bytes, maximum_memory_bytes)
            )

    faulthandler.disable()

    import builtins

    builtins.exit = None  # type: ignore
    builtins.quit = None  # type: ignore

    import os

    os.environ["OMP_NUM_THREADS"] = "1"

    os.kill = None  # type: ignore
    os.system = None  # type: ignore
    os.putenv = None  # type: ignore
    os.remove = None  # type: ignore
    os.removedirs = None  # type: ignore
    os.rmdir = None  # type: ignore
    os.fchdir = None  # type: ignore
    os.setuid = None  # type: ignore
    os.fork = None  # type: ignore
    os.forkpty = None  # type: ignore
    os.killpg = None  # type: ignore
    os.rename = None  # type: ignore
    os.renames = None  # type: ignore
    os.truncate = None  # type: ignore
    os.replace = None  # type: ignore
    os.unlink = None  # type: ignore
    os.fchmod = None  # type: ignore
    os.fchown = None  # type: ignore
    os.chmod = None  # type: ignore
    os.chown = None  # type: ignore
    os.chroot = None  # type: ignore
    os.fchdir = None  # type: ignore
    os.lchflags = None  # type: ignore
    os.lchmod = None  # type: ignore
    os.lchown = None  # type: ignore
    os.getcwd = None  # type: ignore
    os.chdir = None  # type: ignore

    import shutil

    shutil.rmtree = None  # type: ignore
    shutil.move = None  # type: ignore
    shutil.chown = None  # type: ignore

    import subprocess

    subprocess.Popen = None  # type: ignore

    __builtins__["help"] = None  # type: ignore

    import sys

    sys.modules["ipdb"] = None  # type: ignore
    sys.modules["joblib"] = None  # type: ignore
    sys.modules["resource"] = None  # type: ignore
    sys.modules["psutil"] = None  # type: ignore
    sys.modules["tkinter"] = None  # type: ignore


class TimeoutException(Exception):
    pass


@contextlib.contextmanager
def time_limit(seconds: float) -> Iterator[None]:
    def signal_handler(signum: int, frame: Optional[FrameType]) -> None:
        raise TimeoutException("Timed out!")

    signal.setitimer(signal.ITIMER_REAL, seconds)
    signal.signal(signal.SIGALRM, signal_handler)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)


class WriteOnlyStringIO(io.StringIO):
    """StringIO that throws an exception when it's read from"""

    def read(self, *args, **kwargs):  # type: ignore
        raise IOError

    def readline(self, *args, **kwargs):  # type: ignore
        raise IOError

    def readlines(self, *args, **kwargs):  # type: ignore
        raise IOError

    def readable(self, *args: Any, **kwargs: Any) -> bool:
        """Returns True if the IO object can be read."""
        return False


class redirect_stdin(contextlib._RedirectStream):  # type: ignore
    _stream = "stdin"


@contextlib.contextmanager
def swallow_io() -> Iterator[None]:
    stream = WriteOnlyStringIO()
    with contextlib.redirect_stdout(stream):
        with contextlib.redirect_stderr(stream):
            with redirect_stdin(stream):
                yield


@contextlib.contextmanager
def swallow_stderr() -> Iterator[None]:
    stream = WriteOnlyStringIO()
    with contextlib.redirect_stderr(stream):
        yield


def execute(script: str, result: mp.Queue, timeout: Optional[float]) -> None:
    with create_tempdir():
        # These system calls are needed when cleaning up tempdir.
        import os
        import shutil

        rmtree = shutil.rmtree
        rmdir = os.rmdir
        chdir = os.chdir

        # Disable functionalities that can make destructive changes to the test.
        reliability_guard()

        # Construct the check program and run it.
        try:
            with swallow_io():
                with (
                    time_limit(timeout)
                    if timeout is not None
                    else contextlib.nullcontext()
                ):
                    exec(script, {})
            result.put("passed")
        except TimeoutException:
            result.put("timed out")
        except BaseException as e:
            result.put(f"failed: {e}")
        # Needed for cleaning up.
        shutil.rmtree = rmtree
        os.rmdir = rmdir
        os.chdir = chdir


def execute_with_stdio(
    script: str,
    result: mp.Queue,
    input_test_cases: Sequence[str],
    timeout_per_testcase: Optional[float],
) -> None:
    try:
        # As we don't need to compile for each test case, leads to speedup of 0.3ms * num_test_cases
        compiled_script = compile(script, "<string>", "exec")
    except BaseException as e:
        # Returning immediately if code does not compile saves us from needing to run exec() on each test case
        result.put([f"failed to compile script: {e}"] * len(input_test_cases))
        return

    with create_tempdir():
        # These system calls are needed when cleaning up tempdir.
        import os
        import shutil

        rmtree = shutil.rmtree
        rmdir = os.rmdir
        chdir = os.chdir
        unlink = os.unlink

        original_stdin = sys.stdin
        original_stdout = sys.stdout

        outputs = []

        # This signal handler is needed to ensure we reset stdio and os related variables in case this process is terminated due to timeout.
        # We catch the terminate SIGTERM signal and execute this function, before exiting the process.
        def signal_handler(signum, frame):
            shutil.rmtree = rmtree
            os.rmdir = rmdir
            os.chdir = chdir
            os.unlink = unlink
            sys.stdin = original_stdin
            sys.stdout = original_stdout

            sys.exit(0)

        signal.signal(signal.SIGTERM, signal_handler)

        # Disable functionalities that can make destructive changes to the test.
        reliability_guard()

        # Construct the check program and run it.
        for input_test_case in input_test_cases:
            original_stdin = sys.stdin
            original_stdout = sys.stdout
            try:
                with swallow_stderr():
                    with (
                        time_limit(timeout_per_testcase)
                        if timeout_per_testcase is not None
                        else contextlib.nullcontext()
                    ):
                        input_stream = io.StringIO(input_test_case)
                        sys.stdin = input_stream

                        output_str = io.StringIO()
                        sys.stdout = output_str

                        exec(compiled_script, {})

                        outputs.append(output_str.getvalue())
            except TimeoutException:
                outputs.append("test case timed out")
            except BaseException as e:
                outputs.append(f"test case failed: {e}")

            sys.stdin = original_stdin
            sys.stdout = original_stdout

        result.put(outputs)
        # Needed for cleaning up.
        shutil.rmtree = rmtree
        os.rmdir = rmdir
        os.chdir = chdir
        os.unlink = unlink
