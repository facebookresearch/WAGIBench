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

from typing import List, Union

import numpy as np


def nltk_sentence_bleu(prediction_tokens: List[int], target_tokens: List[int]) -> float:
    try:
        from nltk.translate.bleu_score import (  # type: ignore
            sentence_bleu,
            SmoothingFunction,
        )
    except (ImportError, ModuleNotFoundError):
        return -1.0

    return float(
        sentence_bleu(
            [target_tokens],
            prediction_tokens,
            smoothing_function=SmoothingFunction().method1,
        )
    )


def edit_distance(
    prediction_tokens: Union[str, List[int]], target_tokens: Union[str, List[int]]
) -> float:
    # Get minimum edit distance between prediction and targets in the case of multiple targets
    try:
        import editdistance
    except (ImportError, ModuleNotFoundError):
        return -1.0

    return float(editdistance.eval(prediction_tokens, target_tokens))


def longest_common_substring(
    prediction_tokens: List[int], target_tokens: List[int]
) -> float:
    lengths = np.zeros((len(prediction_tokens), len(target_tokens)), dtype=int).tolist()
    longest = 0

    for i in range(len(prediction_tokens)):
        for j in range(len(target_tokens)):
            if prediction_tokens[i] != target_tokens[j]:
                continue
            elif i == 0 or j == 0:
                lengths[i][j] = 1
            else:
                lengths[i][j] = lengths[i - 1][j - 1] + 1

            longest = max(longest, lengths[i][j])

    return float(longest)


def memorization_score(prediction_tokens: List[int], target_tokens: List[int]) -> float:
    # See "Emergent and Predictable Memorization in Large Language Models"
    # https://arxiv.org/pdf/2304.11158.pdf
    correct = sum(
        pred == target for pred, target in zip(prediction_tokens, target_tokens)
    )
    correct_avg = correct / len(target_tokens)

    return float(correct_avg)
