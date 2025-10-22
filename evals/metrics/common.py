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

from collections import Counter
from typing import Any, Dict, List, Sequence, Tuple


def f1(prediction: str, targets: List[str]) -> float:
    def _f1(pred_tokens: List[str], gt_tokens: List[str]) -> float:
        common = Counter(pred_tokens) & Counter(gt_tokens)
        num_same = sum(common.values())
        if num_same == 0:
            return 0
        precision = 1.0 * num_same / len(pred_tokens)
        recall = 1.0 * num_same / len(gt_tokens)
        return (2 * precision * recall) / (precision + recall)

    return max(_f1(prediction.split(), target.split()) for target in targets)


def exact_match(prediction: str, targets: List[str]) -> float:
    return max(float(prediction == target) for target in targets)


def exact_match_f1(prediction: str, targets: List[str]) -> Tuple[float, float]:
    return (exact_match(prediction, targets), f1(prediction, targets))


def bleu(prediction: str, targets: List[str], tokenizer=None, **kwargs: Any) -> float:
    import sacrebleu  # type: ignore

    if not tokenizer:
        return sacrebleu.corpus_bleu([prediction], [targets], **kwargs).score

    return sacrebleu.corpus_bleu(
        [prediction], [targets], tokenize=tokenizer, **kwargs
    ).score


def sentence_bleu(prediction: str, targets: List[str], **kwargs: Any) -> float:
    import sacrebleu  # type: ignore

    return sacrebleu.sentence_bleu(prediction, targets, **kwargs).score


def rouge_score(
    prediction: str,
    targets: List[str],
    types: Sequence[str] = ("rouge3", "rougeL"),
    **kwargs: Any,
) -> Dict[str, float]:
    from rouge_score import rouge_scorer  # type: ignore

    scorer = rouge_scorer.RougeScorer(types, **kwargs)
    if hasattr(scorer, "score_multi"):
        scores = scorer.score_multi(targets, prediction)  # type: ignore
    else:
        assert len(targets) == 1, len(targets)
        scores = scorer.score(targets[0], prediction)
    avg_fmeasures = {s: scores[s].fmeasure for s in types}
    return avg_fmeasures
