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

import unittest
from typing import Any, Callable, Dict, List, Tuple

from evals.metrics import (
    edit_distance,
    longest_common_substring,
    memorization_score,
    nltk_sentence_bleu,
)


class WhitespaceTokenizer:
    def __call__(self, s: str) -> List[str]:
        return s.split()


class StubTokenizer:
    def __init__(self) -> None:
        self.tok_to_id: Dict[str, int] = {}

    def __call__(self, s: str) -> List[int]:
        toks = s.split()
        tokenized = []
        for tok in toks:
            if tok not in self.tok_to_id:
                self.tok_to_id[tok] = len(self.tok_to_id)
            tokenized.append(self.tok_to_id[tok])
        return tokenized


def get_examples() -> List[Tuple[str, str]]:
    return [
        (
            "John Doe and he lives in the United Kingdom .",
            "Jane Doe and she lives in the United States .",
        ),
        (
            "the ratio of the radius of a circle to its",
            "a famous decimal that never enters a repeating pattern .",
        ),
        (
            "Billy Bob . They are on trial for tax fraud",
            "Billy Bob . Are they really on trial for tax",
        ),
        (
            "Billy Bob . They are on trial for tax fraud",
            "Billy Bob . They are on trial for tax fraud",
        ),
    ]


class TestSimilarity(unittest.TestCase):
    def run_scenarios(
        self,
        score_fn: Callable[..., Any],
        expected: List[float],
    ) -> None:
        for i, (q, g) in enumerate(get_examples()):
            # Ensure it works with str tokens and token ids (List[int])
            for tok_cls in [WhitespaceTokenizer, StubTokenizer]:
                with self.subTest(tokenizer=tok_cls.__name__, example_id=i):
                    tok = tok_cls()
                    self.assertAlmostEqual(
                        expected[i],
                        score_fn(tok(q), tok(g)),  # type: ignore
                        places=4,  # type: ignore
                    )

    def test_bleu(self) -> None:
        self.run_scenarios(
            score_fn=nltk_sentence_bleu,
            expected=[0.3247, 0.0211, 0.3799, 1.0],
        )

    def test_edit_distance(self) -> None:
        self.run_scenarios(
            score_fn=edit_distance,
            expected=[3.0, 9.0, 4.0, 0.0],
        )

    def test_memorization_score(self) -> None:
        self.run_scenarios(
            score_fn=memorization_score,
            expected=[0.7, 0.1, 0.3, 1.0],
        )

    def test_longest_common_substring(self) -> None:
        self.run_scenarios(
            score_fn=longest_common_substring,
            expected=[4.0, 1.0, 4.0, 10.0],
        )


if __name__ == "__main__":
    unittest.main()
