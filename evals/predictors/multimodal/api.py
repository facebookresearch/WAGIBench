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

import abc
from typing import Any, Callable, Dict, List, Optional

import torch

from evals.api import Example, Prediction, Predictor
from evals.utils import get_local_rank
from torch.utils.data import Dataset

from tqdm import tqdm


class TransformDataset(Dataset):
    def __init__(
        self,
        annotations: List[Example],  # JSONL lines
        transform: Callable[[Example], Example] = lambda x: x,
    ) -> None:
        super().__init__()
        self.annotations = annotations
        assert len(set(annot["id"] for annot in self.annotations)) == len(
            self.annotations
        )
        self.transform = transform

    def __len__(self) -> int:
        return len(self.annotations)

    def __getitem__(self, index: int) -> Example:
        datum = self.annotations[index]
        return self.transform(datum)


def build_image_dataloader(
    annotations: List[Example],
    annotation_transform: Callable[[Example], Example],
    batch_size: int,
    num_workers: int = 0,
    collate_fn: Optional[Callable[[List[Example]], Example]] = None,
) -> torch.utils.data.DataLoader:
    dataset = TransformDataset(annotations=annotations, transform=annotation_transform)
    mp_context_kwargs = (
        {"multiprocessing_context": "forkserver"} if num_workers > 0 else {}
    )
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        drop_last=False,
        shuffle=False,
        collate_fn=collate_fn,
        **mp_context_kwargs,
    )
    return dataloader


class MultiModalPredictor(Predictor, abc.ABC):
    batch_size: int
    _collate_fn: Optional[Callable[[List[Example]], Example]] = None

    @abc.abstractmethod
    def preprocess_data(self, sample: Example) -> Example:
        """
        This will be used to initialize an instance of `ImageDataset`
        TODO: assume to be single-image only currently

        Args:
            - "image_path": str
            - "prompt": str
                May contain special tokens like "<image>", "{caption}", etc.
            - "targets": List[str]
        """
        pass

    @torch.no_grad()
    def __call__(  # type: ignore
        self,
        annotations: List[Example],
        max_gen_len: int,
        show_progress: bool,
        generate_kwargs: Dict[str, Any],
    ) -> List[Prediction]:
        dataloader = build_image_dataloader(
            annotations=annotations,
            annotation_transform=self.preprocess_data,
            batch_size=self.batch_size,
            collate_fn=self._collate_fn,
            num_workers=getattr(self, "num_workers", 0),
        )

        disable_progress = not show_progress or get_local_rank() != 0
        progress = tqdm(total=len(dataloader), disable=disable_progress, leave=False)

        predictions = []
        for batch in dataloader:
            pred = self.predict_batch(batch, max_gen_len, generate_kwargs)
            predictions.extend(pred)
            progress.update(1)
        progress.close()

        return predictions

    @abc.abstractmethod
    def predict_batch(
        self, batch: Example, max_gen_len: int, generate_kwargs: Dict[str, Any]
    ) -> List[Prediction]:
        pass
