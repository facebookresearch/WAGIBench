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

import math
from functools import partial
from typing import Any, Dict, Tuple

import numpy as np

import torch

from PIL import Image
from torchvision import transforms
from torchvision.transforms import functional as F

TGenerationInputs = Tuple[Dict[str, Any], Dict[str, Any]]


def move_to_device(obj: Any, device: torch.device) -> Any:
    if torch.is_tensor(obj):
        return obj.to(device)
    elif isinstance(obj, tuple) or isinstance(obj, list):
        return type(obj)(map(partial(move_to_device, device=device), obj))
    elif isinstance(obj, dict):
        return type(obj)(map(partial(move_to_device, device=device), obj.items()))
    else:
        return obj


class ImagePadding(object):
    """
    Pad image to square (size, size) shape using mean pixel value of the image.
    """

    def __init__(self, size: int) -> None:
        self.size = size

    def __call__(self, image: Image.Image) -> Image.Image:
        w, h = image.size
        new_w = max(w, self.size)
        new_h = max(h, self.size)
        mean_per_channel = tuple(
            np.clip(np.array(image).mean(axis=(0, 1)), 0, 255).astype(np.uint8)
        )
        new_im = Image.new(mode="RGB", size=(new_h, new_w), color=mean_per_channel)  # type: ignore
        new_im.paste(image, (max(0, w - new_w), max(0, h - new_h)))
        return new_im


class BatchSample(Dict[str, Any]):
    """
    Allow data dict keys to be accessed like attributes, this is the general
    data format in the omnivision codebase (e.g. omnivore.data.api.Sample)
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.__dict__: Dict[str, Any] = self

    @property
    def length(self) -> int:
        length = None
        for k, v in self.__dict__.items():
            if length is None:
                length = len(v)
            else:
                assert len(v) == length, f"Batch size mismatch for {k}"
        return 0 if length is None else length


class ImageTransform(object):
    """
    Image transform will resize the longer edge to a given size and pad the shorter edge with mean pixel value of the image.
    """

    def __init__(self, size: int) -> None:
        self.size = size

    def _resize(self, image: Image.Image) -> Image.Image:
        # Resize longer edge to given size.
        w, h = image.size
        scale = w / h

        if scale > 1.0:
            # width > height
            new_w = self.size
            new_h = math.floor(self.size / scale)
        else:
            # height >= width
            new_h = self.size
            new_w = math.floor(self.size * scale)

        image = F.resize(image, (new_h, new_w))
        return image

    def _pad(self, image: Image.Image) -> Image.Image:
        # Pad image to given size
        w, h = image.size
        new_w = max(w, self.size)
        new_h = max(h, self.size)
        mean_per_channel = tuple(
            np.clip(np.array(image).mean(axis=(0, 1)), 0, 255).astype(np.uint8)
        )
        new_im = Image.new(mode="RGB", size=(new_h, new_w), color=mean_per_channel)  # type: ignore
        new_im.paste(image, (max(0, w - new_w), max(0, h - new_h)))
        return new_im

    def __call__(self, image: Image.Image) -> Image.Image:
        image = self._resize(image)
        image = self._pad(image)
        return image


def build_image_transform_xlformers(image_size: int) -> transforms.Compose:
    """
    Build image transform function based on image_size for xlformer model.
    """
    image_normalize_mean = (0.48145466, 0.4578275, 0.40821073)
    image_normalize_std = (0.26862954, 0.26130258, 0.27577711)
    image_transform = transforms.Compose(
        [
            ImageTransform(size=image_size),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=image_normalize_mean,
                std=image_normalize_std,
                inplace=True,
            ),
        ]
    )
    return image_transform


def build_image_transform(image_size: int) -> transforms.Compose:
    """
    Build image transform function based on image_size
    """
    image_transform = transforms.Compose(
        [
            transforms.Resize(
                size=image_size, max_size=image_size + 1, interpolation=3
            ),
            ImagePadding(image_size),
            transforms.CenterCrop(size=image_size),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.48145466, 0.4578275, 0.40821073],
                std=[0.26862954, 0.26130258, 0.27577711],
            ),
        ]
    )
    return image_transform


class VideoTransform(object):
    """
    Video transform will resize the longer edge to a given size and pad the shorter edge with mean pixel value of the image.
    """

    def __init__(self, size):
        self.size = size

    def _resize(self, image: torch.Tensor):
        # Resize longer edge to given size.
        n, c, h, w = image.size()
        scale = w / h

        if scale > 1.0:
            # width > height
            new_w = self.size
            new_h = math.floor(self.size / scale)
        else:
            # height >= width
            new_h = self.size
            new_w = math.floor(self.size * scale)

        image = F.resize(image, (new_h, new_w))
        return image

    def _pad(self, image: torch.Tensor):
        # Pad image to given size
        n, c, h, w = image.size()
        new_w = max(w, self.size)
        new_h = max(h, self.size)

        mean_per_frame_channel = torch.clamp(
            torch.mean(torch.reshape(image, (n, c, -1)), dim=2, dtype=torch.float),
            min=0,
            max=255,
        ).reshape(n, c, 1, 1)

        new_im = torch.empty((n, c, new_h, new_w), dtype=torch.float)
        new_im[:, :] = mean_per_frame_channel
        new_im[:, :, 0:h, 0:w] = image.to(dtype=torch.float)
        new_im = new_im / 255.0
        return new_im

    def __call__(self, image: torch.Tensor):
        image = self._resize(image)
        image = self._pad(image)
        return image


def build_video_transform_xlformers(image_size: int) -> transforms.Compose:
    """
    Build image transform function based on image_size for xlformer model.
    """
    image_normalize_mean = (0.48145466, 0.4578275, 0.40821073)
    image_normalize_std = (0.26862954, 0.26130258, 0.27577711)
    video_transform = transforms.Compose(
        [
            VideoTransform(size=image_size),
            transforms.Normalize(
                mean=image_normalize_mean,
                std=image_normalize_std,
                inplace=True,
            ),
        ]
    )
    return video_transform
