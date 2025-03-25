"""
Provides the EllipsesDataset.
From https://github.com/educating-dip/subspace_dip_learning/blob/main/subspace_dip/data/datasets/ellipses.py
"""

from typing import Union, Iterator, Tuple
import numpy as np
import torch
from torch import Tensor
from itertools import repeat

try:
    from odl import uniform_discr
    from odl.phantom import ellipsoid_phantom
except ModuleNotFoundError:
    print("Please install ODL: pip install git+https://github.com/odlgroup/odl")    

class EllipsesDataset(torch.utils.data.IterableDataset):
    """
    Dataset with images of multiple random ellipses.
    This dataset uses :meth:`odl.phantom.ellipsoid_phantom` to create
    the images. The images are normalized to have a value range of ``[0., 1.]`` with a
    background value of ``0.``.
    """

    def __init__(
        self,
        shape: Tuple[int, int] = (128, 128),
        length: int = 3200,
        fixed_seed: int = 1,
        fold: str = "train",
        max_n_ellipse: int = 70,
    ):
        self.shape = shape
        min_pt = [-self.shape[0] / 2, -self.shape[1] / 2]
        max_pt = [self.shape[0] / 2, self.shape[1] / 2]
        self.space = uniform_discr(min_pt, max_pt, self.shape)
        self.length = length
        self.max_n_ellipse = max_n_ellipse
        self.ellipses_data = []
        self.setup_fold(fixed_seed=fixed_seed, fold=fold)
        super().__init__()

    def setup_fold(self, fixed_seed: int = 1, fold: str = "train"):
        fixed_seed = None if fixed_seed in [False, None] else int(fixed_seed)
        if (fixed_seed is not None) and (fold == "validation"):
            fixed_seed = fixed_seed + 1
        self.rng = np.random.RandomState(fixed_seed)

    def __len__(self) -> Union[int, float]:
        return self.length if self.length is not None else float("inf")

    def _extend_ellipses_data(self, min_length: int) -> None:
        ellipsoids = np.empty((self.max_n_ellipse, 6))
        n_to_generate = max(min_length - len(self.ellipses_data), 0)
        for _ in range(n_to_generate):
            v = self.rng.uniform(-0.4, 1.0, (self.max_n_ellipse,))
            a1 = 0.2 * self.rng.exponential(1.0, (self.max_n_ellipse,))
            a2 = 0.2 * self.rng.exponential(1.0, (self.max_n_ellipse,))
            x = self.rng.uniform(-0.9, 0.9, (self.max_n_ellipse,))
            y = self.rng.uniform(-0.9, 0.9, (self.max_n_ellipse,))
            rot = self.rng.uniform(0.0, 2 * np.pi, (self.max_n_ellipse,))
            n_ellipse = min(self.rng.poisson(self.max_n_ellipse), self.max_n_ellipse)
            v[n_ellipse:] = 0.0
            ellipsoids = np.stack((v, a1, a2, x, y, rot), axis=1)
            self.ellipses_data.append(ellipsoids)

    def _generate_item(self, idx: int) -> Tensor:
        ellipsoids = self.ellipses_data[idx]
        image = ellipsoid_phantom(self.space, ellipsoids)
        # normalize the foreground (all non-zero pixels) to [0., 1.]
        image[np.array(image) != 0.0] -= np.min(image)
        image /= np.max(image)

        return torch.from_numpy(image.asarray()[None]).float()  # add channel dim

    def __iter__(self) -> Iterator[Tensor]:
        it = repeat(None, self.length) if self.length is not None else repeat(None)
        for idx, _ in enumerate(it):
            self._extend_ellipses_data(idx + 1)
            yield self._generate_item(idx)

    def __getitem__(self, idx: int) -> Tensor:
        self._extend_ellipses_data(idx + 1)
        return self._generate_item(idx)


def get_ellipses_dataset(
    fold: str = "train",
    im_size: int = 128,
    length: int = 3200,
    max_n_ellipse: int = 70,
    device=None,
) -> EllipsesDataset:
    image_dataset = EllipsesDataset(
        (im_size, im_size), length=length, fold=fold, max_n_ellipse=max_n_ellipse
    )

    return image_dataset


class DiskDistributedEllipsesDataset(EllipsesDataset):
    def __init__(
        self,
        shape: Tuple[int, int] = (128, 128),
        length: int = 3200,
        fixed_seed: int = 1,
        fold: str = "train",
        diameter: float = 0.4745,
        max_n_ellipse: int = 70,
    ):
        super().__init__(
            shape=shape,
            length=length,
            fixed_seed=fixed_seed,
            fold=fold,
            max_n_ellipse=max_n_ellipse,
        )
        self.diameter = diameter

    def _extend_ellipses_data(self, min_length: int) -> None:
        ellipsoids = np.empty((self.max_n_ellipse, 6))
        n_to_generate = max(min_length - len(self.ellipses_data), 0)
        for _ in range(n_to_generate):
            v = self.rng.uniform(-0.4, 1.0, (self.max_n_ellipse,))
            a1 = 0.2 * self.diameter * self.rng.exponential(1.0, (self.max_n_ellipse,))
            a2 = 0.2 * self.diameter * self.rng.exponential(1.0, (self.max_n_ellipse,))
            c_r = self.rng.triangular(
                0.0, self.diameter, self.diameter, size=(self.max_n_ellipse,)
            )
            c_a = self.rng.uniform(0.0, 2 * np.pi, (self.max_n_ellipse,))
            x = np.cos(c_a) * c_r
            y = np.sin(c_a) * c_r
            rot = self.rng.uniform(0.0, 2 * np.pi, (self.max_n_ellipse,))
            n_ellipse = min(self.rng.poisson(self.max_n_ellipse), self.max_n_ellipse)
            v[n_ellipse:] = 0.0
            ellipsoids = np.stack((v, a1, a2, x, y, rot), axis=1)
            self.ellipses_data.append(ellipsoids)


def get_disk_dist_ellipses_dataset(
    fold: str = "train",
    im_size: int = 128,
    length: int = 3200,
    diameter: float = 0.4745,
    max_n_ellipse: int = 70,
    device=None,
) -> DiskDistributedEllipsesDataset:
    image_dataset = DiskDistributedEllipsesDataset(
        (im_size, im_size),
        **{"length": length, "fold": fold},
        diameter=diameter,
        max_n_ellipse=max_n_ellipse,
    )

    return image_dataset

