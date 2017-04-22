import numpy as np
from skimage.filters import gaussian
from skimage.transform import swirl, resize
from skimage.util import random_noise, crop


def random_transforms(items, nb_min=0, nb_max=5, rng=np.random):

    all_transforms = [
        lambda x: x,
        lambda x: np.fliplr(x),
        lambda x: np.flipud(x),
        lambda x: np.rot90(x, 1),
        lambda x: np.rot90(x, 2),
        lambda x: np.rot90(x, 3),
    ]

    n = rng.randint(nb_min, nb_max + 1)
    items_t = [item.copy() for item in items]
    for _ in range(n):
        idx = rng.randint(0, len(all_transforms))
        transform = all_transforms[idx]
        items_t = [transform(item) for item in items_t]
    return items_t
