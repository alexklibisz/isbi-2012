import numpy as np
from skimage.filters import gaussian
from skimage.transform import swirl, resize
from skimage.util import random_noise, crop


def random_transforms(items, nb_min=0, nb_max=6):

    def _zoom(x):
        cropsz = [(int(x.shape[0] * 0.05), int(x.shape[0] * 0.05)),
                  (int(x.shape[1] * 0.05), int(x.shape[1] * 0.05))]
        r = resize(crop(x, cropsz), x.shape)
        if len(np.unique(x)) > 2:
            return r
        return r.round().astype(x.dtype)

    def _swirl(x, strength):
        cx = int(x.shape[0] / 2)
        cy = int(x.shape[1] / 2)
        s = swirl(x, center=(cx, cy), strength=strength, radius=int(x.shape[0] * 0.4))
        if len(np.unique(x)) > 2:
            return s
        return s.round().astype(x.dtype)

    all_transforms = [
        # Non-desctructive transforms.
        lambda x: x,
        lambda x: np.fliplr(x),
        lambda x: np.flipud(x),
        lambda x: np.rot90(x, 1),
        lambda x: np.rot90(x, 2),
        lambda x: np.rot90(x, 3),

        # lambda x: x,
        # lambda x: np.fliplr(x),
        # lambda x: np.flipud(x),
        # lambda x: np.rot90(x, 1),
        # lambda x: np.rot90(x, 2),
        # lambda x: np.rot90(x, 3),

        # # Destructive transforms. These somewhat alter the grount-truth, so I'm not sure if
        # # it's a good idea to use them a lot.
        # lambda x: _swirl(x, 3),
        # lambda x: _zoom(x)
    ]

    n = np.random.randint(nb_min, nb_max + 1)
    items_t = [item.copy() for item in items]
    for _ in range(n):
        idx = np.random.randint(0, len(all_transforms))
        transform = all_transforms[idx]
        items_t = [transform(item) for item in items_t]
    return items_t
