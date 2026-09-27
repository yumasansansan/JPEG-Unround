# SPDX-FileCopyrightText: 2026 Yuma Kakei <yumasansansan@gmail.com>
# SPDX-License-Identifier: GPL-3.0-or-later
"""Small pictures for the tests, with fixed seeds, which the tests encode as JPEG files."""

import numpy as np
import numpy.typing as npt

type Array = npt.NDArray[np.float64]


def picture(rows: int, columns: int, seed: int) -> Array:
    """A canvas of samples about 128: a ramp, a bright disc with a sharp edge, and a little noise."""
    rng = np.random.default_rng(seed)
    y, x = np.mgrid[0:rows, 0:columns].astype(np.float64)
    ramp = 90.0 * (x / max(columns - 1, 1)) - 60.0 * (y / max(rows - 1, 1))
    centre_y, centre_x = rng.uniform(0.3, 0.7) * rows, rng.uniform(0.3, 0.7) * columns
    disc = np.where((y - centre_y) ** 2 + (x - centre_x) ** 2 < (0.25 * min(rows, columns)) ** 2, 70.0, 0.0)
    return np.asarray(ramp + disc + rng.normal(0.0, 2.0, size=(rows, columns)) + 108.0, dtype=np.float64)


def colour_picture(rows: int, columns: int, seed: int) -> Array:
    """An RGB picture, (rows, columns, 3), about 128: ramps of their own in each channel, a disc
    of another colour with a sharp edge, and a little noise."""
    rng = np.random.default_rng(seed)
    y, x = np.mgrid[0:rows, 0:columns].astype(np.float64)
    across, down = x / max(columns - 1, 1), y / max(rows - 1, 1)
    ramps = np.stack((90.0 * across - 40.0 * down, 60.0 * down - 30.0 * across, 50.0 * (across + down)), axis=-1)
    centre_y, centre_x = rng.uniform(0.3, 0.7) * rows, rng.uniform(0.3, 0.7) * columns
    inside = (y - centre_y) ** 2 + (x - centre_x) ** 2 < (0.25 * min(rows, columns)) ** 2
    disc = np.where(inside[:, :, np.newaxis], np.array([70.0, -50.0, 20.0]), 0.0)
    noise = rng.normal(0.0, 2.0, size=(rows, columns, 3))
    return np.asarray(ramps + disc + noise + 100.0, dtype=np.float64)
