# SPDX-FileCopyrightText: 2026 Yuma Kakei <yumasansansan@gmail.com>
# SPDX-License-Identifier: GPL-3.0-or-later
"""Settings shared by the tests.

Hypothesis draws its examples from a fixed seed and keeps no database, so that
every run tries the same inputs, as every other random test here does.
"""

from hypothesis import settings

settings.register_profile("fixed", derandomize=True, database=None, deadline=None, print_blob=True)
settings.load_profile("fixed")
