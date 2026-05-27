# SPDX-FileCopyrightText: Aresys S.r.l. <info@aresys.it>
# SPDX-License-Identifier: MIT

"""Tests for point_target_analysis/core/irf.py core functionalities"""

from __future__ import annotations

import numpy as np
import pytest

from perseo_quality.point_targets_analysis.core.irf import (
    compute_islr_2d,
    compute_pslr_2d,
    compute_sslr_2d,
)
from tests.unit_tests import test_utils


class TestPointTargetIRF:
    """Testing point_target_analysis/core/irf.py core functionalities"""

    @pytest.fixture(autouse=True)
    def _setup(self) -> None:
        # creating test data
        self.data, self.peak_pos, self.target_pos = test_utils.generate_data_for_test(
            lines=test_utils.default_input_data_generation["lines"],
            samples=test_utils.default_input_data_generation["samples"],
            samples_start=test_utils.default_input_data_generation["samples_start"],
            lines_step=test_utils.default_input_data_generation["lines_step"],
            samples_step=test_utils.default_input_data_generation["samples_step"],
            fc_hz=test_utils.default_input_data_generation["fc_hz"],
        )
        # resolution
        self.rng_res = 1.1
        self.az_res = 1.2

        # benchmarking values
        self.pslr_ref = (-25.50375928748676, -25.57507201905654, -25.50375928748676)
        self.islr_ref = (-18.58654341614361, -19.459148122742103, -15.99066660365714)
        self.sslr_ref = (-39.91270389195101, -39.912703891950976, -39.912703891950976)
        self.rcs_res = test_utils.ref_data_rcs_results

    def test_compute_pslr(self):
        """Testing IRF PSLR computation"""
        pslr = compute_pslr_2d(data=self.data, resolution=[self.rng_res * 8, self.az_res * 8])
        np.testing.assert_allclose(pslr, self.pslr_ref, atol=1e-6, rtol=0)

    def test_compute_islr(self):
        """Testing IRF ISLR computation"""
        islr = compute_islr_2d(data=self.data, resolution=[self.rng_res * 8, self.az_res * 8])
        np.testing.assert_allclose(islr, self.islr_ref, atol=1e-6, rtol=0)

    def test_compute_sslr(self):
        """Testing IRF SSLR computation"""
        sslr = compute_sslr_2d(
            data=self.data,
            resolution=[self.rng_res * 8, self.az_res * 8],
            side_lobes_directions=(np.inf, 0.0),
        )
        np.testing.assert_allclose(sslr, self.sslr_ref, atol=1e-6, rtol=0)
