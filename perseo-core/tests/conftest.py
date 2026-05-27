# SPDX-FileCopyrightText: Aresys S.r.l. <info@aresys.it>
# SPDX-License-Identifier: MIT

"""Shared pytest fixtures for models module tests."""

import pytest

from tests.fixtures.geometry_data import (
    get_angles_test_data,
    get_doppler_test_data,
    get_ecef_eci_coords_conversions_test_data,
    get_utm2llh_coords_conversions_test_data,
    get_xyz2llh_coords_conversions_test_data,
)
from tests.fixtures.geometry_geocoding_data import (
    get_direct_geocoding_test_data,
    get_direct_geocoding_with_looks_test_data,
    get_inverse_geocoding_test_data,
)
from tests.fixtures.geometry_utilities_data import (
    get_ground_velocity_test_data,
    get_reference_frames_test_data,
    get_rotation_test_data,
)
from tests.fixtures.models_data import get_attitude_test_data, get_testing_trajectory, get_trajectory_test_data
from tests.fixtures.timing_data import get_gps_week_conversion_test_data, get_precise_datetime_to_numpy_test_data
from tests.fixtures.trajectory_angles_data import get_angles_from_trajectory_test_data


@pytest.fixture
def attitude_test_data():
    """Return fixture data for ``Attitude`` tests."""
    return get_attitude_test_data()


@pytest.fixture
def trajectory_test_data():
    """Return fixture data for ``CubicSplineTrajectory`` tests."""
    return get_trajectory_test_data()


@pytest.fixture
def testing_trajectory():
    """Getting a testing trajectory object"""
    return get_testing_trajectory()


@pytest.fixture
def angles_test_data():
    """Return fixture data for look/incidence-angle tests."""
    return get_angles_test_data()


@pytest.fixture
def xyz2llh_coords_conversions_test_data():
    """Return fixture data for ``xyz2llh`` and ``llh2xyz`` tests."""
    return get_xyz2llh_coords_conversions_test_data()


@pytest.fixture
def reference_frames_test_data():
    """Return fixture data for reference-frame tests."""
    return get_reference_frames_test_data()


@pytest.fixture
def rotation_test_data():
    """Return fixture data for Euler-angle rotation tests."""
    return get_rotation_test_data()


@pytest.fixture
def ground_velocity_test_data():
    """Return fixture data for ground-velocity tests."""
    return get_ground_velocity_test_data()


@pytest.fixture
def angles_from_trajectory_test_data():
    """Return fixture data for angles computation from trajectory tests."""
    return get_angles_from_trajectory_test_data()


@pytest.fixture
def gps_week_conversion_test_data():
    """Return fixture data for ``date_to_gps_week`` tests."""
    return get_gps_week_conversion_test_data()


@pytest.fixture
def precise_datetime_to_numpy_test_data():
    """Return fixture data for ``precise_datetime_to_numpy`` tests."""
    return get_precise_datetime_to_numpy_test_data()


@pytest.fixture
def direct_geocoding_with_looks_test_data():
    """Return fixture data for ``direct_geocoding_with_looks`` tests."""
    return get_direct_geocoding_with_looks_test_data()


@pytest.fixture
def direct_geocoding_test_data():
    """Return fixture data for ``direct_geocoding_monostatic`` and ``direct_geocoding_bistatic`` tests."""
    return get_direct_geocoding_test_data()


@pytest.fixture
def inverse_geocoding_test_data():
    """Return fixture data for ``inverse_geocoding_monostatic`` and ``inverse_geocoding_bistatic`` tests."""
    return get_inverse_geocoding_test_data()


@pytest.fixture
def doppler_test_data():
    """Return fixture data for doppler functions tests."""
    return get_doppler_test_data()


@pytest.fixture
def ecef_eci_coords_conversions_test_data():
    """Return fixture data for ``ecef2eci`` and ``eci2ecef`` tests."""
    return get_ecef_eci_coords_conversions_test_data()


@pytest.fixture
def utm2llh_coords_conversions_test_data():
    """Return fixture data for ``utm2llh`` and ``llh2utm`` tests."""
    return get_utm2llh_coords_conversions_test_data()
