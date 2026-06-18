# SPDX-FileCopyrightText: Aresys S.r.l. <info@aresys.it>
# SPDX-License-Identifier: MIT

"""
Atmospheric Delay Corrections: Tropospheric submodule.

Through neutral troposphere, propagation delays are caused by air refractivity gradients.

Air refractivity gradients in the troposphere are first due to the dry air pressure and temperature and, to a lesser
extent, to air moisture and condensed water in clouds or rain. The dry air temperature and pressure can be considered
as mostly vertically stratified, and thus lead to a large phase delay varying only with elevation in a radar scene.
On the contrary, the air water vapor varies both vertically and laterally over short distances.
"""

from __future__ import annotations

import re
from datetime import datetime
from enum import Enum, auto
from io import BytesIO, StringIO
from pathlib import Path

import numpy as np
import pandas as pd
from perseo_core.geometry import compute_incidence_angles_core
from perseo_core.geometry.coordinates import xyz2llh
from perseo_core.timing import PreciseDateTime
from scipy.interpolate import griddata, interp1d

from perseo_perturbations import grid_stations_coarse, grid_stations_fine, tropospheric_legendre_coeff

# constants definition
SECONDS_IN_A_DAY = 86400  # [s]
DAYS_IN_YEAR = 365.25  # [d]
ATMOSPHERIC_PRESSURE_MB = 1013.25  # [mbar]
TROPOSPHERE_TEMP_LAPSE_RATE = 0.0065  # [K/m]
TROPOSPHERE_TEMP_REFERENCE = 288.15  # [K]
GRAVITATIONAL_ACCELERATION = 9.80665  # [m/s^2]
MOLAR_MASS_AIR = 0.0289644  # [kg/mol]
UNIVERSAL_GAS_CONSTANT = 8.3144598  # [J/mol/K]
SAASTAMOINEN_CNTS = (0.0022768, 0.00266, 0.28e-6)


# custom errors
class TroposphericGridStationFileNotFoundError(FileNotFoundError):
    """Could not find the specified grid points station coordinates file"""


class TroposphericGridResolutionNotSupportedError(ValueError):
    """Grid resolution provided is not supported"""


# custom enum classes
class TroposphericMapModel(Enum):
    """Tropospheric map data model"""

    DORIS = auto()
    GNSS = auto()
    GRID = auto()
    SLR = auto()
    VLBI = auto()


class TroposphericGRIDResolution(Enum):
    """Tropospheric map resolution for GRID model"""

    FINE = "1x1"
    MEDIUM = "2.5x2"
    COARSE = "5x5"


class TroposphericMapType(Enum):
    """Tropospheric map data type"""

    GRAD = auto()
    LHG = auto()
    RAYTR = auto()
    V3GR = auto()
    VMF1 = auto()
    VMF3 = auto()
    VMF3o = auto()


class TroposphericMapVersion(Enum):
    """Tropospheric map data version"""

    EI = auto()
    FC = auto()
    OP = auto()
    RADIATE = auto()
    TRP = auto()


class TroposphericGridInterpolationMethod(Enum):
    """Tropospheric grid interpolation method using griddata from scipy.interpolate, check the function for the
    available methods"""

    LINEAR = auto()
    NEAREST = auto()
    CUBIC = auto()


def _troposphere_barometric_formula(height: float) -> float:
    """Computing the barometric formula (pressure variation with altitude) for the troposphere ISA level.

    Implemented formula:

    $$
    P = P_0 \\cdot \\left[1 - \\frac{L_t}{T_t} \\cdot (h-h_t)\\right]^\\left(\\frac{g_0\\cdot M}{L_t \\cdot
    R^{\\prime}}\\right)
    $$

    where $t$ subscript denotes the troposphere ISA level, $L$ is the temperature lapse rate, $T$ the reference
    temperature, $h$ the height, $g_0$ the gravitational acceleration constant, $M$ the air molar mass and
    $R^{\\prime}}$ the universal gas constant.

    Parameters
    ----------
    height : float
        height from sea level

    Returns
    -------
    float
        pressure at that height
    """

    exponent = GRAVITATIONAL_ACCELERATION * MOLAR_MASS_AIR / UNIVERSAL_GAS_CONSTANT / TROPOSPHERE_TEMP_LAPSE_RATE
    pressure = (
        ATMOSPHERIC_PRESSURE_MB * (1 - TROPOSPHERE_TEMP_LAPSE_RATE / TROPOSPHERE_TEMP_REFERENCE * height) ** exponent
    )

    return pressure


def generate_tropospheric_map_name_for_vmf_data(
    acq_time: PreciseDateTime, map_type: TroposphericMapType
) -> tuple[list[str], list[PreciseDateTime]]:
    """Generating the name of the 4 files needed to perform the tropospheric delay estimation for VMF data.

    Data can be found at this location:
    https://vmf.geo.tuwien.ac.at/trop_products/

    A note on file selection criterion:
    files are selected with respect to the map data type and the acquisition time
    tropospheric map files are recorded every 6 hours across the day, every day, so there are 4 files pertaining
    to each day H00, H06, H12, H18
    4 files are needed to perform a proper interpolation of values and estimate at best the tropospheric delay
    the main file, the closest to the acquisition time hour (approximated by defect)
    the previous file (the one 6 hours before the main one)
    the two files after the main one (6 hours after and 12 hours after)

    Parameters
    ----------
    acq_time : PreciseDateTime
        acquisition time at which the tropospheric delay must be estimated
    map_type : TroposphericMapType
        type of tropospheric map data

    Returns
    -------
    list[str]
        list of file names
    list[PreciseDateTime]
        list of file dates in PreciseDatetime form
    """

    base_name = map_type.name + "_"

    # finding the base date corresponding to the acquisition time of interest
    acq_base_date_components = [
        acq_time.year,
        acq_time.month,
        acq_time.day_of_the_month,
    ]
    acq_base_date = PreciseDateTime.fromisoformat(
        datetime.strptime("-".join([str(t) for t in acq_base_date_components]), "%Y-%m-%d").isoformat()
    )
    # determine the previous and next days
    prev_date = acq_base_date - SECONDS_IN_A_DAY
    next_date = acq_base_date + SECONDS_IN_A_DAY

    # looping over indexes to select proper files based on which is the closest one to the acquisition date
    time_ids = [0, 6, 12, 18]
    closest_recorded_hour_id = np.where([acq_time.hour_of_day >= t for t in time_ids])[0][-1]
    closest_acquisition = acq_base_date_components.copy()
    closest_acquisition.append(time_ids[closest_recorded_hour_id])

    # to properly interpolate and estimate data, the previous and the following two recordings must be taken
    previous_date_id = closest_recorded_hour_id - 1
    next_date_id = (closest_recorded_hour_id + 1) % len(time_ids)
    next2_date_id = (closest_recorded_hour_id + 2) % len(time_ids)

    # initializing all other time combinations for file searching
    previous_acquisition = acq_base_date_components.copy()
    previous_acquisition.append(time_ids[previous_date_id])
    next_acquisition = acq_base_date_components.copy()
    next_acquisition.append(time_ids[next_date_id])
    next2_acquisition = acq_base_date_components.copy()
    next2_acquisition.append(time_ids[next2_date_id])

    if closest_recorded_hour_id == 0:
        # if previous recording is a day before
        previous_acquisition = [
            prev_date.year,
            prev_date.month,
            prev_date.day_of_the_month,
            time_ids[previous_date_id],
        ]

    if closest_recorded_hour_id in (2, 3):
        # if next next recording is the day after
        next2_acquisition = [next_date.year, next_date.month, next_date.day_of_the_month, time_ids[next2_date_id]]

    if closest_recorded_hour_id == 3:
        # if also the next recording is the day after
        next_acquisition = [next_date.year, next_date.month, next_date.day_of_the_month, time_ids[next_date_id]]

    # assembling filenames
    file_names = [
        base_name + f"{t[0]}" + f"{t[1]:02}" + f"{t[2]:02}" + f".H{t[3]:02}"
        for t in [previous_acquisition, closest_acquisition, next_acquisition, next2_acquisition]
    ]
    times = [previous_acquisition, closest_acquisition, next_acquisition, next2_acquisition]
    times = [[str(t) for t in lst] for lst in times]
    times = [PreciseDateTime.fromisoformat(datetime.strptime("-".join(t), "%Y-%m-%d-%H").isoformat()) for t in times]

    return file_names, times


# tropospheric delay estimator from VMF3 data
class TroposphericDelayEstimator:
    """Tropospheric Delay Estimator from Vienna Mapping Function 3 data.
    The delay is due to the refractive index of the medium the signal is passing through.

    The Troposphere is neutral and non-dispersive for frequencies up to 30 GHz, so the path through this medium is not
    depended on the carrier frequency for microwave signals, in contrast to the ionosphere.

    The refractive index of the troposphere is a function of pressure, temperature and partial water vapor pressure.
    The total delay can be decomposed into a wet and dry (hydrostatic) component.
    - hydrostatic part comprises approximately 90% of the overall delay and is well suited to pressure driven
    modelling. Closely linked to topography, reflector height has to be taken into account.
    - wet component undergoes rapid changes due to the high spatiotemporal variability of water vapor.

    The equation to be solved is the following:

    $$
    \\Delta L (\\epsilon) = \\Delta L_h^Z \\cdot mf_h(\\epsilon) +  \\Delta L_w^Z \\cdot mf_w(\\epsilon)
    $$

    where $\\Delta L(\\epsilon)$ is the total delay time and it is composed by an hydrostatic component $h$ and a wet
    component $w$.
    $mf(\\epsilon)$ are the wet and hydrostatic mapping functions generated in this algorithm starting from the Vienna
    Mapping Function 3 data and the spherical harmonics Legendre polynomials.
    """

    def __init__(
        self,
        acquisition_time: PreciseDateTime,
        interpolation_method: TroposphericGridInterpolationMethod,
        map_folder: Path = None,
        map_model: TroposphericMapModel = TroposphericMapModel.GRID,
        map_grid_res: TroposphericGRIDResolution = TroposphericGRIDResolution.FINE,
        map_type: TroposphericMapType = TroposphericMapType.VMF3,
        map_version: TroposphericMapVersion = TroposphericMapVersion.OP,
    ) -> None:
        """Tropospheric delay estimation using Vienna Mapping Function 3 (VMF3) data. Empirical ""b" and "c"
        coefficients, "a" coefficients determined epoch-wise (00, 06, 12, 18 UT) from ray traced-delays at 3°
        elevation and 8 equally spaced azimuth angles.

        Only "OP" (operational) "GRID" (grid-wise) data are supported in this application.

        Reference articles:
        Landskron, D., Böhm, J. VMF3/GPT3: refined discrete and empirical troposphere mapping functions.
        J Geod 92, 349-360 (2018). "https://doi.org/10.1007/s00190-017-1066-2"

        U. Balss et al., "Survey protocol for geometric SAR sensor analysis", German Aerospace Center (DLR),
        Tech. Univ. Munich (TUM), Remote Sensing Lab. Univ. Zurich, Zürich, Switzerland,
        Tech. Rep. DLRFRM4SAR-TN-200, Apr. 2018.

        Data source:
        VMF Data Server: "https://vmf.geo.tuwien.ac.at/"

        Parameters
        ----------
        acquisition_time : PreciseDateTime
            acquisition time of the scene, a.k.a. the time at which estimate the delay
        interpolation_method : TroposphericGridInterpolationMethod
            method for data grid interpolation
        map_folder : Path, optional
            path to the folder containing the map files, by default None
        map_model : TroposphericMapModel, optional
            map file model, by default TroposphericMapModel.GRID
        map_grid_res : TroposphericGRIDResolution, optional
            map grid resolution, by default TroposphericGRIDResolution.FINE
        map_type : TroposphericMapType, optional
            map data type, by default TroposphericMapType.VMF3
        map_version : TroposphericMapVersion, optional
            map data version, by default TroposphericMapVersion.OP
        """
        self.acquisition_time = acquisition_time
        self.interp_method = interpolation_method
        self.map_model = map_model
        self.map_type = map_type
        self.map_version = map_version
        self.map_grid_res = map_grid_res

        # map folder
        self.tropospheric_map_folder = None
        if isinstance(map_folder, Path):
            self.tropospheric_map_folder = map_folder
        elif isinstance(map_folder, str):
            self.tropospheric_map_folder = Path(map_folder)

    def _load_station_altitudes(
        self, grid: TroposphericGRIDResolution, search_input_fldr: bool = False
    ) -> pd.DataFrame:
        """Loading data from grid points station coordinates files. Using default ones in this module resources if not
        explicitly specified by the search_input_fldr flag.

        Online Ref: 'https://vmf.geo.tuwien.ac.at/station_coord_files/'

        Parameters
        ----------
        grid : TroposphericGRIDResolution
            grid resolution enum
        search_input_fldr : bool, optional
            if this flag is True, the files are searched in the input map directory provided at init, by default False

        Returns
        -------
        pd.DataFrame
            pandas dataframe containing the grid point station coordinates

        Raises
        ------
        TroposphericGridStationFileNotFoundError
            grid points station coordinates file not found in input folder
        GridResolutionNotSupportedError
            selected grid resolution not implemented
        """

        # loading data from file
        col_names = ["point", "lat", "lon", "ellipsoidal_height_m", "orthometric_height_m"]
        if search_input_fldr:
            # if files should be loaded from given folder path
            file = self.tropospheric_map_folder.joinpath("gridpoint_coord_" + grid.value).with_suffix(".txt")
            if file.is_file():
                with open(file, mode="rb") as f_in:
                    content = f_in.read()
                    clean_file = re.sub(b" +", b";", content)
            else:
                raise TroposphericGridStationFileNotFoundError(f"{str(file)} not found")
        else:
            # if files are default ones, stored in this module resources
            if grid == TroposphericGRIDResolution.FINE:
                grid_file = grid_stations_fine.read_bytes()
                clean_file = re.sub(b" +", b";", grid_file)
            elif grid == TroposphericGRIDResolution.COARSE:
                grid_file = grid_stations_coarse.read_bytes()
                clean_file = re.sub(b" +", b";", grid_file)
            else:
                raise TroposphericGridResolutionNotSupportedError(f"{grid} not supported")

        # converting data to pandas dataframe
        grid_data = pd.read_csv(BytesIO(clean_file), sep=";", comment="%", header=None, names=col_names)
        grid_data.drop(columns=["point"], inplace=True)

        # shifting longitude axes ([0,360]->[-180,180])
        grid_data.loc[grid_data["lon"] > 180, "lon"] -= 360

        return grid_data

    @staticmethod
    def _generate_lagrange_polynomials(
        x_uv: float, y_uv: float, z_uv: float, poly_order: int = 12
    ) -> tuple[np.ndarray, np.ndarray]:
        """Calculating the Lagrange spherical harmonics polynomial arrays up to order and degree poly_order, by default
        12.

        Parameters
        ----------
        x_uv : float
            x unit vector
        y_uv : float
            y unit vector
        z_uv : float
            z unit vector
        poly_order : int, optional
            order and degree of polynomial, by default 12

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            Legendre polynomial arrays
        """

        v_func = np.zeros((poly_order + 1, poly_order + 1))
        w_func = np.zeros((poly_order + 1, poly_order + 1))
        v_func[0, 0] = 1
        w_func[0, 0] = 0
        v_func[1, 0] = z_uv * v_func[0, 0]
        w_func[1, 0] = 0

        # populating lagrange spherical harmonics polynomials arrays
        for num in range(1, poly_order):
            v_func[num + 1, 0] = ((2 * num + 1) * z_uv * v_func[num, 0] - (num) * v_func[num - 1, 0]) / (num + 1)
            w_func[num + 1, 0] = 0

        for num in range(poly_order):
            v_func[num + 1, num + 1] = (2 * num + 1) * (x_uv * v_func[num, num] - y_uv * w_func[num, num])
            w_func[num + 1, num + 1] = (2 * num + 1) * (x_uv * w_func[num, num] + y_uv * v_func[num, num])
            if num < poly_order - 1:
                v_func[num + 2, num + 1] = (2 * num + 3) * z_uv * v_func[num + 1, num + 1]
                w_func[num + 2, num + 1] = (2 * num + 3) * z_uv * w_func[num + 1, num + 1]

            for num_ in range(num + 2, poly_order):
                v_func[num_ + 1, num + 1] = (
                    (2 * num_ + 1) * z_uv * v_func[num_, num + 1] - (num_ + num + 1) * v_func[num_ - 1, num + 1]
                ) / (num_ - num)
                w_func[num_ + 1, num + 1] = (
                    (2 * num_ + 1) * z_uv * w_func[num_, num + 1] - (num_ + num + 1) * w_func[num_ - 1, num + 1]
                ) / (num_ - num)

        return v_func, w_func

    def _generate_mapping_function(
        self,
        lat: np.ndarray,
        lon: np.ndarray,
        acq_time: PreciseDateTime,
        incidence_angle: np.ndarray,
        a_h: np.ndarray,
        a_w: np.ndarray,
    ) -> dict:
        """Generating the wet and hydrostatic mapping functions to estimate the zenith delay. This computation is
        performed for each value in the input arrays separately in case of multiple point targets to be analyzed.

        python-implementation and refinement of matlab script from
        (c) Department of Geodesy and Geoinformation, Vienna University of Technology, 2016

        This subroutine determines the VMF3 hydrostatic and wet mapping functions.
        The "a" coefficients have to be inserted from discrete data, while the "b" and "c" coefficients are of
        empirical nature containing a geographical and temporal dependence, represented in spherical harmonics. The
        spherical harmonics coefficients are developed to degree and order 12 and are based on a 5x5 grid containing
        ray-tracing data from 2001-2010.

        Hydrostatic and wet mapping functions are evaluated in the form (latex):

        $$
        mf = \\frac{1+\\frac{a}{1+\\frac{b}{1+c}}}{\\sin(\\epsilon)+\\frac{a}{\\sin(\\epsilon)+\\frac{b}{\\sin(\\epsilon) + c}}}
        $$

        where $\\epsilon$ is the elevation angle (incidence angle in this case).

        Parameters
        ----------
        lat : np.ndarray
            array of point targets latitude coordinates in radians
        lon : np.ndarray
            array of point targets longitude coordinates in radians
        acq_time : PreciseDateTime
            acquisition time of the scene at which the mapping factors should be evaluated
        incidence_angle : np.ndarray
            array of incidence angles for each point target in radians
        a_h : np.ndarray
            array of hydrostatic "a" coefficients for each point target
        a_w : np.ndarray
            array of wet "a" coefficients for each point target

        Returns
        -------
        dict
            dictionary with keys 'hydrostatic' and 'wet' for both mapping functions evaluated for each point target
        """  # noqa: E501

        # loading Legendre polynomials for bh, bw, ch, cw coefficients
        tropospheric_legendre_coeff_loaded = dict.fromkeys(tropospheric_legendre_coeff.keys())
        for key, value in tropospheric_legendre_coeff.items():
            tropospheric_legendre_coeff_loaded[key] = pd.read_csv(BytesIO(value), sep=" ", header=None).to_numpy()

        # computing unit vectors
        distance_from_pole = np.pi / 2 - lat
        x_uv = np.sin(distance_from_pole) * np.cos(lon)
        y_uv = np.sin(distance_from_pole) * np.sin(lon)
        z_uv = np.cos(distance_from_pole)

        # compute Legendre polynomials
        poly_order = 12  # should not be changed

        # evaluating the mapping functions for each point target
        mapping_functions = {"wet": [], "hydrostatic": []}
        for point in range(lat.size):
            v_func, w_func = self._generate_lagrange_polynomials(x_uv=x_uv[point], y_uv=y_uv[point], z_uv=z_uv[point])

            epsilon = np.pi / 2 - incidence_angle[point]

            coeff = {}
            coeff["bh"] = [0, 0, 0, 0, 0]
            coeff["bw"] = [0, 0, 0, 0, 0]
            coeff["ch"] = [0, 0, 0, 0, 0]
            coeff["cw"] = [0, 0, 0, 0, 0]
            for num in range(poly_order + 1):
                cumulative_idx = sum(range(num + 1))
                for key, value in coeff.items():
                    for num_ in range(len(value)):
                        coeff[key][num_] += np.sum(
                            tropospheric_legendre_coeff_loaded["anm_" + key][
                                cumulative_idx : cumulative_idx + num + 1, num_
                            ]
                            * v_func[num, : num + 1]
                            + tropospheric_legendre_coeff_loaded["bnm_" + key][
                                cumulative_idx : cumulative_idx + num + 1, num_
                            ]
                            * w_func[num, : num + 1]
                        )

            # adding the seasonal amplitudes for the specified day of the year to the mean values
            doy_ratio_rad = acq_time.day_of_the_year / DAYS_IN_YEAR * 2 * np.pi
            for key, value in coeff.items():
                coeff[key] = (
                    value[0]
                    + value[1] * np.cos(doy_ratio_rad)
                    + value[2] * np.sin(doy_ratio_rad)
                    + value[3] * np.cos(doy_ratio_rad * 2)
                    + value[4] * np.sin(doy_ratio_rad * 2)
                )

            # computing the mapping functions
            mfh = (1 + (a_h[point] / (1 + coeff["bh"] / (1 + coeff["ch"])))) / (
                np.sin(epsilon) + (a_h[point] / (np.sin(epsilon) + coeff["bh"] / (np.sin(epsilon) + coeff["ch"])))
            )
            mfw = (1 + (a_w[point] / (1 + coeff["bw"] / (1 + coeff["cw"])))) / (
                np.sin(epsilon) + (a_w[point] / (np.sin(epsilon) + coeff["bw"] / (np.sin(epsilon) + coeff["cw"])))
            )
            mapping_functions["wet"].append(mfw)
            mapping_functions["hydrostatic"].append(mfh)

        return mapping_functions

    @staticmethod
    def _filtering_df_lat_lon(
        data: list[pd.DataFrame], lat_bound: tuple[float, float], lon_bound: tuple[float, float]
    ) -> list[pd.DataFrame]:
        """Filtering input dataframes by latitude and longitude based on the provided boundaries to select only rows
        whit lat/lon inside those intervals.

        Parameters
        ----------
        data : list[pd.DataFrame]
            list of dataframes with 'lat' and 'lon' columns
        lat_bound : tuple[float, float]
            latitude boundaries, in the form (maximum lat, minimum lat)
        lon_bound : tuple[float, float]
            longitude boundaries, in the form (maximum lon, minimum lon)

        Returns
        -------
        list[pd.DataFrame]
            same list of input but with filtered dataframes
        """

        filtered_data = []
        for df_ in data:
            filtered_data.append(
                df_.query("lat < @lat_bound[0] & lat > @lat_bound[1] & lon < @lon_bound[0] & lon > @lon_bound[1]")
            )

        return filtered_data

    @staticmethod
    def _interpolating_lat_lon(
        lat: tuple[np.ndarray],
        lon: tuple[np.ndarray],
        values: tuple[np.ndarray],
        method: TroposphericGridInterpolationMethod,
    ) -> list[np.ndarray]:
        """Interpolating the input latitude and longitude data to the desired value. Latitude is provided as a tuple of
        arrays, with the grid axis as the first element and the desired interpolation value as the second element.
        Same for longitude. Values are the quantities to be estimated through interpolation at the desired lat, lot
        coordinates provided as lat[1] and lon[1].

        Parameters
        ----------
        lat : tuple[np.ndarray]
            tuple of 2 arrays, the latitude axis [0] and the value(s) to be interpolated [1]
        lon : tuple[np.ndarray]
            tuple of 2 arrays, the longitude axis [0] and the value(s) to be interpolated [1]
        values : tuple[np.ndarray]
            tuple of arrays of values associated to each point of the lat-lon grid (same length of lon[0] and lat[0]).
            For each array in this tuple, interpolated value is appended to the output list.
        method : TroposphericGridInterpolationMethod
            type of interpolation of scipy griddata (linear, cubic, nearest)

        Returns
        -------
        list[np.ndarray]
            interpolated values
        """

        # vertices of the lat-lon grid
        grid_points = np.vstack((lon[0], lat[0])).T
        # points at which to interpolate data
        points_to_be_interpolated = np.vstack((lon[1], lat[1])).T

        interp_space_values = [griddata(grid_points, v, points_to_be_interpolated, method.name.lower()) for v in values]

        return interp_space_values

    @staticmethod
    def read_vmf3_files(files: list[Path]) -> list[pd.DataFrame]:
        """Reading VMF3 OP GRID troposphere data files. VMF3 files are provided as tabular textual data, with an header.
        Data are divided in the following columns:
        lat: latitude in deg,
        lon: longitude in deg,
        ah: "a" coefficient, hydrostatic (dry gases contribution)
        aw: "a" coefficient, wet (water vapor contribution)
        zhd: zenith hydrostatic delay in meters,
        zwd: zenith wet delay in meters

        Parameters
        ----------
        files : list[Path]
            list of path to files

        Returns
        -------
        list[pd.DataFrame]
            list of pandas dataframe from loaded files
        """

        # read troposphere VMF3 map file
        data = []
        for file in files:
            with open(file, mode="rb") as f_in:
                # reading the file in binary format
                lines = f_in.read().splitlines()
                # separating header from the body [lines starting with !]
                header = [line for line in lines if b"!" in line]
                body = [line.decode("UTF-8").lstrip().rstrip() for line in lines if line not in header]
                body = [re.sub(" +", ";", line) for line in body]
                header = [line.decode("UTF-8") for line in header]

            # getting column names from header
            col_names = [h for h in header if "Data_types" in h][0]
            col_names = re.findall(r"\(.*\)", col_names)[0].replace("(", "").replace(")", "").split()

            # reading data as csv from loaded lines
            data_ = pd.read_csv(StringIO("\n".join(body)), names=col_names, header=None, sep=";")

            # shifting longitude axes ([0,360]->[-180,180])
            data_.loc[data_["lon"] > 180, "lon"] -= 360

            data.append(data_)

        return data

    def estimate_delay(
        self, point_targets_coords: np.ndarray, sat_xyz_coords: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Estimate the tropospheric wet and hydrostatic delays from VMF3 data.

        Correcting zenith hydrostatic and wet delays by taking into account the height due to topography dependence.

        Other formulas used in this method:

        Saastamoinen J. formula, hydrostatic delay

        $$
        \\Delta L_h^Z(h) = \\frac{0.0022768 \\cdot P}{1-0.00266 \\cdot \\cos(2\\phi) -0.28 \\cdot 10^{-6} \\cdot
        h_{ell}}
        $$

        wet delay exponential empirical decay correction by height

        $$
        \\Delta L_{w_s}^Z(h) = \\Delta L_{w_g}^Z(h) \\cdot e^{-\\frac{h_s - h_g}{2000}}
        $$

        Parameters
        ----------
        point_targets_coords : np.ndarray
            point targets XYZ coordinates as numpy array of shape Nx3
        sat_xyz_coords : np.ndarray
            satellite XYZ coordinates at which calibration targets are seen as numpy array of shape Nx3

        Returns
        -------
        np.ndarray
            array of estimated hydrostatic delay, one for each point target
        np.ndarray
            array of estimated wet delay, one for each point target

        Raises
        ------
        RuntimeError
            files provided are not in VMF3 format
        """  # noqa: E501

        # determine the map files to be loaded
        files, time_dates = generate_tropospheric_map_name_for_vmf_data(
            acq_time=self.acquisition_time, map_type=self.map_type
        )
        if self.tropospheric_map_folder is not None:
            filepaths = [self.tropospheric_map_folder.joinpath(f) for f in files]

        if self.map_type == TroposphericMapType.VMF3:
            data_list = self.read_vmf3_files(files=filepaths)
        else:
            raise RuntimeError("Files different from VMF3 are not supported")

        llh_coordinates = xyz2llh(point_targets_coords)
        lat = np.rad2deg(llh_coordinates[:, 0])
        lon = np.rad2deg(llh_coordinates[:, 1])
        height = llh_coordinates[:, 2]  # targets height above ellipsoid

        # filtering dataframes lat and lon around point target area
        filtered_data = self._filtering_df_lat_lon(
            data=data_list,
            lat_bound=(np.max(lat) + 10, np.min(lat) - 10),
            lon_bound=(np.max(lon) + 10, np.min(lon) - 10),
        )

        # defining axes for data interpolation along latitude, longitude and time
        lat_axis = np.vstack([item["lat"].to_numpy() for item in filtered_data])
        lon_axis = np.vstack([item["lon"].to_numpy() for item in filtered_data])
        # checking that latitude values in each filtered dataset are equal
        assert np.isclose(lat_axis, lat_axis[0]).all() & np.isclose(lon_axis, lon_axis[0]).all()
        lat_axis = lat_axis[0]
        lon_axis = lon_axis[0]
        # time axis
        time_axis_s = np.array([t - time_dates[0] for t in time_dates])
        acq_time_rel_s = self.acquisition_time - time_dates[0]

        interpolated_values = [
            self._interpolating_lat_lon(
                lat=(lat_axis, lat),
                lon=(lon_axis, lon),
                values=(
                    df["ah"].to_numpy(),
                    df["aw"].to_numpy(),
                    df["zhd"].to_numpy(),
                    df["zwd"].to_numpy(),
                ),
                method=self.interp_method,
            )
            for df in filtered_data
        ]

        # rearranging interpolated values by map and value
        # each array in this list belongs to a distinct tropospheric data (ah, aw, zhd, zwd)
        # each array is of shape (n, m) where n is the number of timestamps (aka number of map files processed) while
        # m is the number of point targets analyzed
        interp_values = [np.vstack([t[i] for t in interpolated_values]) for i in range(len(interpolated_values))]
        interp_troposphere_values = []
        for t_value in interp_values:
            interp_troposphere_values.append(
                np.array(
                    [
                        interp1d(time_axis_s, t_value[:, i], self.interp_method.name.lower())(acq_time_rel_s)
                        for i in range(lat.size)
                    ]
                )
            )
        ah_interp, aw_interp, zhd_interp, zwd_interp = interp_troposphere_values

        # building mapping factors from coefficients and spherical harmonics
        incidence_angles = compute_incidence_angles_core(
            sensor_positions=sat_xyz_coords, ground_points=point_targets_coords
        )
        mapping_factors = self._generate_mapping_function(
            lat=np.radians(lat),
            lon=np.radians(lon),
            acq_time=self.acquisition_time,
            incidence_angle=incidence_angles,
            a_h=ah_interp,
            a_w=aw_interp,
        )

        # correcting delay for delta altitude between troposphere data recording station and point target
        # loading the station grid coordinates file
        grid_point_station = self._load_station_altitudes(grid=self.map_grid_res, search_input_fldr=False)

        # interpolate lat and lon values to get the right ellipsoidal height value for the point targets
        # height of elevation model (ETOPO5) at target location [m]
        point_target_heights_gridpoint_interp = griddata(
            grid_point_station[["lon", "lat"]].to_numpy(),
            grid_point_station["ellipsoidal_height_m"].to_numpy(),
            np.vstack((lon, lat)).T,
            self.interp_method.name.lower(),
        )

        # retrieving atmospheric pressure at point target location from the interpolated zenith hydrostatic delay
        # (inverse relationship) of Saastamoinen formula

        # evaluating pressure at point target elevation model (ETOPO5) heights [mbar]
        pressure_at_point_targets_interp = (
            zhd_interp
            / SAASTAMOINEN_CNTS[0]
            * (
                1
                - SAASTAMOINEN_CNTS[1] * np.cos(2 * np.radians(lat))
                - SAASTAMOINEN_CNTS[2] * point_target_heights_gridpoint_interp
            )
        )

        # determining delta pressure between point target height and the height interpolated on grid points
        delta_p = _troposphere_barometric_formula(height) - _troposphere_barometric_formula(
            point_target_heights_gridpoint_interp
        )
        pressure_at_point_target_height = pressure_at_point_targets_interp + delta_p

        # correcting the zenith delays by height variation
        # hydrostatic delay using the Saastamoinen formula
        zenith_delay_h = (
            SAASTAMOINEN_CNTS[0]
            * pressure_at_point_target_height
            / (1 - SAASTAMOINEN_CNTS[1] * np.cos(2 * np.radians(lat)) - SAASTAMOINEN_CNTS[2] * height)
        )
        # wet delay using the exponential factor formula
        zenith_delay_w = zwd_interp * np.exp(-(height - point_target_heights_gridpoint_interp) / 2000.0)

        # compute tropospheric path delays in slant range [m]
        tropospheric_delay_hydrostatic = zenith_delay_h * mapping_factors["hydrostatic"]
        tropospheric_delay_wet = zenith_delay_w * mapping_factors["wet"]

        return tropospheric_delay_hydrostatic, tropospheric_delay_wet


# main function
def compute_delay(
    acq_time: PreciseDateTime,
    targets_xyz_coords: np.ndarray,
    sat_xyz_coords: np.ndarray,
    map_folder: Path | str | None = None,
    map_resolution: TroposphericGRIDResolution = TroposphericGRIDResolution.FINE,
    interp_method: TroposphericGridInterpolationMethod = TroposphericGridInterpolationMethod.CUBIC,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute signal delay due to tropospheric refractivity gradients.

    Parameters
    ----------
    acq_time : PreciseDateTime
        acquisition time of the scene, a.k.a. the time at which estimate the delay
    targets_xyz_coords : np.ndarray
        point targets XYZ coordinates as numpy array of shape Nx3
    sat_xyz_coords : np.ndarray
        satellite XYZ coordinates at which calibration targets are seen as numpy array of shape Nx3
    map_folder : Path | str | None, optional
        path to the folder containing the map files, by default None
    map_resolution : TroposphericGRIDResolution, optional
        map grid resolution, by default TroposphericGRIDResolution.FINE
    interp_method : TroposphericGridInterpolationMethod, optional
        method for data grid interpolation, by default TroposphericGridInterpolationMethod.CUBIC

    Returns
    -------
    np.ndarray
        array of estimated hydrostatic delay [one for each point target]
    np.ndarray
        array of estimated wet delay [one for each point target]
    """
    # instantiating class
    tropo = TroposphericDelayEstimator(
        acquisition_time=acq_time,
        map_folder=map_folder,
        interpolation_method=interp_method,
        map_grid_res=map_resolution,
    )

    return tropo.estimate_delay(point_targets_coords=targets_xyz_coords, sat_xyz_coords=sat_xyz_coords)
