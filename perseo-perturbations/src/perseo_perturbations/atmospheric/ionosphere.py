# SPDX-FileCopyrightText: Aresys S.r.l. <info@aresys.it>
# SPDX-License-Identifier: MIT

"""
Atmospheric Delay Corrections: Ionospheric submodule
----------------------------------------------------

Through the ionosphere, propagation delays are caused by dispersive effects.
"""

from __future__ import annotations

import re
import warnings
from datetime import datetime
from enum import Enum, auto
from pathlib import Path

import numpy as np
from more_itertools import pairwise
from perseo_core.geometry import compute_incidence_angles_core
from perseo_core.geometry.coordinates import (
    compute_line_ellipsoid_intersections,
    create_inflated_wgs84_ellipsoid,
    xyz2llh,
)
from perseo_core.timing import PreciseDateTime, date_to_gps_week
from scipy.interpolate import RegularGridInterpolator

from perseo_perturbations import DEFAULT_EARTH_RADIUS_M, DEFAULT_IONOSPHERE_HEIGHT_M
from perseo_perturbations.atmospheric import GPS_WEEK_REFERENCE


class IonosphericAnalysisCenters(Enum):
    """Ionospheric available analysis centers"""

    COD = auto()  # Final solution (CODE)
    COR = auto()  # Rapid solution (CODE)
    EHR = auto()  # Rapid high-rate solution, one map per hour, (ESA)
    ESA = auto()  # Final solution (ESA)
    ESR = auto()  # Rapid solution (ESA)
    IGR = auto()  # Rapid solution (IGS combined)
    IGS = auto()  # Final combined solution (IGS combined)
    JPL = auto()  # Final solution (JPL)
    UPC = auto()  # Final solution (UPC)
    UHR = auto()  # Rapid high-rate solution, one map per hour, (UPC)
    UPR = auto()  # Rapid solution (UPC)
    UQR = auto()  # Rapid high-rate solution, one map per 15 minutes, (UPC)


# custom errors
class WrongAnalysisCenterNameError(ValueError):
    """Wrong analysis center name for ionospheric data retrieval"""


class TECMapReadingError(RuntimeError):
    """Error encountered while reading the Ionospheric IONEX TEC Map file"""


class IonosphericMapFileNotFoundError(FileNotFoundError):
    """Could not find the specified ionospheric map file"""


# custom enum classes
class TECIncidenceAngleMethod(Enum):
    """Method for generating mapping function for Ionospheric delay evaluation from TEC data"""

    IPP = auto()
    GROUND = auto()
    GROUND_CONVERTED = auto()


class TECMapSolutionType(Enum):
    """Solution type of TEC maps for Ionospheric delay evaluation from TEC data"""

    FINAL = "FIN"
    RAPID = "RAP"


class TECMapTimeResolution(Enum):
    """Time resolution of TEC maps for Ionospheric delay evaluation from TEC data"""

    HALF_HOUR = "30M"
    HOUR = "01H"
    TWO_HOURS = "02H"


# custom support functions
# angle between n-dimensional vectors
def _angle_between_vectors(vector_1: np.ndarray, vector_2: np.ndarray) -> float:
    """Evaluate the angle in radians between input n-dimensional vectors.

    Parameters
    ----------
    v1 : np.ndarray
        first vector
    v2 : np.ndarray
        second vector

    Returns
    -------
    float
        angle between vectors in radians
    """

    def unit_vector(vector: np.ndarray) -> np.ndarray:
        """Compute the unit vector (unit vector) of the input vector.

        Parameters
        ----------
        vector : np.ndarray
            vector

        Returns
        -------
        np.ndarray
            unit vector
        """
        return vector / np.linalg.norm(vector)

    v1_u = unit_vector(vector_1)
    v2_u = unit_vector(vector_2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))


# defining function to properly process the timestamp info
def _epoch_timestamp_formatter(timestamp: str) -> str:
    """Formatting the epoch timestamp of the current TEC map.

    Parameters
    ----------
    timestamp : str
        string containing the timestamp

    Returns
    -------
    str
        formatted timestamp
    """

    digits = re.findall(r"[0-9]+", timestamp)
    digits = [int(d) for d in digits]
    formatted_timestamp = f"{digits[0]}-{digits[1]:02}-{digits[2]:02} {digits[3]:02}:{digits[4]:02}"

    return formatted_timestamp


# ionosphere map name generator
def generate_ionospheric_map_filename(
    acq_time: PreciseDateTime,
    center: IonosphericAnalysisCenters,
    solution_type: TECMapSolutionType = TECMapSolutionType.FINAL,
    time_resolution: TECMapTimeResolution = TECMapTimeResolution.HOUR,
) -> str:
    """Generating the map file name accordingly to the CDDIS format.

    For more information on the naming convention see also:

    https://cddis.nasa.gov/Data_and_Derived_Products/GNSS/atmospheric_products.html

    Parameters
    ----------
    acq_time : PreciseDateTime
        time of interest to retrieve the correct ionospheric data
    center : IonosphericAnalysisCenters
        analysis center name
    solution_type : IonosphericAnalysisCenters, optional
        TEC map solution type, by default TECMapSolutionType.FINAL
    time_resolution : TECMapTimeResolution, optional
        TEC map time resolution, by default TECMapTimeResolution.HOUR

    Returns
    -------
    str
        name of the ionospheric map file
    """

    gps_week, _ = date_to_gps_week(date=acq_time)

    if gps_week < GPS_WEEK_REFERENCE:
        # composing the name of the map file BEFORE GPS Week 2237
        # name format YYYY/DDD/AAAgDDD#.YYi.Z
        return center.name.lower() + f"g{acq_time.day_of_the_year:03}0.{(acq_time.year % 100):02}i"

    # format SINCE GPS Week 2238
    # name format AAA0OPSTYP_YYYYDDDHHMM_01D_SMP_CNT.INX.gz
    return (
        center.name
        + "0OPS"
        + solution_type.value
        + "_"
        + str(acq_time.year)
        + f"{acq_time.day_of_the_year:03}"
        # + f"{acq_time.hour_of_day:02}"
        # + f"{acq_time.minute_of_hour:02}"
        + "0000"
        + "_01D"
        + "_"
        + time_resolution.value
        + "_GIM"
        + ".INX"
    )


# ionospheric delay estimator from IONEX TEC MAP data
class IonosphericDelayEstimator:
    """Ionospheric Delay Estimator (first order approximation, slant range) from CDDIS IONEX TEC MAP class.
    The delay is due to the refractive index of the medium the signal is passing through.

    The ionosphere is the zone of the terrestrial atmosphere that contains a partially ionised medium, as result of the
    X and UV rays of Solar Radiation and the incidence of charged particles.

    The propagation speed of the electromagnetic signals in the ionosphere depends on its electron density:

    - during the day, sun radiation causes ionisation of neutral atoms producing free electrons and ions (TEC)
    - during the night, the recombination process prevails, where free electrons and ions are recombined, lowering TEC

    Ionosphere is a dispersive media, with the following Relation of Dispersion between $\\omega$ and $k$ of the
    incoming wave:

    $$
    \\omega^2 = c^2 \\cdot k^2 + \\omega_p^2
    $$

    where $\\omega_p$ is the critical frequency of the ionospheric plasma, meaning that only signals with
    $\\omega > \\omega_p$ through the ionized plasma medium. Therefore, ionosphere causes a frequency dependent path
    delay for microwave signals.
    """

    def __init__(
        self,
        acquisition_time: PreciseDateTime,
        analysis_center: str,
        fc_hz: float,
        ionospheric_delay_scaling_factor: float,
        tec_mapping_method: TECIncidenceAngleMethod,
        tec_solution_type: TECMapSolutionType = TECMapSolutionType.FINAL,
        tec_time_resolution: TECMapTimeResolution = TECMapTimeResolution.HOUR,
        map_folder: Path | str | None = None,
    ) -> None:
        """
        Ionosphere delay estimator class from TEC MAP file NASA CDDIS format. This class, through the method
        estimate delay, provides as output the first order approximation of the ionospheric path delay experiences by
        the electromagnetic signal of the sensor converted in slant range for each set of point target coordinates.

        Parameters
        ----------
        acquisition_time : PreciseDateTime
            scene acquisition time, the time at which the ionospheric delay must be evaluated
        analysis_center : str
            analysis center for solutions from those supported
        fc_hz : float
            carrier frequency of the sensors signal
        ionospheric_delay_scaling_factor : float, optional
            scaling factor to be applied to the ionospheric delay
        tec_mapping_method : TECIncidenceAngleMethod, optional
            selected method to compute the mapping function
        tec_solution_type : TECMapSolutionType, optional
            selected type of solution for TEC maps, by default TECMapSolutionType.FINAL
        tec_time_resolution : TECMapTimeResolution, optional
            selected type of time resolution for TEC maps, by default TECMapTimeResolution.HOUR
        map_folder : Path | str | None, optional
            path to the folder where the TEC MAP file is placed, by default None

        Raises
        ------
        WrongAnalysisCenterNameError
            if the analysis center is not among those implemented, this error is raised
        """
        self.acquisition_time = acquisition_time
        self.ionospheric_delay_scaling_factor = ionospheric_delay_scaling_factor
        self.carrier_freq = fc_hz
        self.tec_mapping_method = tec_mapping_method
        self.solution_type = tec_solution_type
        self.time_resolution = tec_time_resolution
        # updated by the one read directly from the map file, if the operation is successful
        self._ionosphere_height = DEFAULT_IONOSPHERE_HEIGHT_M
        self._earth_radius = DEFAULT_EARTH_RADIUS_M

        # analysis center
        if isinstance(analysis_center, IonosphericAnalysisCenters):
            self.analysis_center = analysis_center
        else:
            try:
                # if it's a string instead
                self.analysis_center = IonosphericAnalysisCenters[analysis_center.upper()]
            except KeyError as exc:
                raise WrongAnalysisCenterNameError(
                    f"{analysis_center} is not a supported CDDIS analysis center"
                ) from exc

        # map folder
        self.ionospheric_map_folder = None
        if isinstance(map_folder, Path):
            self.ionospheric_map_folder = map_folder
        elif isinstance(map_folder, str):
            self.ionospheric_map_folder = Path(map_folder)

    @staticmethod
    def _tec_map_parsing(content: list, exponent_factor: float) -> tuple[list, list]:
        """Parsing TEC MAP file to isolate different sections and extract data and timestamps in usable format.

        Parameters
        ----------
        content : list
            whole input file separated by lines
        exponent_factor : float
            scaling exponent factor to be applied to the extracted TEC data (the scaling factor is 10^exponent)

        Returns
        -------
        list
            list of timestamps for each tec map
        list
            list of data arrays for each tec map

        Raises
        ------
        TECMapReadingError
            error in reading the TEC map sections of the IONEX map file
        """

        # identifying the start of each tec map section
        tec_start_id = [index for index, s in enumerate(content) if "START OF TEC MAP" in s]
        # same for the end of each section
        tec_end_id = [index for index, s in enumerate(content) if "END OF TEC MAP" in s]

        # checking that start and end of section indexes are of the same length
        if not len(tec_start_id) == len(tec_end_id):
            raise TECMapReadingError("Could not isolate each TEC MAP section")

        # combining indexes to define start and end of each section
        tec_sections_boundaries = [(tec_start_id[index], tec_end_id[index]) for index in range(len(tec_start_id))]

        # isolating each tec map section
        tec_sections = [content[b[0] + 1 : b[1]] for b in tec_sections_boundaries]
        tec_timestamps = [item for section in tec_sections for item in section if "EPOCH OF CURRENT MAP" in item]
        # formatting timestamps
        tec_timestamps = list(map(_epoch_timestamp_formatter, tec_timestamps))

        tec_data = []
        for section in tec_sections:
            section_data = [line for line in section if "EPOCH OF CURRENT MAP" not in line]
            # identifying the sub-sections of lat lon h, adding the last line to have a full boundary for segmentation
            subsection_ids = [index for index, value in enumerate(section_data) if "LAT/LON1/LON2/DLON/H" in value]
            subsection_ids.append(len(section_data))
            # subsections segmentation + conversion to numpy array
            data_subsections = [
                np.fromstring("".join(section_data[indexes[0] + 1 : indexes[1]]), sep=" ")
                for indexes in pairwise(subsection_ids)
            ]
            # multiplying whole data array by scaling exponential factor
            tec_data.append(np.vstack(data_subsections) * 10**exponent_factor)

        return tec_timestamps, tec_data

    @staticmethod
    def _detect_pierce_point(
        sat_coords: np.ndarray,
        pt_coords: np.ndarray,
        earth_radius: float = DEFAULT_EARTH_RADIUS_M,
        ionosphere_height: float = DEFAULT_IONOSPHERE_HEIGHT_M,
    ) -> tuple[np.ndarray, np.ndarray, list[np.ndarray]]:
        """Detecting ionospheric pierce points (IPP) for each line of sight sensor/point-target.

        Parameters
        ----------
        sat_coords : np.ndarray
            satellite coordinates at which the point targets are seen by the sensor
        pt_coords : np.ndarray
            point target coordinates
        earth_radius : float, optional
            earth radius, by default DEFAULT_EARTH_RADIUS
        ionosphere_height : float, optional
            ionosphere height, by default DEFAULT_IONOSPHERE_HEIGHT

        Returns
        -------
        np.ndarray
            latitude coordinates of pierce points [deg]
        np.ndarray
            longitude coordinates of pierce points [deg]
        list[np.ndarray]
            list of ionospheric pierce points xyz coordinates
        """

        # defining ionosphere inflated ellipsoid
        iono_inflated_ellipsoid = create_inflated_wgs84_ellipsoid(height=ionosphere_height)

        line_of_sight = pt_coords - sat_coords

        #  finding intersection between line of sight and the inflated ellipsoid
        first_intersections, _ = compute_line_ellipsoid_intersections(
            ellipsoid=iono_inflated_ellipsoid, line_origins=sat_coords, line_directions=line_of_sight
        )

        # taking just the first intersection solutions for each point and converting to lat/lon [deg]
        ipp_llh = xyz2llh(first_intersections)
        ipp_llh[:, :2] = np.rad2deg(ipp_llh[:, :2])

        return ipp_llh[:, 0], ipp_llh[:, 1], np.vstack(first_intersections)

    @staticmethod
    def _generate_mapping_function(
        sat_coords: np.ndarray,
        method: TECIncidenceAngleMethod,
        ionosphere_height: float = DEFAULT_IONOSPHERE_HEIGHT_M,
        ipp_coords: np.ndarray = None,
        pt_coords: np.ndarray = None,
    ) -> np.ndarray:
        """Generating a mapping function based on incidence angle using the selected method. Mapping function is needed
        for conversion into slant delay using the zenith angle.

        For modelling the correction of microwave signals, ionospheric layers are condensed into a single layer
        containing the total number of electrons. The layer is assigned the altitude of the maximum electron
        concentration. This method of modelling the ionosphere follows the principles applied in GNSS, which makes it
        readily applicable to all the SAR payloads.

        Parameters
        ----------
        sat_coords : np.ndarray
            sensor coordinates in XYZ format
        method : TECIncidenceAngleMethod
            selected method to evaluate the mapping function from incidence angle
        ionosphere_height : float, optional
            ionosphere height, by default DEFAULT_IONOSPHERE_HEIGHT
        ipp_coords : np.ndarray, optional
            ionosphere pierce point coordinates in XYZ format, by default None
        pt_coords : np.ndarray, optional
            point targets coordinates in XYZ format, by default None

        Returns
        -------
        np.ndarray
            mapping function from incidence angles
        """

        if method == TECIncidenceAngleMethod.IPP:
            diff_sat_ipp = sat_coords - ipp_coords
            zenith_angles = [
                _angle_between_vectors(ipp_coords[p, :], diff_sat_ipp[p, :]) for p in range(ipp_coords.shape[0])
            ]
            mapping_function = 1 / np.cos(zenith_angles)
        elif method == TECIncidenceAngleMethod.GROUND:
            incidence_angle = compute_incidence_angles_core(sensor_positions=sat_coords, ground_points=pt_coords)
            mapping_function = 1 / np.cos(incidence_angle)
        elif method == TECIncidenceAngleMethod.GROUND_CONVERTED:
            incidence_angle = compute_incidence_angles_core(sensor_positions=sat_coords, ground_points=pt_coords)
            mapping_function = 1 / np.sqrt(
                1
                - (DEFAULT_EARTH_RADIUS_M / (DEFAULT_EARTH_RADIUS_M + ionosphere_height) * np.sin(incidence_angle)) ** 2
            )

        return mapping_function

    def read_ionosphere_map_file(self, ionosphere_map_file: Path) -> tuple[list, list, np.ndarray, np.ndarray]:
        """Read the Ionosphere IONEX map file to extract data on Total Electron Content.

        Parameters
        ----------
        ionosphere_map_file : Path
            path to the ionosphere map file, not zipped

        Returns
        -------
        list
            list of tec data arrays for each lat/lon
        list
            list of recording hours
        np.ndarray
            latitude axis (monotonically increasing)
        np.ndarray
            longitude axis (monotonically increasing)
        """

        # reading file
        if ionosphere_map_file.is_file():
            # if file is not compressed
            with open(ionosphere_map_file, "r", encoding="UTF-8") as f_in:
                # Read file removing header section
                file_content = f_in.read().splitlines()
        else:
            raise IonosphericMapFileNotFoundError(
                f"{ionosphere_map_file} file not found in specified folder {str(ionosphere_map_file.parent)}"
            )

        # extracting ionosphere height from map file
        try:
            # overwriting default value set by init
            self._ionosphere_height = [float(f.strip().split()[0]) for f in file_content if "HGT1" in f][0] * 1000
        except Exception:
            warnings.warn(
                "Error while trying to extract Ionospheric Height from map data, "
                + f"using default value {DEFAULT_IONOSPHERE_HEIGHT_M} [m]",
                stacklevel=2,
            )

        # extracting earth radius from map file
        try:
            # overwriting default value set by init
            self._earth_radius = [float(f.strip().split()[0]) for f in file_content if "BASE RADIUS" in f][0] * 1000
        except Exception:
            warnings.warn(
                "Error while trying to extract Earth Radius from map data, "
                + f"using default value {DEFAULT_EARTH_RADIUS_M} [m]",
                stacklevel=2,
            )

        # extracting data exponent factor from file
        tec_scaling_exponent = [float(f.strip().split()[0]) for f in file_content if "EXPONENT" in f][0]

        # parsing the file to isolate TEC map data
        timestamps, tec_data = self._tec_map_parsing(content=file_content, exponent_factor=tec_scaling_exponent)
        timestamps = [datetime.strptime(t, "%Y-%m-%d %H:%M") for t in timestamps]
        timestamps = [PreciseDateTime.fromisoformat(t.isoformat()) for t in timestamps]

        # Make latitude and longitude axis, must be monotonically increasing (required by interpolator)
        tec_map_lat_axis = np.arange(-87.5, (87.5 + 1), 2.5)
        tec_map_lon_axis = np.arange(-180, 180 + 1, 5)

        # changing order of rows in each tec array due to the mismatch between the latitude axis herein built and the
        # one in the loaded file
        tec_data = [np.flip(item, axis=0) for item in tec_data]

        return tec_data, timestamps, tec_map_lat_axis, tec_map_lon_axis

    def estimate_delay(self, sat_xyz_coords: np.ndarray, point_targets_coords: np.ndarray) -> np.ndarray:
        """Estimation of the ionospheric time delay as first order approximation of the ionospheric path delay in
        slant range.

        The equation used is the following (latex):

        $$
        \\Delta L = \\frac{40.3 \\cdot 10^{16}}{f_c} \\cdot MF(z) \\cdot v_{TEC}
        $$

        where f_c is the carrier frequency of the signal piercing through the ionosphere, MF(z) is the mapping function
        mapping function for conversion into slant delay using the zenith angle z, and vTEC is the vertical total
        electron content of the ionosphere in that point, in TEC units (1 TECU = 10E16 electrons per m^2)

        Parameters
        ----------
        sat_xyz_coords : np.ndarray
            satellite XYZ coordinates at which calibration targets are seen as numpy array of shape Nx3
        point_targets_coords : np.ndarray
            point targets XYZ coordinates as numpy array of shape Nx3

        Returns
        -------
        np.ndarray
            ionospheric delay
        """

        # building the ionospheric map filename
        ionospheric_map_filename = generate_ionospheric_map_filename(
            acq_time=self.acquisition_time,
            center=self.analysis_center,
            solution_type=self.solution_type,
            time_resolution=self.time_resolution,
        )

        path_to_file = self.ionospheric_map_folder.joinpath(ionospheric_map_filename)

        # reading ionospheric map data
        tec_data, timestamps, lat_axis, lon_axis = self.read_ionosphere_map_file(path_to_file)

        # find the recording timestamps closest to the acquisition time
        closest_timestamp_id = np.argmin([abs(self.acquisition_time - t) for t in timestamps])
        if timestamps[closest_timestamp_id] - self.acquisition_time > 0:
            # if the acquisition time is above the half of the hour, i.e. 11:37:00.00, the closest timestamp
            # of the map file would be the next hour. Subtracting 1 to avoid this issue
            closest_timestamp_id -= 1
        # taking the closest value and the next one for interpolation purposes
        timestamps_selected = timestamps[closest_timestamp_id : closest_timestamp_id + 2]
        tec_data_selected = tec_data[closest_timestamp_id : closest_timestamp_id + 2]

        # find ionospheric pierce point
        ipp_latitude_deg, ipp_longitude_deg, ipp_xyz = self._detect_pierce_point(
            sat_coords=sat_xyz_coords,
            pt_coords=point_targets_coords,
            earth_radius=self._earth_radius,
            ionosphere_height=self._ionosphere_height,
        )

        # calculating longitude accounting for Earth rotation
        # NOTE this will probably give wrong results for points close to either +-180 longitudes
        time_deltas = np.array([(t - self.acquisition_time) / 3600 for t in timestamps_selected])
        ipp_long_selected = [ipp_longitude_deg + 360.0 / 24.0 * t for t in time_deltas]

        # tec grid interpolation
        interpolating_functions = [RegularGridInterpolator((lat_axis, lon_axis), tec) for tec in tec_data_selected]
        # bilinear interpolation over latitude and longitude
        interpolated_values1 = interpolating_functions[0]((ipp_latitude_deg, ipp_long_selected[0]))
        interpolated_values2 = interpolating_functions[1]((ipp_latitude_deg, ipp_long_selected[1]))

        # linear interpolation in time
        t_diff = time_deltas[1] - time_deltas[0]
        tec_interpolated = [
            np.abs(time_deltas[1]) / t_diff * interpolated_values1,
            np.abs(time_deltas[0]) / t_diff * interpolated_values2,
        ]
        tec_interpolated = np.add(tec_interpolated[0], tec_interpolated[1])

        # generating mapping function
        mapping_function = self._generate_mapping_function(
            sat_coords=sat_xyz_coords,
            ipp_coords=ipp_xyz,
            method=self.tec_mapping_method,
            ionosphere_height=self._ionosphere_height,
            pt_coords=point_targets_coords,
        )

        # computing the ionospheric delay
        # first order approximation of the ionospheric path delay in slant range
        ionospheric_delay = (
            (40.3 * 1e16 / self.carrier_freq**2)
            * tec_interpolated
            * mapping_function
            * self.ionospheric_delay_scaling_factor
        )

        return ionospheric_delay


# main function
def compute_delay(
    acq_time: PreciseDateTime,
    targets_xyz_coords: np.ndarray,
    sat_xyz_coords: np.ndarray,
    analysis_center: str | IonosphericAnalysisCenters,
    fc_hz: float,
    map_folder: Path | str | None = None,
    delay_scaling_factor: float = 1.0,
    tec_mapping_method: TECIncidenceAngleMethod = TECIncidenceAngleMethod.GROUND_CONVERTED,
    tec_solution_type: TECMapSolutionType = TECMapSolutionType.FINAL,
    tec_time_resolution: TECMapTimeResolution = TECMapTimeResolution.HOUR,
) -> np.ndarray:
    """Compute ionospheric signal delay from date and coordinates (sensor and ground).

    Parameters
    ----------
    acq_time : PreciseDateTime
        scene acquisition time, the time at which the ionospheric delay must be evaluated
    targets_xyz_coords : np.ndarray
        point targets XYZ coordinates as numpy array of shape Nx3
    sat_xyz_coords : np.ndarray
        satellite XYZ coordinates at which calibration targets are seen as numpy array of shape Nx3
    analysis_center : str | IonosphericAnalysisCenters]
        analysis center for solutions from those supported
    fc_hz : float
        carrier frequency of the sensor's signal
    map_folder : Path | str | None, optional
        path to the folder where the TEC MAP file is placed, by default None
    delay_scaling_factor : float, optional
        scaling factor to be applied to the ionospheric delay, by default 1.0
    tec_mapping_method : TECIncidenceAngleMethod, optional
        selected method to compute the mapping function, by default TECIncidenceAngleMethod.GROUND_CONVERTED
    tec_solution_type : TECMapSolutionType, optional
        selected type of solution for TEC maps, by default TECMapSolutionType.FINAL
    tec_time_resolution : TECMapTimeResolution, optional
        selected type of time resolution for TEC maps, by default TECMapTimeResolution.HOUR

    Returns
    -------
    np.ndarray
        ionospheric delay
    """

    # instantiating class
    ion = IonosphericDelayEstimator(
        acquisition_time=acq_time,
        analysis_center=analysis_center,
        fc_hz=fc_hz,
        map_folder=map_folder,
        ionospheric_delay_scaling_factor=delay_scaling_factor,
        tec_mapping_method=tec_mapping_method,
        tec_solution_type=tec_solution_type,
        tec_time_resolution=tec_time_resolution,
    )

    return ion.estimate_delay(point_targets_coords=targets_xyz_coords, sat_xyz_coords=sat_xyz_coords)
