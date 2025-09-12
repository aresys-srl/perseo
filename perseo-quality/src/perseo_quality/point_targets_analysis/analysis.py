# SPDX-FileCopyrightText: Aresys S.r.l. <info@aresys.it>
# SPDX-License-Identifier: MIT

"""Point Target Analysis: IRF, RCS and Localization Errors"""

from __future__ import annotations

import contextlib
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from arepytools.geometry.inverse_geocoding_core import inverse_geocoding_monostatic_core
from arepytools.timing.precisedatetime import PreciseDateTime

import perseo_quality.core.custom_errors as c_err
import perseo_quality.core.generic_dataclasses as gdt
import perseo_quality.core.signal_processing as sp
import perseo_quality.point_targets_analysis.custom_dataclasses as ptdt
from perseo_quality.core.common import check_targets_visibility
from perseo_quality.logger import quality_logger as log
from perseo_quality.point_targets_analysis.config import PointTargetAnalysisConfig, RCSParameters
from perseo_quality.point_targets_analysis.core.irf import compute_point_target_irf_analysis
from perseo_quality.point_targets_analysis.core.localization_error import compute_localization_errors_pixels
from perseo_quality.point_targets_analysis.core.pre_processing import (
    compute_data_resolution_pixel,
    compute_roi,
    detect_data_type,
    extract_target_area,
    generate_irf_axis,
    target_area_interpolation,
)
from perseo_quality.point_targets_analysis.core.rcs import compute_additional_rcs_values, compute_point_target_rcs
from perseo_quality.point_targets_analysis.custom_errors import PointTargetComputationError
from perseo_quality.point_targets_analysis.output_reference import (
    PROD_INFO_COLS,
    PTA_OUTPUT_COLUMNS_DF_UM,
    REMOVED_OUTPUT_COLS,
    add_unit_of_measure_to_df_columns,
)
from perseo_quality.point_targets_analysis.support import SideLobesDirections, compute_side_lobes_directions

if TYPE_CHECKING:
    from arepytools.io.io_support import NominalPointTarget

    from perseo_quality.io.quality_input_protocol import ChannelData, QualityInputProduct


def point_target_analysis(
    product: QualityInputProduct,
    point_targets: dict[str, NominalPointTarget],
    config: PointTargetAnalysisConfig | None = None,
) -> tuple[pd.DataFrame, list[ptdt.PointTargetGraphicalData]]:
    """Compute the full point target analysis: IRF, RCS and localization errors.

    Parameters
    ----------
    product : QualityInputProduct
        object satisfying the QualityInputProduct protocol
    point_targets : dict[str, NominalPointTarget]
        dictionary of point targets locations, with keys being the target id label and value a NominalPointTarget
        dataclass instance with point target location data
    config : PointTargetAnalysisConfig, optional
        config file PointTargetAnalysisConfig dataclass to enable and manage different features, if provided,
        by default None

    Returns
    -------
    pd.DataFrame
        pandas dataframe containing all the computed features for each point target
    list[ptdt.PointTargetGraphicalData]
        list of PointTargetGraphicalData data output for plotting graphs

    Raises
    ------
    c_err.SideLobesDirectionsEstimationError
        could not compute side lobes directions
    """
    # defining default config if None is given
    if config is None:
        log.info("Configuration file not provided, using default")
        config = PointTargetAnalysisConfig()

    log.info(f"Starting Point Target Analysis on {product.name}")
    log.info(f"Selected Product has {len(product.channels_list)} channels")

    # check which target are inside the scene
    if config.check_targets_in_scene:
        log.info("Checking which targets are visible in the scene")
        visible_targets = check_targets_visibility(product, point_targets)
        not_visible_targets = visible_targets.query("burst == None")["id"]
        visible_targets = visible_targets.query("burst.notnull()", engine="python")
        if not not_visible_targets.empty:
            log.info(f"Point Targets {not_visible_targets} are not visible in the scene.")

    # initializing lists for storing data
    graphs = []
    res = []
    for channel_id in visible_targets["channel"].unique():
        swath = visible_targets[visible_targets["channel"] == channel_id]["swath"].iloc[0]
        polar = visible_targets[visible_targets["channel"] == channel_id]["polarization"].iloc[0].upper()
        log.info(f"Analyzing Channel {channel_id}, Swath {swath}, Polarization {polar}...")

        # recovering metadata for the current channel
        channel_data = product.get_channel_data(channel_id=channel_id)

        # recovering only targets visible by this channel
        targets_visible_by_channel = visible_targets[visible_targets["channel"] == channel_id]["id"]
        num_visible_targets = len(targets_visible_by_channel)

        for target_count, target_id in enumerate(targets_visible_by_channel):
            bursts_selection = visible_targets.query("channel == @channel_id & id == @target_id")
            bursts_selection = bursts_selection.loc[:, "burst"].to_list()[0]

            for burst in bursts_selection:
                log.info(
                    f"Processing Target Point {target_id} ({target_count + 1}/{num_visible_targets}), Burst #{burst}",
                )
                output, graph = point_target_analysis_single(
                    channel_data=channel_data,
                    burst=burst,
                    point_target=point_targets[target_id],
                    config=config,
                )

                output.product_name = product.name
                output.target = target_id
                graph.target = target_id

                res.append(output)
                graphs.append(graph)

    if res:
        results_df = _results_to_dataframe(res)
        results_df.sort_values(by=["target", "polarization"], inplace=True)
    else:
        log.critical("Provided Point Targets are not visible in the scene. Analysis could not be performed.")
        results_df = pd.DataFrame(columns=["error_point_target_not_in_scene"])
        graphs = [ptdt.PointTargetGraphicalData()]

    return results_df, graphs


def point_target_analysis_single(
    channel_data: ChannelData,
    burst: int,
    point_target: NominalPointTarget,
    config: PointTargetAnalysisConfig,
) -> tuple[ptdt.PointTargetAnalysisOutput, ptdt.PointTargetGraphicalData]:
    """Analysis of a single point target.

    Parameters
    ----------
    channel_data : ChannelData
        current channel data
    burst : int
        current burst id
    point_target : NominalPointTarget
        current point target to be analyzed
    config : PointTargetAnalysisConfig
        point target analysis configuration

    Returns
    -------
    ptdt.PointTargetAnalysisOutput
        quantitative analysis results
    ptdt.PointTargetGraphicalData
        additional data for graphical output generation

    Raises
    ------
    c_err.SideLobesDirectionsEstimationError
        if there is an error in computing the side lobes direction
    """
    output = ptdt.PointTargetAnalysisOutput(
        channel=str(channel_data.channel_id),
        sensor_name=channel_data.sensor_name,
        info=ptdt.GenericInfoOutput(
            swath=channel_data.swath_name,
            burst=burst,
            product_type=channel_data.image_type.name.lower(),
            polarization=channel_data.polarization.value,
            acquisition_mode=channel_data.acquisition_mode.name,
        ),
    )
    graph = ptdt.PointTargetGraphicalData(
        channel=channel_data.channel_id,
        swath=channel_data.swath_name,
        burst=burst,
        polarization=channel_data.polarization,
    )

    try:
        az_rng_coords = compute_sar_coordinates(target=point_target, channel_data=channel_data, burst=burst)
    except Exception:
        return output, graph

    location_data: gdt.LocationData = channel_data.get_location_data(
        azimuth_time=az_rng_coords.azimuth,
        range_time=az_rng_coords.range,
    )

    output.additional_info = ptdt.PTAdditionalInfo(
        orbit_direction=channel_data.orbit_direction.name,
        look_angle=np.rad2deg(location_data.look_angle),
        ground_velocity=location_data.ground_velocity,
    )

    # ale limits conversion from meters to pixels
    ale_pixel_limits = config.ale_limits
    if ale_pixel_limits is not None:
        ale_pixel_limits = (
            np.max([np.ceil(ale_pixel_limits[0] / channel_data.range_step_m * 2), 3]),
            np.max([np.ceil(ale_pixel_limits[1] / location_data.azimuth_step_m * 2), 3]),
        )

    # extracting cropped target area centered on peak coordinates
    target_area, peak_coords, nominal_coords, peak_coords_swath = extract_target_area(
        channel_data=channel_data,
        azimuth_range_coordinates=az_rng_coords,
        ale_limits=ale_pixel_limits,
        initial_crop=config.irf_parameters.peak_finding_roi_size,
        final_crop=config.irf_parameters.analysis_roi_size,
    )

    if target_area is None or peak_coords is None or nominal_coords is None or peak_coords_swath is None:
        return output, graph

    az_time_peak, rng_time_peak = channel_data.pixel_to_times_conversion(
        azimuth_index=peak_coords_swath[1],
        range_index=peak_coords_swath[0],
        burst=burst,
    )

    try:
        side_lobes_directions, squint_angle, doppler_centroid = compute_side_lobes_directions(
            channel_data=channel_data,
            peak_azimuth_time=az_time_peak,
            peak_range_time=rng_time_peak,
            azimuth_step_m=location_data.azimuth_step_m,
        )
    except Exception as err:
        msg = "Could not evaluate side lobes directions"
        raise c_err.SideLobesDirectionsEstimationError(msg) from err

    log.info(f"Measured squint angle: {np.round(np.rad2deg(squint_angle), 4)} \u00b0")

    if abs(np.rad2deg(squint_angle)) <= config.irf_parameters.zero_doppler_abs_squint_threshold_deg:
        log.info(
            f"Measured squint angle is below threshold {config.irf_parameters.zero_doppler_abs_squint_threshold_deg}",
        )
        log.info("Assuming zero doppler")
        side_lobes_directions = (np.inf, 0.0)

    # updating the azimuth and range steps for localization purposes due to side lobes directions
    original_rng_step = location_data.range_step_m
    original_az_step = location_data.azimuth_step_m
    location_data.azimuth_step_m, location_data.range_step_m = update_steps(
        azimuth_step=location_data.azimuth_step_m,
        range_step=location_data.range_step_m,
        side_lobes_directions=side_lobes_directions,
        target_area_shape=target_area.shape,
    )

    try:
        output.irf, output.rcs, graph.irf, graph.rcs = point_target_analysis_core_computation(
            target_area=target_area,
            location_data=location_data,
            target_pos_nominal=nominal_coords,
            target_pos_real=peak_coords,
            sensor_position_at_target=channel_data.trajectory.evaluate(az_rng_coords.azimuth),
            side_lobes_directions=side_lobes_directions,
            target_info=point_target,
            original_range_step_m=original_rng_step,
            original_azimuth_step_m=original_az_step,
            carrier_frequency=channel_data.carrier_frequency,
            polarization=channel_data.polarization,
            projection=channel_data.projection,
            config=config,
        )
    except PointTargetComputationError:
        graph.irf = ptdt.IRFGraphDataOutput()
        graph.rcs = ptdt.RCSGraphDataOutput()

    # Compute additional information for output results

    output.additional_info.doppler_rate_real = (
        channel_data.doppler_rate.evaluate(azimuth_time=az_time_peak, range_time=rng_time_peak)
        if channel_data.doppler_rate is not None
        else None
    )
    output.additional_info.doppler_rate_theoretical = float(
        sp.compute_doppler_rate_theoretical(
            trajectory=channel_data.trajectory,
            azimuth_time=az_time_peak,
            coords=point_target.xyz_coordinates,
            fc_hz=channel_data.carrier_frequency,
        ),
    )

    mid_burst_az, _ = channel_data.get_mid_burst_times(burst=burst)
    az_steering_rate = channel_data.get_steering_rate(azimuth_time=az_time_peak, burst=burst)
    steering_doppler_freq = 0
    if (
        mid_burst_az is not None
        and az_steering_rate is not None
        and output.additional_info.doppler_rate_real is not None
    ):
        steering_doppler_freq = sp.compute_steering_doppler_frequency(
            trajectory=channel_data.trajectory,
            azimuth_time=az_time_peak,
            az_mid_burst_time=mid_burst_az,
            az_steering_rate_rad_s=az_steering_rate,
            doppler_rate=output.additional_info.doppler_rate_real,
            fc_hz=channel_data.carrier_frequency,
        )

    output.additional_info.doppler_frequency = doppler_centroid
    output.additional_info.peak_azimuth_time = az_time_peak
    output.additional_info.peak_range_time = rng_time_peak
    output.additional_info.peak_azimuth_from_burst_start = peak_coords_swath[1] - sum(
        channel_data.lines_per_burst[:burst],
    )
    output.additional_info.steering_doppler_frequency = steering_doppler_freq
    output.info.squint_angle = np.round(squint_angle, 6)
    output.info.range_position = peak_coords_swath[0]
    output.info.azimuth_position = peak_coords_swath[1]
    output.info.incidence_angle = np.rad2deg(location_data.incidence_angle)

    return output, graph


def compute_sar_coordinates(target: NominalPointTarget, channel_data: ChannelData, burst: int) -> gdt.SARCoordinates:
    """Extract azimuth and range coordinates.

    Parameters
    ----------
    target : NominalPointTarget
        current point target nominal data
    channel_data : ChannelData
        current channel data
    burst : int
        burst id

    Returns
    -------
    gdt.SARCoordinates
        SAR coordinates of the current point target as seen by the SAR product
    """
    az_time, rg_time = inverse_geocoding_monostatic_core(
        trajectory=channel_data.trajectory,
        ground_points=target.xyz_coordinates,
        initial_guesses=channel_data.mid_azimuth_time,
        frequencies_doppler_centroid=0,
        wavelength=1,
    )
    assert isinstance(az_time, PreciseDateTime)
    assert isinstance(rg_time, float)

    if target.delay is not None:
        rg_time += target.delay

    az_index, rg_index = channel_data.times_to_pixel_conversion(azimuth_time=az_time, range_time=rg_time, burst=burst)

    return gdt.SARCoordinates(
        azimuth=az_time,
        range=rg_time,
        azimuth_index_subpx=az_index,
        range_index_subpx=rg_index,
        burst=burst,
    )


def update_steps(
    azimuth_step: float,
    range_step: float,
    side_lobes_directions: SideLobesDirections,
    target_area_shape: tuple[int, ...],
) -> tuple[float, float]:
    """Update the azimuth and range steps for localization purposes due to side lobes directions.

    Parameters
    ----------
    azimuth_step : float
        original azimuth step
    range_step : float
        original range step
    side_lobes_directions : SideLobesDirections
        range and azimuth cuts angular coefficients in samples
    target_area_shape : tuple[int, ...]
        target area shape

    Returns
    -------
    float
        updated azimuth step
    float
        updated range step
    """
    original_range_step = range_step
    original_azimuth_step = azimuth_step

    az_side_lobe, rg_side_lobe = side_lobes_directions
    if not np.isinf(az_side_lobe):
        aspect_ratio = target_area_shape[1] / target_area_shape[0]

        if np.abs(az_side_lobe * aspect_ratio) > 1:
            range_step = np.sqrt(original_range_step**2 + (original_azimuth_step / az_side_lobe) ** 2)
        else:
            range_step = np.sqrt((az_side_lobe * original_range_step) ** 2 + original_azimuth_step**2)

        if np.abs(rg_side_lobe * aspect_ratio) > 1:
            azimuth_step = np.sqrt(original_range_step**2 + (original_azimuth_step / rg_side_lobe) ** 2)
        else:
            azimuth_step = np.sqrt((rg_side_lobe * original_range_step) ** 2 + original_azimuth_step**2)

    return azimuth_step, range_step


def _results_to_dataframe(results: list[ptdt.PointTargetAnalysisOutput]) -> pd.DataFrame:
    """Organizing results dataclass into a single pandas dataframe for easy exporting.

    Parameters
    ----------
    results : list[dtc.PointTargetAnalysisOutput]
        list of PointTargetAnalysisOutput dataclass with stored results

    Returns
    -------
    pd.DataFrame
        pandas dataframe containing all the results organized
    """
    # extracting dataframes
    info_df = pd.DataFrame([r.info for r in results])
    additional_info_df = pd.DataFrame([r.additional_info for r in results])
    irf_df = pd.DataFrame([r.irf for r in results])
    rcs_df = pd.DataFrame([r.rcs for r in results])
    ch_trgt_df = pd.DataFrame(
        [(r.target, r.channel, r.product_name, r.sensor_name) for r in results],
        columns=PROD_INFO_COLS,
    )

    # merging dataframe horizontally
    df_res = pd.concat([ch_trgt_df, info_df, additional_info_df, irf_df, rcs_df], axis=1)
    df_res.drop(REMOVED_OUTPUT_COLS, axis=1, inplace=True)

    # adding unit of measure to column names
    new_col = add_unit_of_measure_to_df_columns(columns=list(df_res.columns))
    assert new_col == PTA_OUTPUT_COLUMNS_DF_UM
    df_res.columns = new_col
    with contextlib.suppress(ValueError):
        df_res["target"] = pd.to_numeric(df_res["target"])

    return df_res


def point_target_analysis_core_computation(
    target_area: np.ndarray,
    location_data: gdt.LocationData,
    target_pos_real: np.ndarray,
    target_pos_nominal: np.ndarray,
    sensor_position_at_target: np.ndarray,
    side_lobes_directions: SideLobesDirections,
    projection: gdt.SARProjection,
    polarization: gdt.SARPolarization,
    carrier_frequency: float,
    original_range_step_m: float,
    original_azimuth_step_m: float,
    target_info: NominalPointTarget,
    config: PointTargetAnalysisConfig,
) -> tuple[ptdt.IRFDataOutput, ptdt.RCSDataOutput, ptdt.IRFGraphDataOutput, ptdt.RCSGraphDataOutput]:
    """Perform the core computation of the point target analysis.

    Parameters
    ----------
    target_area : np.ndarray
        target area to be analyzed
    location_data : gdt.LocationData
        location data
    target_pos_real : np.ndarray
        real peak position of the target as detected by the SAR product
    target_pos_nominal : np.ndarray
        nominal target position
    sensor_position_at_target : np.ndarray
        sensor position in orbit at target location
    side_lobes_directions : SideLobesDirections
        range and azimuth cuts angular coefficients in samples
    projection : gdt.SARProjection
        SAR product projection
    polarization : gdt.SARPolarization
        SAR product polarization
    carrier_frequency : float
        signal carrier frequency
    original_range_step_m : float
        original range step in meters
    original_azimuth_step_m : float
        original azimuth step in meters
    target_info : NominalPointTarget
        current target to be analyzed
    config : PointTargetAnalysisConfig
        point target analysis configuration

    Returns
    -------
    ptdt.IRFDataOutput
        IRF analysis results
    ptdt.RCSDataOutput
        RCS analysis results
    ptdt.IRFGraphDataOutput
        data for IRF graphical output
    ptdt.RCSGraphDataOutput
        data for RCS graphical output
    """
    assert len(target_area.shape) == 2

    data_type = detect_data_type(target_area=target_area)

    # Cropping and oversampling
    crop_size = compute_roi(data_shape=target_area.shape, oversampling_factor=config.irf_parameters.oversampling_factor)
    recentered_target_area_interp = target_area_interpolation(
        target_area=target_area,
        target_pos_real=target_pos_real,
        roi=crop_size,
        oversampling_factor=config.irf_parameters.oversampling_factor,
    )
    ovs_range_step_m = location_data.range_step_m / config.irf_parameters.oversampling_factor
    ovs_ground_range_step_m = location_data.ground_range_step_m / config.irf_parameters.oversampling_factor
    ovs_azimuth_step_m = location_data.azimuth_step_m / config.irf_parameters.oversampling_factor

    log.debug("Computing IRF resolution...")
    graph_irf = irf_analysis_profiles(
        recentered_target_area_interp=recentered_target_area_interp,
        data_type=data_type,
        range_step_m=ovs_range_step_m,
        azimuth_step_m=ovs_azimuth_step_m,
        side_lobes_directions=side_lobes_directions,
    )
    assert graph_irf.rng_resolution is not None and graph_irf.az_resolution is not None

    if config.perform_irf:
        log.debug("Performing IRF analysis...")
        data_type = detect_data_type(target_area=target_area)
        if data_type == gdt.TargetDataType.DETECTED:
            recentered_target_area_interp = np.sqrt(recentered_target_area_interp)
        irf_output = compute_point_target_irf_analysis(
            recentered_target_area_interp=recentered_target_area_interp,
            range_resolution_px=graph_irf.rng_resolution,
            azimuth_resolution_px=graph_irf.az_resolution,
            side_lobes_directions=side_lobes_directions,
            mask_method=config.irf_parameters.masking_method,
            pslr_flag=config.evaluate_pslr,
            sslr_flag=config.evaluate_sslr,
            islr_flag=config.evaluate_islr,
        )

        irf_res = ptdt.IRFDataOutput(
            range_resolution=(
                graph_irf.rng_resolution * ovs_range_step_m
                if projection == gdt.SARProjection.SLANT_RANGE
                else graph_irf.rng_resolution * ovs_ground_range_step_m
            ),
            azimuth_resolution=graph_irf.az_resolution * ovs_azimuth_step_m,
            ground_range_resolution=graph_irf.rng_resolution * ovs_ground_range_step_m,
            azimuth_pslr=irf_output.azimuth_pslr,
            range_pslr=irf_output.range_pslr,
            pslr_2d=irf_output.pslr_2d,
            azimuth_sslr=irf_output.azimuth_sslr,
            range_sslr=irf_output.range_sslr,
            sslr_2d=irf_output.sslr_2d,
            azimuth_islr=irf_output.azimuth_islr,
            range_islr=irf_output.range_islr,
            islr_2d=irf_output.islr_2d,
        )

    else:
        log.warning("IRF analysis has been disabled in configuration file.")
        irf_res = ptdt.IRFDataOutput()

    if config.evaluate_localization:
        log.debug("Computing Localization errors...")
        slant_range_error, azimuth_error, ground_range_error = compute_localization_errors_pixels(
            target_pos_real=target_pos_real,
            target_pos_ref=target_pos_nominal,
        )
        # saving Localization Errors for numerical output
        irf_res.slant_range_localization_error = -original_range_step_m * slant_range_error
        irf_res.ground_range_localization_error = -location_data.ground_range_step_m * ground_range_error
        irf_res.azimuth_localization_error = -original_azimuth_step_m * azimuth_error
    else:
        log.warning("Localization Errors computation has been disabled in configuration file.")

    if config.perform_rcs:
        log.debug("Performing RCS analysis...")
        range_resolution_px = graph_irf.rng_resolution / config.irf_parameters.oversampling_factor
        azimuth_resolution_px = graph_irf.az_resolution / config.irf_parameters.oversampling_factor
        # correcting resolution taking into account the change in steps due to side lobes directions
        if not np.isinf(side_lobes_directions[0]):
            range_resolution_px *= location_data.range_step_m / original_range_step_m
            azimuth_resolution_px *= location_data.azimuth_step_m / original_azimuth_step_m

        step_distances = [location_data.range_step_m, location_data.azimuth_step_m]

        rcs_res, graph_rcs = rcs_analysis(
            target_area=target_area,
            target_pos_real=target_pos_real,
            rcs_parameters=config.rcs_parameters,
            polarization=polarization,
            target_info=target_info,
            sensor_position_at_target=sensor_position_at_target,
            carrier_frequency=carrier_frequency,
            range_resolution_px=range_resolution_px,
            azimuth_resolution_px=azimuth_resolution_px,
            step_distances=step_distances,
        )
    else:
        log.warning("RCS analysis has been disabled in configuration file.")
        rcs_res = ptdt.RCSDataOutput()
        graph_rcs = ptdt.RCSGraphDataOutput()

    # Convert pixel related quantities to original sampling
    graph_irf.rng_resolution = graph_irf.rng_resolution / config.irf_parameters.oversampling_factor
    graph_irf.az_resolution = graph_irf.az_resolution / config.irf_parameters.oversampling_factor
    graph_irf.rng_axis = graph_irf.rng_axis / config.irf_parameters.oversampling_factor
    graph_irf.az_axis = graph_irf.az_axis / config.irf_parameters.oversampling_factor
    graph_irf.rng_step_distance = original_range_step_m
    graph_irf.az_step_distance = original_azimuth_step_m

    return irf_res, rcs_res, graph_irf, graph_rcs


def irf_analysis_profiles(
    recentered_target_area_interp: np.ndarray,
    data_type: gdt.TargetDataType,
    range_step_m: float,
    azimuth_step_m: float,
    side_lobes_directions: SideLobesDirections,
) -> ptdt.IRFGraphDataOutput:
    """Compute the IRF analysis profiles.

    Parameters
    ----------
    recentered_target_area_interp : np.ndarray
        target area recentered on signal peak position
    data_type : gdt.TargetDataType
        data type
    range_step_m : float
        range step in meters
    azimuth_step_m : float
        azimuth step in meters
    side_lobes_directions : SideLobesDirections
        range and azimuth cuts angular coefficients in samples

    Returns
    -------
    ptdt.IRFGraphDataOutput
        data for IRF graphical output

    Raises
    ------
    PointTargetComputationError
        if IRF resolution could not be computed
    """
    range_profile, azimuth_profile, range_resolution_px, azimuth_resolution_px = compute_data_resolution_pixel(
        recentered_target_area_interp=recentered_target_area_interp,
        data_type=data_type,
        side_lobes_directions=side_lobes_directions,
    )
    if np.isnan(range_resolution_px) or np.isnan(azimuth_resolution_px):
        log.warning("IRF Resolution couldn't be properly assessed.")
        raise PointTargetComputationError

    graph = ptdt.IRFGraphDataOutput(
        rng_step_distance=range_step_m,
        az_step_distance=azimuth_step_m,
        side_lobes_directions=side_lobes_directions,
        rng_resolution=range_resolution_px,
        az_resolution=azimuth_resolution_px,
        image=recentered_target_area_interp,
        rng_axis=generate_irf_axis(
            stop=recentered_target_area_interp.shape[0],
            offset=recentered_target_area_interp.shape[0] // 2,
            scaling=1,
        ),
        az_axis=generate_irf_axis(
            stop=recentered_target_area_interp.shape[1],
            offset=recentered_target_area_interp.shape[1] // 2,
            scaling=1,
        ),
        rng_profile=range_profile,
        az_profile=azimuth_profile,
    )
    return graph


def rcs_analysis(
    target_area: np.ndarray,
    target_pos_real: np.ndarray,
    rcs_parameters: RCSParameters,
    polarization: gdt.SARPolarization,
    target_info: NominalPointTarget,
    sensor_position_at_target: np.ndarray,
    carrier_frequency: float,
    range_resolution_px: float,
    azimuth_resolution_px: float,
    step_distances: list[float],
) -> tuple[ptdt.RCSDataOutput, ptdt.RCSGraphDataOutput]:
    """Perform the RCS analysis.

    Parameters
    ----------
    target_area : np.ndarray
        target area to be analyzed
    target_pos_real : np.ndarray
        real peak position of the target as detected by the SAR product
    rcs_parameters : RCSParameters
        rcs parameters
    polarization : gdt.SARPolarization
        SAR product polarization
    target_info : NominalPointTarget
        current target to be analyzed
    sensor_position_at_target : np.ndarray
        sensor position in orbit at target location
    carrier_frequency : float
        signal carrier frequency
    range_resolution_px : float
        range resolution in pixels
    azimuth_resolution_px : float
        azimuth resolution in pixels
    step_distances : list[float]
        step distances

    Returns
    -------
    ptdt.RCSDataOutput
        RCS analysis results
    ptdt.RCSGraphDataOutput
        data for RCS graphical output
    """
    data_type = detect_data_type(target_area=target_area)
    assert len(target_area.shape) == 2
    rcs_roi = compute_roi(
        data_shape=target_area.shape,
        oversampling_factor=rcs_parameters.roi_dimension / 2,
    ).astype("int")

    rcs_output_parameters, roi_target, roi_background_corners, peak_corners = compute_point_target_rcs(
        target_area=target_area,
        target_pos_real=target_pos_real,
        range_resolution_px=range_resolution_px,
        azimuth_resolution_px=azimuth_resolution_px,
        rcs_interp_factor=rcs_parameters.interpolation_factor,
        rcs_roi=rcs_roi,
        k_lin=rcs_parameters.calibration_factor,
        s_f=rcs_parameters.resampling_factor,
    )

    # rescaling rcs values, computing peak phase error
    rcs, rcs_db, rcs_error, peak_phase_error = compute_additional_rcs_values(
        rcs_input=rcs_output_parameters,
        step_distances=step_distances,
        interp_factor=rcs_parameters.interpolation_factor,
        polarization=polarization,
        target_info=target_info,
        sensor_position=sensor_position_at_target,
        fc_hz=carrier_frequency,
    )

    graph = ptdt.RCSGraphDataOutput(
        image=roi_target,
        data_type=data_type,
        roi_size=rcs_roi.copy(),
        roi_background=roi_background_corners,
        roi_peak=peak_corners,
        interp_factor=rcs_parameters.interpolation_factor,
        rng_step_distance=step_distances[0],
        az_step_distance=step_distances[1],
        rcs_lin=rcs,
        rcs_db=rcs_db,
    )

    results = ptdt.RCSDataOutput(
        clutter=rcs_output_parameters.clutter,
        scr=rcs_output_parameters.scr,
        peak_value_complex=rcs_output_parameters.peak_value_complex,
        rcs=rcs_db,
        rcs_error=rcs_error,
        peak_phase_error=peak_phase_error,
    )
    return results, graph
