# SPDX-FileCopyrightText: Aresys S.r.l. <info@aresys.it>
# SPDX-License-Identifier: MIT

"""Creating a map of IGRF magnetic field."""

from datetime import datetime
from pathlib import Path

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cartopy.mpl.ticker as cticker
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import TwoSlopeNorm

from perseo_perturbations.geomagnetic import get_geodetic_igrf, get_magnetic_declination, get_magnetic_inclination


def create_lat_lon_grid(lon_step_deg: float, lat_step_deg: float, height_km: float):

    # lat/lon grid
    lat_deg = np.arange(-90, 90 + lat_step_deg, lat_step_deg)
    lon_deg = np.arange(-180, 180 + lon_step_deg, lon_step_deg)

    n_lat = len(lat_deg)
    n_lon = len(lon_deg)

    lon_grid, lat_grid, height_grid = np.meshgrid(lon_deg, lat_deg, np.array([height_km]), indexing="xy")

    # flatten to (N,3)
    return np.column_stack([lon_grid.ravel(), lat_grid.ravel(), height_grid.ravel()]), n_lon, n_lat


def field_computation_by_year(date: datetime, lon_lat_grid: np.ndarray, n_lon: int, n_lat: int):

    # get geodetic IGRF
    geodetic_igrf = get_geodetic_igrf(coordinates=lon_lat_grid, date=date)

    # geodetic_igrf Be[:, 0], Bn[:, 1], Bu[:, 2]
    magnetic_field_intensity = np.sqrt(geodetic_igrf[:, 0] ** 2 + geodetic_igrf[:, 1] ** 2 + geodetic_igrf[:, 2] ** 2)
    inclination = get_magnetic_inclination(geodetic_magnetic_field=geodetic_igrf, degrees=True)
    declination = get_magnetic_declination(geodetic_magnetic_field=geodetic_igrf, degrees=True)

    magnetic_field_intensity = magnetic_field_intensity.reshape(n_lat, n_lon)
    inclination = inclination.reshape(n_lat, n_lon)
    declination = declination.reshape(n_lat, n_lon)

    return {
        "year": date.year,
        "magnetic_field_intensity": magnetic_field_intensity,
        "inclination": inclination,
        "declination": declination,
    }


def plot_igrf_map(igrf_results: list[dict[str]], field: str, output_dir: Path, relative: bool) -> None:

    vmin = np.nanmin([np.nanmin(res[field]) for res in igrf_results])
    vmax = np.nanmax([np.nanmax(res[field]) for res in igrf_results])
    if relative:
        base = igrf_results[0][field]
        base_year = igrf_results[0]["year"]
        vmin = np.nanmin([np.nanpercentile(res[field] - base, 1) for res in igrf_results])
        vmax = np.nanmax([np.nanpercentile(res[field] - base, 99) for res in igrf_results])
    for res in igrf_results:
        print(f"Plotting magnetic field intensity {res['year']}")
        if relative:
            plot_igrf_map_core(
                data=res[field] - base,
                field=field,
                output_dir=output_dir,
                vmin=vmin,
                vmax=vmax,
                year=res["year"],
                base_year=base_year,
            )
        else:
            plot_igrf_map_core(
                data=res[field], field=field, output_dir=output_dir, vmin=vmin, vmax=vmax, year=res["year"]
            )


def plot_igrf_map_core(
    data: np.ndarray, field: str, output_dir: Path, vmin: float, vmax: float, year: int, base_year: int = None
) -> None:

    fig = plt.figure(figsize=(12, 6), dpi=150)  # noqa: F841
    proj = ccrs.Robinson()
    ax = plt.axes(projection=proj)

    if base_year is not None:
        abs_max = max(abs(vmin), abs(vmax))
        levels = np.linspace(-abs_max, abs_max, 40)
        norm = TwoSlopeNorm(vmin=-abs_max, vcenter=0.0, vmax=abs_max)
        cf = ax.contourf(
            Lon,
            Lat,
            data,
            levels=levels,
            norm=norm,
            transform=ccrs.PlateCarree(),
            cmap="PRGn",
            extend="both",
        )
    else:
        levels = np.linspace(vmin, vmax, 40)
        cf = ax.contourf(
            Lon,
            Lat,
            data,
            levels=levels,
            transform=ccrs.PlateCarree(),
            cmap="PRGn",
            extend="both",
        )

    # thin contour lines
    cs = ax.contour(
        Lon,
        Lat,
        data,
        levels=12,
        colors="k",
        linewidths=0.3,
        alpha=0.4,
        transform=ccrs.PlateCarree(),
    )

    ax.clabel(
        cs,
        inline=True,
        fontsize=6,
        fmt="%d",  # no decimals
        inline_spacing=5,
    )

    for txt in cs.labelTexts:
        txt.set_bbox(dict(facecolor="white", edgecolor="black", linewidth=0.5, alpha=0.9, boxstyle="round,pad=0.2"))

    # coastlines
    ax.coastlines(linewidth=0.5, color="black")

    # land styling
    ax.add_feature(cfeature.LAND, facecolor="lightgray", zorder=0)

    gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True, linewidth=0.6, color="gray", alpha=0.5, linestyle="--")
    gl.top_labels = False
    gl.right_labels = False
    gl.left_labels = True
    gl.bottom_labels = True

    gl.xlabel_style = {"size": 10}
    gl.ylabel_style = {"size": 10}
    gl.xlocator = cticker.LongitudeLocator(30)  # every 30°
    gl.ylocator = cticker.LatitudeLocator(15)  # every 15°

    cb = plt.colorbar(cf, orientation="horizontal", pad=0.08, shrink=0.8, extendrect=True, fraction=0.06)

    if base_year is not None:
        if field == "magnetic_field_intensity":
            cb.set_label("Total Magnetic Intensity Variation [nT]", fontsize=11)
            plt.title(f"Global {field.replace('_', ' ').title()} Variation - {year} vs {base_year}", fontsize=16, pad=20)
        else:
            cb.set_label(f"{field.capitalize()} Variation [degrees]", fontsize=11)
            plt.title(f"Global Magnetic {field.replace('_', ' ').title()} Variation - {year} vs {base_year}", fontsize=16, pad=20)
        out_file = output_dir.joinpath(f"igrf_map_variation_{year}vs{base_year}.png")
    else:
        if field == "magnetic_field_intensity":
            cb.set_label("Total Magnetic Intensity [nT]", fontsize=11)
            plt.title(f"Global {field.replace('_', ' ').title()} - Year {year}", fontsize=16, pad=20)
        else:
            cb.set_label(f"{field.capitalize()} [degrees]", fontsize=11)
            plt.title(f"Global Magnetic {field.replace('_', ' ').title()} - Year {year}", fontsize=16, pad=20)
        out_file = output_dir.joinpath(f"igrf_map_{year}.png")

    plt.savefig(out_file, dpi=320, bbox_inches="tight", pad_inches=0.3)
    plt.close("all")


if __name__ == "__main__":
    relative = False
    output_dir = Path(r"/path/to/output/dir")
    # create lon/lat grid
    lon_lat_grid, n_lon, n_lat = create_lat_lon_grid(lon_step_deg=0.5, lat_step_deg=0.5, height_km=0.0)

    Lon = lon_lat_grid[:, 0].reshape(n_lat, n_lon)
    Lat = lon_lat_grid[:, 1].reshape(n_lat, n_lon)

    igrf_results = []
    for date in [datetime(year, 1, 1) for year in range(2020, 2031, 5)]:
        print(f"Computing geodetic IGRF on Lat/Lon grid for {date}")
        results = field_computation_by_year(date=date, lon_lat_grid=lon_lat_grid, n_lon=n_lon, n_lat=n_lat)
        igrf_results.append(results)

    plot_igrf_map(igrf_results=igrf_results, field="magnetic_field_intensity", output_dir=output_dir, relative=relative)

    # optional, create a movie
    import imageio.v2 as imageio

    files = sorted(output_dir.glob("*.png"))

    movie_title = "igrf_evolution.mp4" if not relative else "igrf_evolution_vs1990.mp4"
    with imageio.get_writer(output_dir.joinpath(movie_title), fps=1) as writer:
        for f in files:
            writer.append_data(imageio.imread(f))
