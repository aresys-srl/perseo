---
icon: lucide/cloud-download
tags:
    - atmospheric perturbations
    - ionosphere
    - troposphere
    - localization
---

# Atmospheric Delays { #atm data-toc-label="Atmospheric Delays" }

Synthetic Aperture Radar (SAR) signals propagate through the Earth’s atmosphere before reaching the sensor. During this path,
the signal is affected by variations in refractive index caused primarily by the **troposphere** and the **ionosphere**.
These effects introduce phase delays that can significantly impact applications such as interferometry (InSAR), geolocation,
and deformation monitoring.

---

## Tropospheric Delay

The troposphere (from the surface up to ~10–12 km) is a **non-dispersive medium** at microwave frequencies, meaning the delay
is independent of signal frequency. The total tropospheric delay is typically decomposed into:

- ==**Hydrostatic (dry) component**==: caused by dry gases in the atmosphere; it accounts for ~90% of the total delay and
can be modeled accurately from surface pressure.

- ==**Wet component**==: caused by water vapor; highly variable in space and time, making it the main source of uncertainty
in tropospheric correction.

The propagation delay $\Delta L$ of a signal can be expressed as:

$$
\Delta L = 10^{-6} \int N(s) ds
$$

where $N$ is the refractivity and the integral is taken along the signal path.

In SAR interferometry, spatial and temporal variations in the wet delay can produce phase artifacts that may be
misinterpreted as ground deformation and affect the localization accuracy.

To mitigate these effects, this module uses the **Vienna Mapping Function 3 (VMF3)**, which relies on:

- Precomputed Legendre coefficients
- Station grid point coordinates (1°×1° and 5°×5° grids)

VMF3 provides accurate mapping of zenith delays to slant delays based on the acquisition geometry, improving the
correction of tropospheric phase contributions.

---

## Ionospheric Delay

The ionosphere (extending from ~60 km to over 1000 km) is a **dispersive medium**, meaning the delay depends on the radar
signal frequency. The ionospheric phase advance is proportional to the **Total Electron Content (TEC)** along the signal path:

$$
\Delta \Phi \propto \frac{TEC}{f}
$$

where $TEC$ is the integrated electron density and $f$ is the radar frequency.

Key characteristics:

- Stronger impact at lower frequencies (e.g., L-band SAR)
- Causes phase advance (opposite sign compared to tropospheric delay)
- Can introduce:
    - Azimuth streaking
    - Phase ramps
    - Signal decorrelation

---

## Impact on SAR and InSAR Products

Uncompensated atmospheric delays can lead to:

- Phase biases in interferograms
- False deformation signals
- Reduced coherence
- Errors in geolocation and height estimation
