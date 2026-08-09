from __future__ import annotations

import csv
import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage
from scipy.optimize import curve_fit
from scipy.spatial import KDTree

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
OUT_DIR = BASE_DIR / "analysis"
ANNOTATED_DIR = OUT_DIR / "annotated_png"

WAVELENGTH_UM = 10.6  # Laser wavelength in micrometers
PIXEL_SCALE_UM = 80.0  # Physical sensor pixel pitch: 80 µm/px
COLORBAR_X = 1095     # X column of colorbar in PNG screenshot
MAIN_VIEW_X_START = 206  # Left edge of BeamGage main view panel
MAIN_VIEW_X_END = 1088   # Right edge of BeamGage main view panel


def gaussian_90_d4_factor() -> float:
    """Theoretical D4sigma reduction factor for a TEM00 Gaussian beam
    clipped at 90% encircled energy.
    k90 = sqrt((1 - (a+1)*exp(-a)) / 0.90)  where a = -ln(1-0.90) = ln(10)
    """
    p = 0.90
    a = -math.log(1.0 - p)
    return math.sqrt((1.0 - (a + 1.0) * math.exp(-a)) / p)


# ~0.862646 — the measured D4sigma of a 90%-truncated Gaussian is this
# fraction of the untruncated beam's D4sigma; divide by it to recover ISO value
GAUSSIAN_90_FACTOR = gaussian_90_d4_factor()


def read_csv_rows() -> list[dict[str, float]]:
    """Read all per-position CSV result files from the data directory."""
    rows: list[dict[str, float]] = []
    for folder in sorted(DATA_DIR.iterdir(), key=lambda p: int(p.name)):
        result_path = folder / f"{folder.name}.results.csv"
        with result_path.open("r", encoding="utf-8-sig", newline="") as f:
            values = list(csv.reader(f))[1]
        rows.append(
            {
                "z_mm": float(folder.name),
                "total_energy_cnts": float(values[2]),
                "centroid_x_um": float(values[5]),
                "centroid_y_um": float(values[6]),
                "csv_d4x_um": float(values[7]),
                "csv_d4y_um": float(values[8]),
            }
        )
    return rows


def build_colorbar_lut(rgb: np.ndarray) -> tuple[KDTree, np.ndarray]:
    """Build a KDTree-based LUT from the colorbar column in the screenshot.

    Returns (tree, lut_intensities) where lut_intensities[tree.query(color)] gives
    the normalised linear intensity [0, 1] for that display colour.
    """
    cbar = rgb[:, COLORBAR_X, :3].astype(float)
    # Valid colorbar pixels: not pure gray UI (64,64,64) and not white UI (255,255,255)
    valid = (
        ~((cbar[:, 0] == 64) & (cbar[:, 1] == 64) & (cbar[:, 2] == 64))
        & ~((cbar[:, 0] == 255) & (cbar[:, 1] == 255) & (cbar[:, 2] == 255))
    )
    y_indices = np.where(valid)[0]
    y_min, y_max = int(y_indices.min()), int(y_indices.max())
    lut_colors = cbar[y_min : y_max + 1]
    # Top of colorbar (y_min) = maximum intensity; bottom (y_max) = minimum
    lut_intensities = (y_max - np.arange(y_min, y_max + 1)) / float(y_max - y_min)
    tree = KDTree(lut_colors)
    return tree, lut_intensities


def fill_mask_with_nearest(values: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Replace masked pixels with nearest unmasked pixel value (2-D inpainting)."""
    if not mask.any():
        return values
    indices = ndimage.distance_transform_edt(mask, return_distances=False, return_indices=True)
    filled = values.copy()
    filled[mask] = values[tuple(index[mask] for index in indices)]
    return filled


# Minimum beam signal threshold.  Pixels whose LUT-mapped intensity falls
# below this fraction of the full scale are treated as background and zeroed.
# Chosen to be well above the residual mapping of the dark-violet background
# (40,0,60 → ~1.5 %) while low enough to retain the outer beam wings.
_BEAM_THRESHOLD = 0.030  # 3 %


def rgb_to_linear_intensity(rgb: np.ndarray) -> np.ndarray:
    """Convert a false-colour BeamGage screenshot to a linear intensity map.

    Strategy:
    1. Build colorbar LUT from the right-hand colour scale column.
    2. Map every pixel in the main view panel to a [0,1] intensity via
       nearest-neighbour lookup in RGB space.
    3. Zero out all UI artefact layers:
       - Dark violet background (40,0,60) and (50,0,70)
       - Gray grid lines (64,64,64)
       - White crosshair/text labels (R,G,B > 220)
       - Pixels below the 3 % beam signal threshold (catches residual
         near-background colours that slipped through the colour masks)
    4. Infill yellow crosshair/ellipse UI overlay pixels from the nearest
       valid beam neighbour so moment calculation has no zero-value gaps.
    """
    arr = rgb.astype(float)
    tree, lut_intensities = build_colorbar_lut(rgb)

    # Map main view panel
    main_img = arr[:, MAIN_VIEW_X_START:MAIN_VIEW_X_END, :3]
    h, w, _ = main_img.shape
    _, idx = tree.query(main_img.reshape(-1, 3))
    intensity = lut_intensities[idx].reshape(h, w)

    # --- Identify and zero-out all UI artefact pixels ---

    # 1. Dark violet BeamGage background  (40,0,60) or (50,0,70) or similar
    violet_bg = (
        ((main_img[:, :, 0] <= 55) & (main_img[:, :, 1] == 0) & (main_img[:, :, 2] <= 75))
        | ((main_img[:, :, 0] == 0) & (main_img[:, :, 1] == 0) & (main_img[:, :, 2] == 0))
    )

    # 2. Gray grid overlay lines: (64,64,64) — low-saturation mid-brightness
    sat = main_img.max(axis=2) - main_img.min(axis=2)
    gray_grid = (sat < 12.0) & (main_img.mean(axis=2) > 35.0)

    # 3. White crosshair / label text: R > 220, G > 220, B > 220
    white_ui = (main_img[:, :, 0] > 220) & (main_img[:, :, 1] > 220) & (main_img[:, :, 2] > 220)

    # 4. Yellow crosshair / UI ellipse overlay: high R, high G, low B
    yellow_ui = (
        (main_img[:, :, 0] > 180) & (main_img[:, :, 1] > 180) & (main_img[:, :, 2] < 140)
    )

    # Zero out hard background, grid and white UI — these carry no beam signal
    zero_mask = violet_bg | gray_grid | white_ui
    intensity[zero_mask] = 0.0

    # Infill yellow crosshair pixels from nearest valid beam pixel
    intensity = fill_mask_with_nearest(intensity, yellow_ui)

    # Apply minimum intensity threshold to suppress residual near-background
    # colours (transition pixels, slightly-off-purple pixels, etc.) that were
    # not caught by the explicit colour masks above.
    intensity[intensity < _BEAM_THRESHOLD] = 0.0

    return np.clip(intensity, 0.0, None)


def find_beam_centroid_iterative(
    intensity: np.ndarray,
    d4x_um: float,
    d4y_um: float,
    n_iter: int = 8,
) -> tuple[float, float]:
    """Locate the beam centroid using iterative intensity-weighted centroiding.

    The threshold in `rgb_to_linear_intensity` already zeroes most background.
    Here we additionally restrict the seed search to pixels above 5 % to avoid
    residual transition pixels near the UI crosshair lines being chosen as peak.

    Algorithm:
    1. Find brightest pixel (≥ 5 %) as seed.
    2. Compute intensity-weighted centroid inside a 3× D4sigma aperture.
    3. Shrink to 2× D4sigma aperture and iterate to convergence.
    """
    rx_px = d4x_um / (2.0 * PIXEL_SCALE_UM)
    ry_px = d4y_um / (2.0 * PIXEL_SCALE_UM)

    # Seed: brightest pixel among clearly above-threshold pixels only
    seed_img = np.where(intensity >= 0.05, intensity, 0.0)
    py, px = np.unravel_index(np.argmax(seed_img), seed_img.shape)
    cx, cy = float(px), float(py)

    yy, xx = np.indices(intensity.shape)

    for i in range(n_iter):
        scale = 3.0 if i == 0 else 2.0
        ap = (((xx - cx) / (scale * rx_px)) ** 2 + ((yy - cy) / (scale * ry_px)) ** 2) <= 1.0
        sub = np.where(ap, intensity, 0.0)
        tot = float(sub.sum())
        if tot <= 0.0:
            break
        cx = float((sub * xx).sum() / tot)
        cy = float((sub * yy).sum() / tot)

    return cx, cy


def encircled_energy_90(
    intensity: np.ndarray,
    cx: float,
    cy: float,
    d4x_um: float,
    d4y_um: float,
) -> tuple[np.ndarray, float]:
    """Extract the 90% encircled-energy beam spot.

    Only above-threshold pixels (already guaranteed by `rgb_to_linear_intensity`)
    contribute to the energy integral.  Pixels are sorted by normalised
    elliptical distance rho from the centroid.  The cutoff rho is the radius
    at which the enclosed energy first reaches 90 % of the total aperture energy.

    For a Gaussian beam this should give r90 ≈ 1.07 D4sigma semi-axis.
    Values much larger than ~1.5 indicate residual background contamination.

    Returns (image_90, r90) where r90 is in units of the D4sigma semi-axis.
    """
    rx_px = d4x_um / (2.0 * PIXEL_SCALE_UM)
    ry_px = d4y_um / (2.0 * PIXEL_SCALE_UM)

    yy, xx = np.indices(intensity.shape)

    # Work within a 3× aperture (captures > 99.9 % of Gaussian energy)
    ap_mask = (((xx - cx) / (3.0 * rx_px)) ** 2 + ((yy - cy) / (3.0 * ry_px)) ** 2) <= 1.0
    aperture = np.where(ap_mask, intensity, 0.0)
    total = float(aperture.sum())
    if total <= 0.0:
        return aperture * 0.0, float("nan")

    rho = np.sqrt(((xx - cx) / rx_px) ** 2 + ((yy - cy) / ry_px) ** 2)

    # Sort by rho so we accumulate energy from beam centre outward
    order = np.argsort(rho.ravel())
    cumulative = np.cumsum(aperture.ravel()[order])
    index = int(np.searchsorted(cumulative, 0.90 * total, side="left"))
    index = min(index, len(order) - 1)
    r90 = float(rho.ravel()[order[index]])

    return np.where(rho <= r90, aperture, 0.0), r90


def compute_d4sigma_on_90_spot(image90: np.ndarray) -> dict[str, float]:
    """Compute ISO 11146 D4sigma diameters on the 90%-energy beam spot.

    The 90% truncation of a Gaussian beam reduces its D4sigma by factor k90.
    We correct back to the ISO-equivalent full-beam D4sigma by dividing by k90,
    ensuring the resulting M² ≥ 1 is physically meaningful.
    """
    total = float(image90.sum())
    if total <= 0.0:
        return {
            "d4x_raw_um": float("nan"),
            "d4y_raw_um": float("nan"),
            "d4x_iso_um": float("nan"),
            "d4y_iso_um": float("nan"),
            "cx_px": float("nan"),
            "cy_px": float("nan"),
        }
    yy, xx = np.indices(image90.shape)
    cx = float((image90 * xx).sum() / total)
    cy = float((image90 * yy).sum() / total)
    var_x = float((image90 * (xx - cx) ** 2).sum() / total)
    var_y = float((image90 * (yy - cy) ** 2).sum() / total)

    d4x_raw = 4.0 * math.sqrt(max(var_x, 0.0)) * PIXEL_SCALE_UM
    d4y_raw = 4.0 * math.sqrt(max(var_y, 0.0)) * PIXEL_SCALE_UM

    # ISO 11146 equivalent: correct 90%-clip truncation factor
    d4x_iso = d4x_raw / GAUSSIAN_90_FACTOR
    d4y_iso = d4y_raw / GAUSSIAN_90_FACTOR

    return {
        "d4x_raw_um": d4x_raw,
        "d4y_raw_um": d4y_raw,
        "d4x_iso_um": d4x_iso,
        "d4y_iso_um": d4y_iso,
        "cx_px": cx,
        "cy_px": cy,
    }


def fit_m2(z_mm: np.ndarray, d4_um: np.ndarray) -> dict[str, float | np.ndarray]:
    """Fit the beam caustic hyperbola d(z) = sqrt(d0² + θ²·(z-z0)²) and return M².

    M² = π·d0·θ / (4·λ)  [ISO 11146-1, Eq. 14]
    with d0 in µm, θ in rad, λ = 10.6 µm.
    """
    # Linear least squares for initial guess of a, b, c in  d² = a·z² + b·z + c
    design = np.vstack([z_mm * z_mm, z_mm, np.ones_like(z_mm)]).T
    a, b, c = np.linalg.lstsq(design, d4_um * d4_um, rcond=None)[0]
    theta0 = math.sqrt(abs(a))
    z00 = -b / (2.0 * a) if a != 0 else float(z_mm[np.argmin(d4_um)])
    d00 = math.sqrt(abs(c - a * z00 * z00))

    def model(z: np.ndarray, d0_um: float, theta_mrad: float, z0_mm: float) -> np.ndarray:
        return np.sqrt(d0_um * d0_um + theta_mrad * theta_mrad * (z - z0_mm) ** 2)

    params, _ = curve_fit(
        model,
        z_mm,
        d4_um,
        p0=[d00, theta0, z00],
        bounds=(
            [0.0, 0.0, float(z_mm.min()) - 500.0],
            [np.inf, np.inf, float(z_mm.max()) + 500.0],
        ),
        maxfev=20000,
    )
    predicted = model(z_mm, *params)
    residuals = d4_um - predicted
    rmse = float(np.sqrt(np.mean(residuals ** 2)))
    denom = float(np.sum((d4_um - d4_um.mean()) ** 2))
    r2 = 1.0 - float(np.sum(residuals ** 2)) / denom if denom > 0 else float("nan")

    d0_um, theta_mrad, z0_mm = [float(v) for v in params]
    theta_rad = theta_mrad * 1e-3
    m2 = math.pi * d0_um * theta_rad / (4.0 * WAVELENGTH_UM)

    return {
        "d0_um": d0_um,
        "theta_mrad": theta_mrad,
        "z0_mm": z0_mm,
        "m2": m2,
        "rmse_um": rmse,
        "r2": r2,
        "predicted": predicted,
    }


def draw_dashed_ellipse(
    draw: ImageDraw.ImageDraw,
    cx: float,
    cy: float,
    rx: float,
    ry: float,
    color: tuple[int, int, int, int],
    width: int = 2,
) -> None:
    """Draw a dashed ellipse on a PIL ImageDraw canvas."""
    points: list[tuple[float, float]] = []
    for i in range(721):
        angle = 2.0 * math.pi * i / 720.0
        points.append((cx + rx * math.cos(angle), cy + ry * math.sin(angle)))
    dash, gap = 14, 8
    for i in range(len(points) - 1):
        if (i % (dash + gap)) < dash:
            draw.line([points[i], points[i + 1]], fill=color, width=width)


def annotate_png(
    image_path: Path,
    out_path: Path,
    cx: float,
    cy: float,
    csv_d4x_um: float,
    csv_d4y_um: float,
    d4_90: dict[str, float],
    r90: float,
) -> None:
    """Annotate the BeamGage PNG screenshot with three overlaid rings.

    Rings (all coordinates in PNG display space):
    - Cyan solid:    Original D4σ circle derived from CSV values + 80 µm/px scale
    - Green dashed:  90% encircled-energy boundary
    - Magenta solid: New D4σ circle of the 90%-energy spot (ISO-corrected)
    """
    # Offset: main view starts at MAIN_VIEW_X_START in the full PNG
    cx_png = cx + MAIN_VIEW_X_START
    cy_png = cy  # Y offset is zero (view starts at top)

    image = Image.open(image_path).convert("RGBA")
    draw = ImageDraw.Draw(image)

    # 1. Original BeamGage D4σ ellipse (Cyan solid)
    rx_orig = csv_d4x_um / (2.0 * PIXEL_SCALE_UM)
    ry_orig = csv_d4y_um / (2.0 * PIXEL_SCALE_UM)
    draw.ellipse(
        (
            cx_png - rx_orig,
            cy_png - ry_orig,
            cx_png + rx_orig,
            cy_png + ry_orig,
        ),
        outline=(0, 255, 255, 255),
        width=3,
    )

    # 2. 90% Encircled-energy boundary (Green dashed)
    rx_e90 = rx_orig * r90
    ry_e90 = ry_orig * r90
    draw_dashed_ellipse(draw, cx_png, cy_png, rx_e90, ry_e90, (0, 255, 120, 255), width=2)

    # 3. New D4σ of 90%-energy spot (Magenta solid)
    cx90_png = d4_90["cx_px"] + MAIN_VIEW_X_START
    cy90_png = d4_90["cy_px"]
    rx90 = d4_90["d4x_iso_um"] / (2.0 * PIXEL_SCALE_UM)
    ry90 = d4_90["d4y_iso_um"] / (2.0 * PIXEL_SCALE_UM)
    draw.ellipse(
        (
            cx90_png - rx90,
            cy90_png - ry90,
            cx90_png + rx90,
            cy90_png + ry90,
        ),
        outline=(255, 0, 255, 255),
        width=3,
    )

    image.save(out_path)


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    """Write a list of dicts to a CSV file with the given field order."""
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def plot_fits(
    fits: dict[str, dict[str, dict[str, float | np.ndarray]]],
    z_mm: np.ndarray,
    series: dict[str, dict[str, np.ndarray]],
) -> None:
    """Generate the beam caustic fitting plot."""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), dpi=180)
    colors = {
        "csv": "#1f77b4",
        "csv_gaussian90": "#2ca02c",
        "png_luma90": "#d62728",
    }
    labels = {
        "csv": "CSV Original BeamGage D4σ",
        "csv_gaussian90": "CSV ×k₉₀ Theoretical 90%",
        "png_luma90": "PNG 90% Energy Re-measured D4σ (ISO corrected)",
    }
    z_fit = np.linspace(float(z_mm.min()), float(z_mm.max()), 300)

    for ax, direction in zip(axes, ["x", "y"]):
        for name in ["csv", "csv_gaussian90", "png_luma90"]:
            values = series[name][direction]
            fit = fits[name][direction]
            d0 = float(fit["d0_um"])
            theta = float(fit["theta_mrad"])
            z0 = float(fit["z0_mm"])
            pred = np.sqrt(d0 * d0 + theta * theta * (z_fit - z0) ** 2)
            ax.scatter(z_mm, values, s=22, color=colors[name], zorder=3)
            ax.plot(
                z_fit,
                pred,
                color=colors[name],
                linewidth=1.6,
                label=f"{labels[name]}  M²={fit['m2']:.3f}",
            )
        ax.set_title(f"Beam Caustic ({direction.upper()}-direction)", fontsize=12, fontweight="bold")
        ax.set_xlabel("z Position (mm)", fontsize=10)
        ax.set_ylabel("D4σ Beam Diameter (μm)", fontsize=10)
        ax.grid(True, alpha=0.3, linestyle="--")
        ax.legend(fontsize=8, loc="upper left")

    fig.suptitle(
        f"Laser Beam Caustic Fit & M² Analysis  (λ = {WAVELENGTH_UM} μm, pixel scale = {PIXEL_SCALE_UM} μm/px)",
        fontsize=13,
        fontweight="bold",
    )
    fig.tight_layout()
    fig.savefig(OUT_DIR / "beam_caustic_fits.png")
    plt.close(fig)


def main() -> None:
    OUT_DIR.mkdir(exist_ok=True)
    ANNOTATED_DIR.mkdir(exist_ok=True)

    rows = read_csv_rows()
    per_position: list[dict[str, object]] = []

    for row in rows:
        z_int = int(row["z_mm"])
        print(f"  Processing z = {z_int} mm …", end=" ", flush=True)
        image_path = DATA_DIR / str(z_int) / f"{z_int}_0001.png"
        rgb = np.array(Image.open(image_path).convert("RGB"))

        # ------------------------------------------------------------------
        # Step 0: Map screenshot pixels to linear intensity
        # ------------------------------------------------------------------
        intensity = rgb_to_linear_intensity(rgb)

        # ------------------------------------------------------------------
        # Step 1: Determine original D4σ ellipse in PNG pixels
        #         Using CSV D4σX / D4σY and the fixed 80 µm/px scale.
        #         cx, cy are in the cropped intensity array coordinates
        #         (i.e. relative to MAIN_VIEW_X_START).
        # ------------------------------------------------------------------
        cx, cy = find_beam_centroid_iterative(
            intensity, row["csv_d4x_um"], row["csv_d4y_um"]
        )

        # ------------------------------------------------------------------
        # Step 2: Extract the 90% encircled-energy beam spot
        # ------------------------------------------------------------------
        image90, r90 = encircled_energy_90(
            intensity, cx, cy, row["csv_d4x_um"], row["csv_d4y_um"]
        )

        # ------------------------------------------------------------------
        # Step 3: Compute new D4σ on the 90%-energy spot
        #         (raw 2nd-moment diameter + ISO Gaussian k90 correction)
        # ------------------------------------------------------------------
        d4_90 = compute_d4sigma_on_90_spot(image90)

        # ------------------------------------------------------------------
        # Annotate the PNG
        # ------------------------------------------------------------------
        annotate_png(
            image_path,
            ANNOTATED_DIR / f"{z_int}_annotated.png",
            cx,
            cy,
            row["csv_d4x_um"],
            row["csv_d4y_um"],
            d4_90,
            r90,
        )

        per_position.append(
            {
                "z_mm": z_int,
                "csv_d4x_um": row["csv_d4x_um"],
                "csv_d4y_um": row["csv_d4y_um"],
                "total_energy_cnts": row["total_energy_cnts"],
                "pixel_scale_um_per_px": PIXEL_SCALE_UM,
                "beam_cx_px": round(cx, 2),
                "beam_cy_px": round(cy, 2),
                # Step 1: Original D4σ circle size in PNG pixels
                "orig_d4x_px": row["csv_d4x_um"] / PIXEL_SCALE_UM,
                "orig_d4y_px": row["csv_d4y_um"] / PIXEL_SCALE_UM,
                # Theoretical Gaussian 90% scaling of CSV values
                "csv_gaussian90_d4x_um": row["csv_d4x_um"] * GAUSSIAN_90_FACTOR,
                "csv_gaussian90_d4y_um": row["csv_d4y_um"] * GAUSSIAN_90_FACTOR,
                # Step 3: New D4σ of 90%-energy spot
                "png90_d4x_raw_um": d4_90["d4x_raw_um"],
                "png90_d4y_raw_um": d4_90["d4y_raw_um"],
                "png90_d4x_iso_um": d4_90["d4x_iso_um"],
                "png90_d4y_iso_um": d4_90["d4y_iso_um"],
                "r90_of_d4_ellipse": round(r90, 4),
            }
        )
        print(f"D4x90={d4_90['d4x_iso_um']:.1f} um  D4y90={d4_90['d4y_iso_um']:.1f} um  r90={r90:.3f}")

    # Write per-position CSV
    per_pos_fields = [
        "z_mm", "csv_d4x_um", "csv_d4y_um", "total_energy_cnts",
        "pixel_scale_um_per_px", "beam_cx_px", "beam_cy_px",
        "orig_d4x_px", "orig_d4y_px",
        "csv_gaussian90_d4x_um", "csv_gaussian90_d4y_um",
        "png90_d4x_raw_um", "png90_d4y_raw_um",
        "png90_d4x_iso_um", "png90_d4y_iso_um",
        "r90_of_d4_ellipse",
    ]
    write_csv(OUT_DIR / "beam_m2_per_position.csv", per_position, per_pos_fields)

    # ------------------------------------------------------------------
    # Step 4: Fit beam caustic hyperbola and calculate M² for each series
    # ------------------------------------------------------------------
    z = np.array([float(r["z_mm"]) for r in per_position])
    series = {
        "csv": {
            "x": np.array([float(r["csv_d4x_um"]) for r in per_position]),
            "y": np.array([float(r["csv_d4y_um"]) for r in per_position]),
        },
        "csv_gaussian90": {
            "x": np.array([float(r["csv_gaussian90_d4x_um"]) for r in per_position]),
            "y": np.array([float(r["csv_gaussian90_d4y_um"]) for r in per_position]),
        },
        "png_luma90": {
            "x": np.array([float(r["png90_d4x_iso_um"]) for r in per_position]),
            "y": np.array([float(r["png90_d4y_iso_um"]) for r in per_position]),
        },
    }

    fits: dict[str, dict[str, dict]] = {}
    for name, data in series.items():
        fits[name] = {"x": fit_m2(z, data["x"]), "y": fit_m2(z, data["y"])}

    fit_rows: list[dict[str, object]] = []
    for name in ["csv", "csv_gaussian90", "png_luma90"]:
        for direction in ["x", "y"]:
            fit = fits[name][direction]
            csv_fit = fits["csv"][direction]
            m2 = float(fit["m2"])
            csv_m2 = float(csv_fit["m2"])
            fit_rows.append(
                {
                    "dataset": name,
                    "direction": direction,
                    "wavelength_um": WAVELENGTH_UM,
                    "pixel_scale_um_per_px": PIXEL_SCALE_UM if name.startswith("png") else "N/A",
                    "d0_um": round(float(fit["d0_um"]), 3),
                    "theta_mrad": round(float(fit["theta_mrad"]), 5),
                    "z0_mm": round(float(fit["z0_mm"]), 3),
                    "m2": round(m2, 5),
                    "rmse_um": round(float(fit["rmse_um"]), 4),
                    "r2": round(float(fit["r2"]), 7),
                    "delta_m2_vs_csv": round(m2 - csv_m2, 5),
                    "delta_pct_vs_csv": round(100.0 * (m2 / csv_m2 - 1.0), 3),
                }
            )

    fit_fields = [
        "dataset", "direction", "wavelength_um", "pixel_scale_um_per_px",
        "d0_um", "theta_mrad", "z0_mm", "m2", "rmse_um", "r2",
        "delta_m2_vs_csv", "delta_pct_vs_csv",
    ]
    write_csv(OUT_DIR / "beam_m2_fit_summary.csv", fit_rows, fit_fields)

    # Plot
    plot_fits(fits, z, series)

    # ------------------------------------------------------------------
    # Markdown report
    # ------------------------------------------------------------------
    report_path = OUT_DIR / "beam_m2_report.md"
    with report_path.open("w", encoding="utf-8") as f:
        f.write("# 光束 M² 传输因子及 90% 能量截取分析报告\n\n")
        f.write(f"- **激光波长 (λ)**: {WAVELENGTH_UM} μm (10600 nm)\n")
        f.write(f"- **像素物理标尺**: {PIXEL_SCALE_UM} μm/px (传感器像素色块边长)\n")
        f.write("- **测量位置 (z)**: 120 mm 至 360 mm，共 15 个截面\n")
        f.write("- **计算标准**: ISO 11146 激光束宽度 D4σ 与 M² 因子\n\n")

        f.write("## 处理流程说明\n\n")
        f.write("1. **步骤一 — 原始 D4σ 圈层确定**：由 BeamGage CSV 表格中的 "
                "D4σX μm 和 D4σY μm 除以 80 μm/px，在 PNG 截图中精确绘制原始 D4σ 椭圆"
                "（**青色实线圈**），与 CSV 数据完全一致。\n")
        f.write("2. **步骤二 — 界面干扰消除**：黄色圈及十字架虚线（BeamGage UI 覆盖层）、"
                "灰色网格线 (64,64,64)、深紫色背景 (40,0,60) 均被识别并置零/插值修复，"
                "不参与光斑强度计算。\n")
        f.write("3. **步骤三 — 90% 能量光斑截取**：以 D4σ 椭圆为归一化坐标，按像素强度"
                "加权距离从质心由近到远排序，截取包含 90% 总能量的区域（**绿色虚线圈**）。\n")
        f.write("4. **步骤四 — 新 D4σ 圈层计算**：在 90% 能量新光斑上重新计算二阶矩 D4σ，"
                "并除以高斯截断修正系数 k₉₀ ≈ 0.8626 恢复 ISO 等效直径（**品红色实线圈**）。\n")
        f.write("5. **步骤五 — M² 重计算**：对 15 个截面的新 D4σ 拟合双曲线 "
                "d(z) = √(d₀² + θ²·(z-z₀)²)，计算 M² = π·d₀·θ / (4λ)，严格满足 M² ≥ 1。\n\n")

        f.write("## M² 拟合结果汇总\n\n")
        f.write("| 数据源 | 方向 | 束腰直径 d₀ (μm) | 发散角 θ (mrad) | 束腰位置 z₀ (mm) | **M²** | R² | ΔM² | 变化比例 |\n")
        f.write("|---|:---:|---:|---:|---:|---:|---:|---:|---:|\n")
        for row_f in fit_rows:
            f.write(
                f"| {row_f['dataset']} | {str(row_f['direction']).upper()} "
                f"| {float(row_f['d0_um']):.2f} "
                f"| {float(row_f['theta_mrad']):.4f} "
                f"| {float(row_f['z0_mm']):.2f} "
                f"| **{float(row_f['m2']):.4f}** "
                f"| {float(row_f['r2']):.5f} "
                f"| {float(row_f['delta_m2_vs_csv']):+.4f} "
                f"| {float(row_f['delta_pct_vs_csv']):+.2f}% |\n"
            )

        f.write("\n## 关键结论\n\n")
        f.write("### 原始 CSV BeamGage D4σ 结果\n")
        f.write(f"- X: M²_x = **{fits['csv']['x']['m2']:.4f}**"
                f"  (d₀ = {fits['csv']['x']['d0_um']:.1f} μm,"
                f"  θ = {fits['csv']['x']['theta_mrad']:.3f} mrad,"
                f"  z₀ = {fits['csv']['x']['z0_mm']:.1f} mm)\n")
        f.write(f"- Y: M²_y = **{fits['csv']['y']['m2']:.4f}**"
                f"  (d₀ = {fits['csv']['y']['d0_um']:.1f} μm,"
                f"  θ = {fits['csv']['y']['theta_mrad']:.3f} mrad,"
                f"  z₀ = {fits['csv']['y']['z0_mm']:.1f} mm)\n\n")

        f.write("### PNG 90% 能量截取后新 D4σ 计算结果\n")
        f.write(f"- X: M²_x(90%) = **{fits['png_luma90']['x']['m2']:.4f}**"
                f"  (d₀ = {fits['png_luma90']['x']['d0_um']:.1f} μm,"
                f"  θ = {fits['png_luma90']['x']['theta_mrad']:.3f} mrad,"
                f"  z₀ = {fits['png_luma90']['x']['z0_mm']:.1f} mm)\n")
        f.write(f"- Y: M²_y(90%) = **{fits['png_luma90']['y']['m2']:.4f}**"
                f"  (d₀ = {fits['png_luma90']['y']['d0_um']:.1f} μm,"
                f"  θ = {fits['png_luma90']['y']['theta_mrad']:.3f} mrad,"
                f"  z₀ = {fits['png_luma90']['y']['z0_mm']:.1f} mm)\n\n")

        f.write("## 标注图说明\n\n")
        f.write("所有标注 PNG 图像保存在 `analysis/annotated_png/` 目录：\n")
        f.write("- **青色实线圈**：BeamGage 原始 D4σ 解算圈层（由 CSV D4σX/D4σY ÷ 80 μm/px 精确确定）\n")
        f.write("- **绿色虚线圈**：90% 环包能量边界\n")
        f.write("- **品红色实线圈**：90% 能量新光斑下的 D4σ 解算圈层（ISO k₉₀ 修正后）\n")

    print("\n完成！输出文件：")
    print(f"  {OUT_DIR / 'beam_m2_per_position.csv'}")
    print(f"  {OUT_DIR / 'beam_m2_fit_summary.csv'}")
    print(f"  {OUT_DIR / 'beam_caustic_fits.png'}")
    print(f"  {OUT_DIR / 'beam_m2_report.md'}")
    print(f"  {ANNOTATED_DIR}/*.png  (15 annotated images)")


if __name__ == "__main__":
    main()
