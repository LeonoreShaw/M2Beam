from __future__ import annotations

import csv
import math
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage
from scipy.optimize import curve_fit
from scipy.spatial import cKDTree


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
OUT_DIR = BASE_DIR / "analysis"
ANNOTATED_DIR = OUT_DIR / "annotated_png"

WAVELENGTH_UM = 10.6
CAMERA_PIXEL_SIZE_UM = 80.0
MAIN_IMAGE_RIGHT_X = 1088
ENERGY_DOMAIN_RADIUS = 1.5


def read_csv_rows() -> list[dict[str, float]]:
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


def largest_component(mask: np.ndarray) -> tuple[slice, slice, np.ndarray]:
    labels, _ = ndimage.label(mask, structure=np.ones((3, 3), dtype=bool))
    objects = ndimage.find_objects(labels)
    best = None
    for idx, slices in enumerate(objects, start=1):
        if slices is None:
            continue
        ys, xs = slices
        area = int((labels[slices] == idx).sum())
        width = xs.stop - xs.start
        height = ys.stop - ys.start
        if area > 20 and width > 20 and height > 20:
            if best is None or area > best[0]:
                best = (area, idx, ys, xs)
    if best is None:
        raise RuntimeError("No yellow overlay ellipse was found.")
    _, idx, ys, xs = best
    return ys, xs, labels == idx


def detect_overlay_center_and_mask(rgb: np.ndarray) -> dict[str, float | np.ndarray]:
    yy, xx = np.indices(rgb.shape[:2])
    yellow = (
        (rgb[:, :, 0] > 180)
        & (rgb[:, :, 1] > 180)
        & (rgb[:, :, 2] < 190)
        & (xx < MAIN_IMAGE_RIGHT_X)
    )
    ys, xs, yellow_component = largest_component(yellow)
    x0, x1 = xs.start, xs.stop
    y0, y1 = ys.start, ys.stop

    main = rgb[:, :MAIN_IMAGE_RIGHT_X, :]
    white = (main[:, :, 0] > 180) & (main[:, :, 1] > 180) & (main[:, :, 2] > 180)
    yellow_main = yellow_component[:, :MAIN_IMAGE_RIGHT_X]
    overlay = ndimage.binary_dilation(yellow_main, structure=np.ones((5, 5), dtype=bool))
    overlay |= ndimage.binary_dilation(white, structure=np.ones((3, 3), dtype=bool))

    return {
        "center_x_screen": (x0 + x1 - 1.0) / 2.0,
        "center_y_screen": (y0 + y1 - 1.0) / 2.0,
        "yellow_x0": float(x0),
        "yellow_x1": float(x1),
        "yellow_y0": float(y0),
        "yellow_y1": float(y1),
        "overlay_mask": overlay,
    }


def line_run_lengths(colors: np.ndarray) -> list[int]:
    runs: list[int] = []
    last = tuple(colors[0])
    length = 1
    for color in colors[1:]:
        item = tuple(color)
        if item == last:
            length += 1
        else:
            runs.append(length)
            last = item
            length = 1
    runs.append(length)
    return runs


def estimate_display_scale(rgb: np.ndarray, axis: str) -> tuple[float, str]:
    main = rgb[:, :MAIN_IMAGE_RIGHT_X, :]
    counts: Counter[int] = Counter()
    if axis == "x":
        for y in range(10, main.shape[0] - 10, 5):
            for run in line_run_lengths(main[y]):
                if 2 <= run <= 40:
                    counts[run] += 1
    elif axis == "y":
        for x in range(10, main.shape[1] - 10, 5):
            for run in line_run_lengths(main[:, x]):
                if 2 <= run <= 40:
                    counts[run] += 1
    else:
        raise ValueError(f"Unknown axis: {axis}")

    best_scale = 1.0
    best_score = -1.0
    for scale in np.arange(2.0, 12.01, 0.5):
        score = 0.0
        for run, count in counts.items():
            multiple = round(run / scale)
            if multiple < 1:
                continue
            error = abs(run - multiple * scale)
            if error <= max(0.35, 0.12 * scale):
                score += count / (multiple**0.6)
        if score > best_score:
            best_scale = float(scale)
            best_score = score

    common = ";".join(f"{run}:{count}" for run, count in counts.most_common(8))
    return best_scale, common


def extract_palette(rgb: np.ndarray) -> tuple[np.ndarray, cKDTree]:
    colorbar = rgb[:, 1089:1104, :].astype(float)
    median_colors = np.median(colorbar, axis=1)
    saturation = median_colors.max(axis=1) - median_colors.min(axis=1)
    valid = saturation > 40
    runs: list[tuple[int, int, int]] = []
    index = 0
    while index < len(valid):
        if not valid[index]:
            index += 1
            continue
        end = index
        while end < len(valid) and valid[end]:
            end += 1
        runs.append((end - index, index, end))
        index = end
    if not runs:
        raise RuntimeError("No usable colorbar was found.")
    _, start, end = max(runs)
    colors = median_colors[start:end]
    return colors, cKDTree(colors)


def fill_mask_with_nearest(values: np.ndarray, mask: np.ndarray) -> np.ndarray:
    if not mask.any():
        return values
    indices = ndimage.distance_transform_edt(mask, return_distances=False, return_indices=True)
    filled = values.copy()
    filled[mask] = values[tuple(index[mask] for index in indices)]
    return filled


def png_palette_intensity(rgb: np.ndarray, overlay_mask: np.ndarray) -> np.ndarray:
    main = rgb[:, :MAIN_IMAGE_RIGHT_X, :].astype(float)
    colors, tree = extract_palette(rgb)
    _, palette_index = tree.query(main.reshape(-1, 3), k=1)
    intensity = (1.0 - palette_index / (len(colors) - 1)).reshape(main.shape[:2])

    saturation = main.max(axis=2) - main.min(axis=2)
    gray_ui = (saturation < 12.0) & (main.mean(axis=2) > 35.0)
    intensity[gray_ui] = 0.0
    intensity = fill_mask_with_nearest(intensity, overlay_mask)
    return np.clip(intensity, 0.0, None)


def subtract_background(values: np.ndarray, bg_mask: np.ndarray | None = None) -> tuple[np.ndarray, float]:
    image = values.astype(float).copy()
    if bg_mask is not None and bg_mask.any():
        samples = image[bg_mask]
    else:
        samples = image[image > 0]
    samples = samples[samples > 0]
    bg = float(np.percentile(samples, 10)) if samples.size else 0.0
    return np.clip(image - bg, 0.0, None), bg


def d4sigma_screen(
    values: np.ndarray,
    scale_x_screen_per_camera_px: float,
    scale_y_screen_per_camera_px: float,
) -> dict[str, float]:
    total = float(values.sum())
    if total <= 0.0:
        return {
            "d4x_um": float("nan"),
            "d4y_um": float("nan"),
            "cx_screen": float("nan"),
            "cy_screen": float("nan"),
            "total": 0.0,
        }
    yy, xx = np.indices(values.shape)
    cx = float((values * xx).sum() / total)
    cy = float((values * yy).sum() / total)
    var_x = float((values * (xx - cx) ** 2).sum() / total)
    var_y = float((values * (yy - cy) ** 2).sum() / total)
    um_per_screen_x = CAMERA_PIXEL_SIZE_UM / scale_x_screen_per_camera_px
    um_per_screen_y = CAMERA_PIXEL_SIZE_UM / scale_y_screen_per_camera_px
    return {
        "d4x_um": 4.0 * math.sqrt(max(var_x, 0.0)) * um_per_screen_x,
        "d4y_um": 4.0 * math.sqrt(max(var_y, 0.0)) * um_per_screen_y,
        "cx_screen": cx,
        "cy_screen": cy,
        "total": total,
    }


def process_png(row: dict[str, float]) -> dict[str, object]:
    z_int = int(row["z_mm"])
    image_path = DATA_DIR / str(z_int) / f"{z_int}_0001.png"
    rgb = np.array(Image.open(image_path).convert("RGB"))

    overlay = detect_overlay_center_and_mask(rgb)
    scale_x, scale_x_runs = estimate_display_scale(rgb, "x")
    scale_y, scale_y_runs = estimate_display_scale(rgb, "y")
    intensity = png_palette_intensity(rgb, overlay["overlay_mask"])

    center_x = float(overlay["center_x_screen"])
    center_y = float(overlay["center_y_screen"])
    original_width_screen = row["csv_d4x_um"] / CAMERA_PIXEL_SIZE_UM * scale_x
    original_height_screen = row["csv_d4y_um"] / CAMERA_PIXEL_SIZE_UM * scale_y
    rx = original_width_screen / 2.0
    ry = original_height_screen / 2.0

    yy, xx = np.indices(intensity.shape)
    original_radius = np.sqrt(((xx - center_x) / rx) ** 2 + ((yy - center_y) / ry) ** 2)
    energy_domain = original_radius <= ENERGY_DOMAIN_RADIUS
    bg_mask = (original_radius >= ENERGY_DOMAIN_RADIUS + 0.15) & (original_radius <= ENERGY_DOMAIN_RADIUS + 0.8) & (intensity > 0)
    corrected, bg_level = subtract_background(intensity, bg_mask)
    domain_image = np.where(energy_domain, corrected, 0.0)
    domain_total = float(domain_image.sum())

    if domain_total <= 0.0:
        raise RuntimeError(f"No positive PNG energy found for z={z_int}.")

    energy_cx = float((domain_image * xx).sum() / domain_total)
    energy_cy = float((domain_image * yy).sum() / domain_total)
    energy_radius = np.sqrt(((xx - energy_cx) / rx) ** 2 + ((yy - energy_cy) / ry) ** 2)
    flat_radius = energy_radius[energy_domain].ravel()
    flat_energy = domain_image[energy_domain].ravel()
    order = np.argsort(flat_radius)
    cumulative = np.cumsum(flat_energy[order])
    index = int(np.searchsorted(cumulative, 0.90 * domain_total, side="left"))
    index = min(index, len(order) - 1)
    r90 = float(flat_radius[order[index]])
    image90 = np.where((energy_radius <= r90) & energy_domain, domain_image, 0.0)
    result90 = d4sigma_screen(image90, scale_x, scale_y)
    original_check = d4sigma_screen(domain_image, scale_x, scale_y)

    ratio_x = result90["d4x_um"] / original_check["d4x_um"] if original_check["d4x_um"] > 0 else float("nan")
    ratio_y = result90["d4y_um"] / original_check["d4y_um"] if original_check["d4y_um"] > 0 else float("nan")
    inferred90_d4x_um = row["csv_d4x_um"] * ratio_x
    inferred90_d4y_um = row["csv_d4y_um"] * ratio_y
    inferred90_width_screen = inferred90_d4x_um / CAMERA_PIXEL_SIZE_UM * scale_x
    inferred90_height_screen = inferred90_d4y_um / CAMERA_PIXEL_SIZE_UM * scale_y
    energy_boundary_x_um = row["csv_d4x_um"] * r90
    energy_boundary_y_um = row["csv_d4y_um"] * r90
    energy_boundary_width_screen = original_width_screen * r90
    energy_boundary_height_screen = original_height_screen * r90

    energy_fraction = float(image90.sum() / domain_total)
    result = {
        "z_mm": z_int,
        "csv_d4x_um": row["csv_d4x_um"],
        "csv_d4y_um": row["csv_d4y_um"],
        "total_energy_cnts": row["total_energy_cnts"],
        "csv_centroid_x_um": row["centroid_x_um"],
        "csv_centroid_y_um": row["centroid_y_um"],
        "screen_center_x": center_x,
        "screen_center_y": center_y,
        "display_scale_x_screen_px_per_camera_px": scale_x,
        "display_scale_y_screen_px_per_camera_px": scale_y,
        "scale_x_run_counts": scale_x_runs,
        "scale_y_run_counts": scale_y_runs,
        "original_d4_width_screen_px": original_width_screen,
        "original_d4_height_screen_px": original_height_screen,
        "png_original_moment_d4x_um": original_check["d4x_um"],
        "png_original_moment_d4y_um": original_check["d4y_um"],
        "png90_d4x_um": result90["d4x_um"],
        "png90_d4y_um": result90["d4y_um"],
        "png90_to_png_original_ratio_x": ratio_x,
        "png90_to_png_original_ratio_y": ratio_y,
        "inferred90_d4x_um": inferred90_d4x_um,
        "inferred90_d4y_um": inferred90_d4y_um,
        "inferred90_width_screen_px": inferred90_width_screen,
        "inferred90_height_screen_px": inferred90_height_screen,
        "png90_centroid_x_screen": result90["cx_screen"],
        "png90_centroid_y_screen": result90["cy_screen"],
        "png90_energy_boundary_r_over_original_d4_radius": r90,
        "energy_boundary_x_um": energy_boundary_x_um,
        "energy_boundary_y_um": energy_boundary_y_um,
        "energy_boundary_width_screen_px": energy_boundary_width_screen,
        "energy_boundary_height_screen_px": energy_boundary_height_screen,
        "png90_energy_fraction": energy_fraction,
        "png_background_level": bg_level,
        "yellow_overlay_width_screen_px": float(overlay["yellow_x1"]) - float(overlay["yellow_x0"]),
        "yellow_overlay_height_screen_px": float(overlay["yellow_y1"]) - float(overlay["yellow_y0"]),
    }
    annotate_png(image_path, ANNOTATED_DIR / f"{z_int}_annotated.png", result)
    return result


def fit_m2(z_mm: np.ndarray, d4_um: np.ndarray) -> dict[str, float | np.ndarray]:
    design = np.vstack([z_mm * z_mm, z_mm, np.ones_like(z_mm)]).T
    a, b, c = np.linalg.lstsq(design, d4_um * d4_um, rcond=None)[0]
    theta0 = math.sqrt(abs(a))
    z00 = -b / (2.0 * a) if a != 0 else float(z_mm[np.argmin(d4_um)])
    d00_sq = c - a * z00 * z00
    d00 = math.sqrt(abs(d00_sq))

    def model(z: np.ndarray, d0_um: float, theta_um_per_mm: float, z0_mm: float) -> np.ndarray:
        return np.sqrt(d0_um * d0_um + theta_um_per_mm * theta_um_per_mm * (z - z0_mm) ** 2)

    params, _ = curve_fit(
        model,
        z_mm,
        d4_um,
        p0=[d00, theta0, z00],
        bounds=([0.0, 0.0, float(z_mm.min()) - 500.0], [np.inf, np.inf, float(z_mm.max()) + 500.0]),
        maxfev=20000,
    )
    predicted = model(z_mm, *params)
    residuals = d4_um - predicted
    rmse = float(np.sqrt(np.mean(residuals * residuals)))
    denom = float(np.sum((d4_um - d4_um.mean()) ** 2))
    r2 = 1.0 - float(np.sum(residuals * residuals)) / denom if denom else float("nan")
    d0_um, theta_um_per_mm, z0_mm = [float(v) for v in params]
    theta_rad = theta_um_per_mm * 1e-3
    m2 = math.pi * d0_um * theta_rad / (4.0 * WAVELENGTH_UM)
    return {
        "d0_um": d0_um,
        "theta_um_per_mm": theta_um_per_mm,
        "z0_mm": z0_mm,
        "m2": m2,
        "rmse_um": rmse,
        "r2": r2,
        "predicted": predicted,
    }


def draw_dashed_ellipse(draw: ImageDraw.ImageDraw, box: tuple[float, float, float, float], color: tuple[int, int, int], width: int) -> None:
    x0, y0, x1, y1 = box
    cx = (x0 + x1) / 2.0
    cy = (y0 + y1) / 2.0
    rx = (x1 - x0) / 2.0
    ry = (y1 - y0) / 2.0
    points: list[tuple[float, float]] = []
    for i in range(721):
        angle = 2.0 * math.pi * i / 720.0
        points.append((cx + rx * math.cos(angle), cy + ry * math.sin(angle)))
    dash = 14
    gap = 8
    for i in range(len(points) - 1):
        if (i % (dash + gap)) < dash:
            draw.line([points[i], points[i + 1]], fill=color, width=width)


def annotate_png(image_path: Path, out_path: Path, result: dict[str, object]) -> None:
    image = Image.open(image_path).convert("RGBA")
    draw = ImageDraw.Draw(image)

    cx = float(result["screen_center_x"])
    cy = float(result["screen_center_y"])
    original_w = float(result["original_d4_width_screen_px"])
    original_h = float(result["original_d4_height_screen_px"])
    draw.ellipse(
        (cx - original_w / 2.0, cy - original_h / 2.0, cx + original_w / 2.0, cy + original_h / 2.0),
        outline=(0, 255, 255, 255),
        width=3,
    )

    r90 = float(result["png90_energy_boundary_r_over_original_d4_radius"])
    e_cx = float(result["png90_centroid_x_screen"])
    e_cy = float(result["png90_centroid_y_screen"])
    draw_dashed_ellipse(
        draw,
        (
            e_cx - original_w * r90 / 2.0,
            e_cy - original_h * r90 / 2.0,
            e_cx + original_w * r90 / 2.0,
            e_cy + original_h * r90 / 2.0,
        ),
        (0, 255, 120, 255),
        2,
    )

    scale_x = float(result["display_scale_x_screen_px_per_camera_px"])
    scale_y = float(result["display_scale_y_screen_px_per_camera_px"])
    new_w = float(result["inferred90_d4x_um"]) / CAMERA_PIXEL_SIZE_UM * scale_x
    new_h = float(result["inferred90_d4y_um"]) / CAMERA_PIXEL_SIZE_UM * scale_y
    draw.ellipse(
        (e_cx - new_w / 2.0, e_cy - new_h / 2.0, e_cx + new_w / 2.0, e_cy + new_h / 2.0),
        outline=(255, 0, 255, 255),
        width=3,
    )
    image.save(out_path)


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def plot_fits(fits: dict[str, dict[str, dict[str, float | np.ndarray]]], z_mm: np.ndarray, series: dict[str, dict[str, np.ndarray]]) -> None:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), dpi=160)
    colors = {
        "csv_original": "#1f77b4",
        "png90_shape_inferred": "#d62728",
        "png90": "#7f7f7f",
        "png_original_moment_check": "#ff7f0e",
    }
    labels = {
        "csv_original": "CSV original D4sigma",
        "png90_shape_inferred": "CSV-anchored PNG 90% D4sigma",
        "png90": "PNG absolute 90% diagnostic",
        "png_original_moment_check": "PNG original moment check",
    }
    z_fit = np.linspace(float(z_mm.min()), float(z_mm.max()), 300)
    for ax, direction in zip(axes, ["x", "y"]):
        for name in ["csv_original", "png90_shape_inferred", "png90", "png_original_moment_check"]:
            values = series[name][direction]
            fit = fits[name][direction]
            d0 = float(fit["d0_um"])
            theta = float(fit["theta_um_per_mm"])
            z0 = float(fit["z0_mm"])
            prediction = np.sqrt(d0 * d0 + theta * theta * (z_fit - z0) ** 2)
            ax.scatter(z_mm, values, s=18, color=colors[name])
            ax.plot(z_fit, prediction, color=colors[name], linewidth=1.4, label=labels[name])
        ax.set_title(f"{direction.upper()} direction")
        ax.set_xlabel("z position (mm)")
        ax.set_ylabel("D4sigma (um)")
        ax.grid(True, alpha=0.25)
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "beam_caustic_fits.png")
    plt.close(fig)


def main() -> None:
    OUT_DIR.mkdir(exist_ok=True)
    ANNOTATED_DIR.mkdir(exist_ok=True)

    rows = read_csv_rows()
    per_position = [process_png(row) for row in rows]

    fields = [
        "z_mm",
        "csv_d4x_um",
        "csv_d4y_um",
        "total_energy_cnts",
        "csv_centroid_x_um",
        "csv_centroid_y_um",
        "screen_center_x",
        "screen_center_y",
        "display_scale_x_screen_px_per_camera_px",
        "display_scale_y_screen_px_per_camera_px",
        "original_d4_width_screen_px",
        "original_d4_height_screen_px",
        "png_original_moment_d4x_um",
        "png_original_moment_d4y_um",
        "png90_d4x_um",
        "png90_d4y_um",
        "png90_to_png_original_ratio_x",
        "png90_to_png_original_ratio_y",
        "inferred90_d4x_um",
        "inferred90_d4y_um",
        "inferred90_width_screen_px",
        "inferred90_height_screen_px",
        "png90_centroid_x_screen",
        "png90_centroid_y_screen",
        "png90_energy_boundary_r_over_original_d4_radius",
        "energy_boundary_x_um",
        "energy_boundary_y_um",
        "energy_boundary_width_screen_px",
        "energy_boundary_height_screen_px",
        "png90_energy_fraction",
        "png_background_level",
        "yellow_overlay_width_screen_px",
        "yellow_overlay_height_screen_px",
        "scale_x_run_counts",
        "scale_y_run_counts",
    ]
    write_csv(OUT_DIR / "beam_m2_per_position.csv", per_position, fields)

    contour_rows: list[dict[str, object]] = []
    for row in per_position:
        contour_rows.append(
            {
                "z_mm": row["z_mm"],
                "cyan_csv_original_d4x_um": row["csv_d4x_um"],
                "cyan_csv_original_d4y_um": row["csv_d4y_um"],
                "cyan_center_x_screen_px": row["screen_center_x"],
                "cyan_center_y_screen_px": row["screen_center_y"],
                "cyan_width_screen_px": row["original_d4_width_screen_px"],
                "cyan_height_screen_px": row["original_d4_height_screen_px"],
                "green_90_energy_boundary_x_um": row["energy_boundary_x_um"],
                "green_90_energy_boundary_y_um": row["energy_boundary_y_um"],
                "green_center_x_screen_px": row["png90_centroid_x_screen"],
                "green_center_y_screen_px": row["png90_centroid_y_screen"],
                "green_width_screen_px": row["energy_boundary_width_screen_px"],
                "green_height_screen_px": row["energy_boundary_height_screen_px"],
                "green_energy_fraction": row["png90_energy_fraction"],
                "magenta_inferred90_d4x_um": row["inferred90_d4x_um"],
                "magenta_inferred90_d4y_um": row["inferred90_d4y_um"],
                "magenta_center_x_screen_px": row["png90_centroid_x_screen"],
                "magenta_center_y_screen_px": row["png90_centroid_y_screen"],
                "magenta_width_screen_px": row["inferred90_width_screen_px"],
                "magenta_height_screen_px": row["inferred90_height_screen_px"],
                "png_shape_ratio_x": row["png90_to_png_original_ratio_x"],
                "png_shape_ratio_y": row["png90_to_png_original_ratio_y"],
                "diagnostic_png_abs90_d4x_um": row["png90_d4x_um"],
                "diagnostic_png_abs90_d4y_um": row["png90_d4y_um"],
            }
        )
    contour_fields = [
        "z_mm",
        "cyan_csv_original_d4x_um",
        "cyan_csv_original_d4y_um",
        "cyan_center_x_screen_px",
        "cyan_center_y_screen_px",
        "cyan_width_screen_px",
        "cyan_height_screen_px",
        "green_90_energy_boundary_x_um",
        "green_90_energy_boundary_y_um",
        "green_center_x_screen_px",
        "green_center_y_screen_px",
        "green_width_screen_px",
        "green_height_screen_px",
        "green_energy_fraction",
        "magenta_inferred90_d4x_um",
        "magenta_inferred90_d4y_um",
        "magenta_center_x_screen_px",
        "magenta_center_y_screen_px",
        "magenta_width_screen_px",
        "magenta_height_screen_px",
        "png_shape_ratio_x",
        "png_shape_ratio_y",
        "diagnostic_png_abs90_d4x_um",
        "diagnostic_png_abs90_d4y_um",
    ]
    write_csv(OUT_DIR / "beam_contour_layers_xy.csv", contour_rows, contour_fields)

    z = np.array([float(row["z_mm"]) for row in per_position])
    series = {
        "csv_original": {
            "x": np.array([float(row["csv_d4x_um"]) for row in per_position]),
            "y": np.array([float(row["csv_d4y_um"]) for row in per_position]),
        },
        "png90": {
            "x": np.array([float(row["png90_d4x_um"]) for row in per_position]),
            "y": np.array([float(row["png90_d4y_um"]) for row in per_position]),
        },
        "png90_shape_inferred": {
            "x": np.array([float(row["inferred90_d4x_um"]) for row in per_position]),
            "y": np.array([float(row["inferred90_d4y_um"]) for row in per_position]),
        },
        "png_original_moment_check": {
            "x": np.array([float(row["png_original_moment_d4x_um"]) for row in per_position]),
            "y": np.array([float(row["png_original_moment_d4y_um"]) for row in per_position]),
        },
    }
    fits: dict[str, dict[str, dict[str, float | np.ndarray]]] = {}
    for name, data in series.items():
        fits[name] = {"x": fit_m2(z, data["x"]), "y": fit_m2(z, data["y"])}

    fit_rows: list[dict[str, object]] = []
    for name in ["csv_original", "png90_shape_inferred", "png90", "png_original_moment_check"]:
        for direction in ["x", "y"]:
            fit = fits[name][direction]
            csv_fit = fits["csv_original"][direction]
            m2 = float(fit["m2"])
            csv_m2 = float(csv_fit["m2"])
            fit_rows.append(
                {
                    "dataset": name,
                    "direction": direction,
                    "wavelength_um": WAVELENGTH_UM,
                    "camera_pixel_size_um": CAMERA_PIXEL_SIZE_UM,
                    "d0_um": fit["d0_um"],
                    "theta_um_per_mm": fit["theta_um_per_mm"],
                    "z0_mm": fit["z0_mm"],
                    "m2": m2,
                    "rmse_um": fit["rmse_um"],
                    "r2": fit["r2"],
                    "delta_m2_vs_csv_original": m2 - csv_m2,
                    "delta_pct_vs_csv_original": 100.0 * (m2 / csv_m2 - 1.0),
                }
            )
    fit_fields = [
        "dataset",
        "direction",
        "wavelength_um",
        "camera_pixel_size_um",
        "d0_um",
        "theta_um_per_mm",
        "z0_mm",
        "m2",
        "rmse_um",
        "r2",
        "delta_m2_vs_csv_original",
        "delta_pct_vs_csv_original",
    ]
    write_csv(OUT_DIR / "beam_m2_fit_summary.csv", fit_rows, fit_fields)
    plot_fits(fits, z, series)

    report = OUT_DIR / "beam_m2_report.md"
    with report.open("w", encoding="utf-8") as f:
        f.write("# Beam M2 analysis\n\n")
        f.write(f"- Wavelength used for numeric M2: {WAVELENGTH_UM:.4g} um.\n")
        f.write(f"- Camera pixel size used for PNG geometry: {CAMERA_PIXEL_SIZE_UM:.4g} um/pixel.\n")
        f.write("- Folder names were treated as z positions in mm.\n")
        f.write("- Original D4sigma diameters and original M2 come from the CSV result files.\n")
        f.write("- PNG processing uses the screenshot colorbar, masks/fills yellow and white overlays, and evaluates 90% energy inside a 1.5x original D4sigma elliptical domain.\n")
        f.write("- The final 90% D4sigma uses a CSV-anchored shape ratio: CSV original D4sigma * (PNG 90% D4sigma / PNG original moment D4sigma).\n")
        f.write("- The cyan ellipse in annotated PNGs is the CSV-derived original D4sigma. The green dashed ellipse is the 90% energy boundary. The magenta ellipse is the CSV-anchored 90% D4sigma estimate.\n\n")
        f.write("## Key M2 results\n\n")
        f.write("| Dataset | X M2 | Y M2 | X change vs CSV | Y change vs CSV |\n")
        f.write("|---|---:|---:|---:|---:|\n")
        for name in ["csv_original", "png90_shape_inferred", "png90", "png_original_moment_check"]:
            fx = fits[name]["x"]
            fy = fits[name]["y"]
            csv_x = float(fits["csv_original"]["x"]["m2"])
            csv_y = float(fits["csv_original"]["y"]["m2"])
            mx = float(fx["m2"])
            my = float(fy["m2"])
            f.write(
                f"| {name} | {mx:.4f} | {my:.4f} | {mx - csv_x:+.4f} ({100.0 * (mx / csv_x - 1.0):+.2f}%) | "
                f"{my - csv_y:+.4f} ({100.0 * (my / csv_y - 1.0):+.2f}%) |\n"
            )


if __name__ == "__main__":
    main()
