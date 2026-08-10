from __future__ import annotations

"""Design a cylindrical-lens correction from D4-sigma caustic CSV files.

The model uses second-moment beam matrices.  It therefore preserves the fitted
M2 invariant of each transverse axis and does not assume an ideal Gaussian beam.

Default usage (from the repository root):

    python cylindrical_lens_design.py

The script reads DataBase/data/*/*.results.csv and writes a reproducible design
package under outputs/cylindrical_lens_d4sigma_20260810.
"""

import argparse
import csv
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit, minimize_scalar


@dataclass
class FitResult:
    axis: str
    d0_um: float
    theta_um_per_mm: float
    z0_mm: float
    m2: float
    rmse_um: float
    r2: float
    d0_std_um: float
    theta_std_um_per_mm: float
    z0_std_mm: float


def beam_diameter(z_mm: np.ndarray | float, d0_um: float, theta_um_per_mm: float, z0_mm: float):
    z = np.asarray(z_mm, dtype=float)
    return np.sqrt(d0_um**2 + (theta_um_per_mm * (z - z0_mm)) ** 2)


def read_source_data(data_dir: Path) -> list[dict[str, float | str]]:
    rows: list[dict[str, float | str]] = []
    paths = sorted(data_dir.glob("*/*.results.csv"), key=lambda p: float(p.parent.name))
    if not paths:
        raise FileNotFoundError(f"No *.results.csv files found below {data_dir}")
    for path in paths:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            records = list(csv.reader(handle))
        if len(records) < 2 or len(records[1]) < 9:
            raise ValueError(f"Expected at least 9 CSV columns and one data row: {path}")
        values = records[1]
        dx, dy = float(values[7]), float(values[8])
        rows.append(
            {
                "z_mm": float(path.parent.name),
                "d4x_um": dx,
                "d4y_um": dy,
                "ellipticity": max(dx, dy) / min(dx, dy),
                "circularity": min(dx, dy) / max(dx, dy),
                "source_file": str(path.as_posix()),
            }
        )
    return rows


def fit_axis(axis: str, z: np.ndarray, d: np.ndarray, wavelength_um: float) -> FitResult:
    p0 = [float(d.min()), 40.0, float(z[np.argmin(d)])]
    params, covariance = curve_fit(
        beam_diameter,
        z,
        d,
        p0=p0,
        bounds=([1.0, 0.01, z.min() - 100.0], [20_000.0, 500.0, z.max() + 100.0]),
        maxfev=100_000,
    )
    fitted = beam_diameter(z, *params)
    residual = d - fitted
    rmse = float(np.sqrt(np.mean(residual**2)))
    ss_res = float(np.sum(residual**2))
    ss_tot = float(np.sum((d - d.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot
    std = np.sqrt(np.diag(covariance))
    d0, theta, z0 = [float(v) for v in params]
    # theta [um/mm] equals mrad numerically; convert it to rad with /1000.
    m2 = math.pi * d0 * theta / (4000.0 * wavelength_um)
    return FitResult(axis, d0, theta, z0, m2, rmse, r2, *[float(v) for v in std])


def beam_matrix_at(fit: FitResult, z_mm: float) -> np.ndarray:
    dz = z_mm - fit.z0_mm
    theta2 = fit.theta_um_per_mm**2
    return np.array(
        [
            [fit.d0_um**2 + theta2 * dz**2, theta2 * dz],
            [theta2 * dz, theta2],
        ],
        dtype=float,
    )


def apply_thin_lens(matrix: np.ndarray, focal_length_mm: float) -> np.ndarray:
    lens = np.array([[1.0, 0.0], [-1.0 / focal_length_mm, 1.0]])
    return lens @ matrix @ lens.T


def propagated_diameter(matrix_after_lens: np.ndarray, distance_mm: np.ndarray | float):
    t = np.asarray(distance_mm, dtype=float)
    value = matrix_after_lens[0, 0] + 2.0 * t * matrix_after_lens[0, 1] + t**2 * matrix_after_lens[1, 1]
    return np.sqrt(np.maximum(value, 0.0))


def waist_from_matrix(matrix: np.ndarray, lens_z_mm: float) -> tuple[float, float, float]:
    b, c = float(matrix[0, 1]), float(matrix[1, 1])
    waist_z = lens_z_mm - b / c
    waist_d = math.sqrt(max(float(np.linalg.det(matrix)) / c, 0.0))
    return waist_d, math.sqrt(c), waist_z


def ellipticity(dx: np.ndarray, dy: np.ndarray) -> np.ndarray:
    return np.maximum(dx, dy) / np.minimum(dx, dy)


def solve_curve_match(source: FitResult, target_d0: float, target_theta: float, target_z0: float, upstream_of: float):
    """Solve lens position/power that maps source to a requested caustic.

    A thin lens does not change the beam diameter at its own plane.  Therefore
    the source and requested target A matrix elements must be equal there.  The
    equality is quadratic in lens position; the lens power then follows from B.
    """

    sx2 = source.theta_um_per_mm**2
    tt2 = target_theta**2
    # (sx2-tt2) z^2 + ... = 0
    coeff = [
        sx2 - tt2,
        -2.0 * sx2 * source.z0_mm + 2.0 * tt2 * target_z0,
        source.d0_um**2 + sx2 * source.z0_mm**2 - target_d0**2 - tt2 * target_z0**2,
    ]
    roots = np.roots(coeff)
    candidates: list[tuple[float, float]] = []
    for root in roots:
        if abs(root.imag) > 1e-7:
            continue
        z_lens = float(root.real)
        if z_lens >= upstream_of:
            continue
        a = source.d0_um**2 + sx2 * (z_lens - source.z0_mm) ** 2
        b_source = sx2 * (z_lens - source.z0_mm)
        b_target = tt2 * (z_lens - target_z0)
        power = (b_source - b_target) / a
        if power > 0:
            candidates.append((z_lens, 1.0 / power))
    if not candidates:
        raise RuntimeError("No positive cylindrical-lens solution exists within the requested upstream region.")
    return max(candidates, key=lambda item: item[0])


def optimize_position_for_fixed_f(
    fit_x: FitResult,
    fit_y: FitResult,
    focal_length_mm: float,
    band_min_mm: float,
    band_max_mm: float,
) -> float:
    grid = np.linspace(band_min_mm, band_max_mm, 1201)
    dy = beam_diameter(grid, fit_y.d0_um, fit_y.theta_um_per_mm, fit_y.z0_mm)

    def objective(z_lens: float) -> float:
        gx = apply_thin_lens(beam_matrix_at(fit_x, z_lens), focal_length_mm)
        dx = propagated_diameter(gx, grid - z_lens)
        return float(np.max(np.abs(np.log(dx / dy))))

    result = minimize_scalar(
        objective,
        bounds=(band_min_mm - 50.0, band_min_mm - 0.1),
        method="bounded",
        options={"xatol": 1e-10},
    )
    return float(result.x)


def optimize_position_for_focus(
    fit_x: FitResult,
    fit_y: FitResult,
    focal_length_mm: float,
    target_z_mm: float,
    preferred_z_mm: float,
) -> float:
    target_d = float(beam_diameter(target_z_mm, fit_y.d0_um, fit_y.theta_um_per_mm, fit_y.z0_mm))

    def objective(z_lens: float) -> float:
        gx = apply_thin_lens(beam_matrix_at(fit_x, z_lens), focal_length_mm)
        dx = float(propagated_diameter(gx, target_z_mm - z_lens))
        return abs(math.log(dx / target_d))

    # More than one lens position can yield the same diameter at one plane.
    # Select the branch nearest the full point-focus solution so the corrected
    # X waist also stays at the intended focal plane.
    result = minimize_scalar(
        objective,
        bounds=(preferred_z_mm - 2.0, preferred_z_mm + 2.0),
        method="bounded",
        options={"xatol": 1e-10},
    )
    return float(result.x)


def write_csv(path: Path, rows: list[dict], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fields = fieldnames or list(rows[0].keys())
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def set_plot_style() -> None:
    plt.rcParams.update(
        {
            "font.sans-serif": ["Microsoft YaHei", "SimHei", "DejaVu Sans"],
            "axes.unicode_minus": False,
            "figure.dpi": 140,
            "savefig.dpi": 180,
            "axes.grid": True,
            "grid.alpha": 0.22,
        }
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("DataBase/data"))
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/cylindrical_lens_d4sigma_20260810"))
    parser.add_argument("--wavelength-um", type=float, default=10.6)
    parser.add_argument("--band-min-mm", type=float, default=220.0)
    parser.add_argument("--band-max-mm", type=float, default=250.0)
    parser.add_argument("--standard-band-f-mm", type=float, default=50.0)
    parser.add_argument("--standard-focus-f-mm", type=float, default=25.4)
    args = parser.parse_args()

    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    source_rows = read_source_data(args.data_dir)
    z = np.array([row["z_mm"] for row in source_rows], dtype=float)
    d4x = np.array([row["d4x_um"] for row in source_rows], dtype=float)
    d4y = np.array([row["d4y_um"] for row in source_rows], dtype=float)
    fit_x = fit_axis("X", z, d4x, args.wavelength_um)
    fit_y = fit_axis("Y", z, d4y, args.wavelength_um)

    # The best constant ellipticity attainable with one lossless cylindrical
    # lens follows from the invariant M2 ratio.
    uniform_ratio = math.sqrt(fit_x.m2 / fit_y.m2)
    target_d0 = uniform_ratio * fit_y.d0_um
    target_theta = uniform_ratio * fit_y.theta_um_per_mm
    ideal_z, ideal_f = solve_curve_match(
        fit_x, target_d0, target_theta, fit_y.z0_mm, upstream_of=args.band_min_mm
    )

    standard_z = optimize_position_for_fixed_f(
        fit_x, fit_y, args.standard_band_f_mm, args.band_min_mm, args.band_max_mm
    )
    standard_matrix = apply_thin_lens(beam_matrix_at(fit_x, standard_z), args.standard_band_f_mm)
    standard_d0, standard_theta, standard_z0 = waist_from_matrix(standard_matrix, standard_z)

    # Alternative: exact roundness at the Y waist, accepting a steeper X
    # divergence outside the immediate focus neighborhood.
    point_theta = fit_x.d0_um * fit_x.theta_um_per_mm / fit_y.d0_um
    point_ideal_z, point_ideal_f = solve_curve_match(
        fit_x, fit_y.d0_um, point_theta, fit_y.z0_mm, upstream_of=fit_y.z0_mm
    )
    point_standard_z = optimize_position_for_focus(
        fit_x, fit_y, args.standard_focus_f_mm, fit_y.z0_mm, point_ideal_z
    )
    point_matrix = apply_thin_lens(
        beam_matrix_at(fit_x, point_standard_z), args.standard_focus_f_mm
    )
    point_d0, point_divergence, point_z0 = waist_from_matrix(point_matrix, point_standard_z)

    fine_z = np.linspace(min(z.min(), standard_z), z.max(), 481)
    baseline_x = beam_diameter(fine_z, fit_x.d0_um, fit_x.theta_um_per_mm, fit_x.z0_mm)
    baseline_y = beam_diameter(fine_z, fit_y.d0_um, fit_y.theta_um_per_mm, fit_y.z0_mm)
    robust_x = np.where(
        fine_z >= standard_z,
        propagated_diameter(standard_matrix, fine_z - standard_z),
        np.nan,
    )
    point_x = np.where(
        fine_z >= point_standard_z,
        propagated_diameter(point_matrix, fine_z - point_standard_z),
        np.nan,
    )
    base_e = ellipticity(baseline_x, baseline_y)
    robust_e = ellipticity(robust_x, baseline_y)
    point_e = ellipticity(point_x, baseline_y)

    band_mask = (fine_z >= args.band_min_mm) & (fine_z <= args.band_max_mm)
    focus_window_mask = (fine_z >= fit_y.z0_mm - 10.0) & (fine_z <= fit_y.z0_mm + 10.0)
    base_band_max = float(np.nanmax(base_e[band_mask]))
    robust_band_max = float(np.nanmax(robust_e[band_mask]))
    point_window_max = float(np.nanmax(point_e[focus_window_mask]))
    target_idx = int(np.argmin(np.abs(fine_z - fit_y.z0_mm)))

    fit_rows = [asdict(fit_x), asdict(fit_y)]
    design_rows = [
        {
            "design": "理论整段最优",
            "focal_length_mm": ideal_f,
            "lens_z_mm": ideal_z,
            "powered_direction": "X",
            "cylinder_axis_direction": "Y",
            "new_x_waist_d4_um": target_d0,
            "new_x_divergence_um_per_mm": target_theta,
            "new_x_waist_z_mm": fit_y.z0_mm,
            "evaluation_window_mm": f"{args.band_min_mm:g}-{args.band_max_mm:g}",
            "max_ellipticity": uniform_ratio,
            "min_circularity": 1.0 / uniform_ratio,
        },
        {
            "design": "推荐标准件_整段均衡",
            "focal_length_mm": args.standard_band_f_mm,
            "lens_z_mm": standard_z,
            "powered_direction": "X",
            "cylinder_axis_direction": "Y",
            "new_x_waist_d4_um": standard_d0,
            "new_x_divergence_um_per_mm": standard_theta,
            "new_x_waist_z_mm": standard_z0,
            "evaluation_window_mm": f"{args.band_min_mm:g}-{args.band_max_mm:g}",
            "max_ellipticity": robust_band_max,
            "min_circularity": 1.0 / robust_band_max,
        },
        {
            "design": "理论单点圆斑",
            "focal_length_mm": point_ideal_f,
            "lens_z_mm": point_ideal_z,
            "powered_direction": "X",
            "cylinder_axis_direction": "Y",
            "new_x_waist_d4_um": fit_y.d0_um,
            "new_x_divergence_um_per_mm": point_theta,
            "new_x_waist_z_mm": fit_y.z0_mm,
            "evaluation_window_mm": f"z={fit_y.z0_mm:.3f}",
            "max_ellipticity": 1.0,
            "min_circularity": 1.0,
        },
        {
            "design": "备选标准件_单点圆斑",
            "focal_length_mm": args.standard_focus_f_mm,
            "lens_z_mm": point_standard_z,
            "powered_direction": "X",
            "cylinder_axis_direction": "Y",
            "new_x_waist_d4_um": point_d0,
            "new_x_divergence_um_per_mm": point_divergence,
            "new_x_waist_z_mm": point_z0,
            "evaluation_window_mm": f"{fit_y.z0_mm-10:.1f}-{fit_y.z0_mm+10:.1f}",
            "max_ellipticity": point_window_max,
            "min_circularity": 1.0 / point_window_max,
        },
    ]

    prediction_rows: list[dict] = []
    for i, zi in enumerate(fine_z):
        prediction_rows.append(
            {
                "z_mm": float(zi),
                "baseline_d4x_um": float(baseline_x[i]),
                "baseline_d4y_um": float(baseline_y[i]),
                "baseline_ellipticity": float(base_e[i]),
                "robust_d4x_um": None if np.isnan(robust_x[i]) else float(robust_x[i]),
                "robust_d4y_um": float(baseline_y[i]),
                "robust_ellipticity": None if np.isnan(robust_e[i]) else float(robust_e[i]),
                "point_d4x_um": None if np.isnan(point_x[i]) else float(point_x[i]),
                "point_d4y_um": float(baseline_y[i]),
                "point_ellipticity": None if np.isnan(point_e[i]) else float(point_e[i]),
            }
        )

    sensitivity_rows: list[dict] = []
    for dz in np.arange(-2.0, 2.01, 0.5):
        for focal_pct in np.arange(-0.02, 0.0201, 0.01):
            trial_z = standard_z + float(dz)
            trial_f = args.standard_band_f_mm * (1.0 + float(focal_pct))
            trial_matrix = apply_thin_lens(beam_matrix_at(fit_x, trial_z), trial_f)
            band_z = np.linspace(args.band_min_mm, args.band_max_mm, 601)
            trial_x = propagated_diameter(trial_matrix, band_z - trial_z)
            trial_y = beam_diameter(band_z, fit_y.d0_um, fit_y.theta_um_per_mm, fit_y.z0_mm)
            trial_e = ellipticity(trial_x, trial_y)
            sensitivity_rows.append(
                {
                    "lens_position_error_mm": float(dz),
                    "focal_length_error_pct": float(focal_pct * 100.0),
                    "max_ellipticity_220_250": float(trial_e.max()),
                    "mean_ellipticity_220_250": float(trial_e.mean()),
                }
            )

    write_csv(out_dir / "source_d4sigma_consolidated.csv", source_rows)
    write_csv(out_dir / "fit_summary.csv", fit_rows)
    write_csv(out_dir / "design_summary.csv", design_rows)
    write_csv(out_dir / "model_curves.csv", prediction_rows)
    write_csv(out_dir / "sensitivity.csv", sensitivity_rows)

    summary = {
        "assumptions": {
            "wavelength_um": args.wavelength_um,
            "z_unit": "mm",
            "diameter_unit": "um",
            "lens_model": "thin cylindrical lens; X axis powered; free-space propagation after lens",
            "evaluation_band_mm": [args.band_min_mm, args.band_max_mm],
        },
        "fits": {"X": asdict(fit_x), "Y": asdict(fit_y)},
        "baseline": {
            "max_ellipticity_in_band": base_band_max,
            "min_circularity_in_band": 1.0 / base_band_max,
            "m2_ratio_x_over_y": fit_x.m2 / fit_y.m2,
            "one_cylindrical_lens_uniform_ellipticity_lower_bound": uniform_ratio,
        },
        "recommended_standard_design": design_rows[1],
        "point_focus_alternative": design_rows[3],
    }
    (out_dir / "design_results.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    set_plot_style()
    fig, ax = plt.subplots(figsize=(9.2, 5.4))
    ax.scatter(z, d4x, color="#D55E00", marker="o", label="原始测量 D4σX")
    ax.scatter(z, d4y, color="#0072B2", marker="s", label="原始测量 D4σY")
    ax.plot(fine_z, baseline_x, color="#D55E00", alpha=0.55, lw=1.8, label="原始 X 拟合")
    ax.plot(fine_z, baseline_y, color="#0072B2", alpha=0.75, lw=1.8, label="Y 拟合（柱面镜后不变）")
    ax.plot(fine_z, robust_x, color="#009E73", lw=2.4, label="推荐 +50 mm 后 X")
    ax.axvline(standard_z, color="#009E73", ls="--", lw=1.2, label=f"柱面镜 z={standard_z:.2f} mm")
    ax.axvspan(args.band_min_mm, args.band_max_mm, color="#009E73", alpha=0.07)
    ax.set(xlabel="z 位置 (mm)", ylabel="D4σ 直径 (µm)", title="D4σ 焦散：原始数据与推荐柱面镜方案")
    ax.legend(ncol=2, fontsize=8.5)
    fig.tight_layout()
    fig.savefig(out_dir / "caustic_comparison.png", bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9.2, 5.0))
    ax.plot(fine_z, base_e, color="#777777", lw=2.0, label="原始拟合")
    ax.plot(fine_z, robust_e, color="#009E73", lw=2.5, label="推荐 +50 mm（整段均衡）")
    ax.plot(fine_z, point_e, color="#CC79A7", lw=1.8, label="+25.4 mm（焦点单点）")
    ax.axhline(uniform_ratio, color="#E69F00", ls="--", lw=1.4, label=f"单柱面镜整段理论下限 {uniform_ratio:.3f}")
    ax.axvspan(args.band_min_mm, args.band_max_mm, color="#009E73", alpha=0.07)
    ax.set_xlim(args.band_min_mm - 5, args.band_max_mm + 10)
    ax.set_ylim(0.98, max(1.85, float(np.nanmax(base_e[(fine_z >= args.band_min_mm - 5) & (fine_z <= args.band_max_mm + 10)])) * 1.03))
    ax.set(xlabel="z 位置 (mm)", ylabel="椭圆率 = 长轴/短轴", title="焦点附近椭圆率对比（越接近 1 越圆）")
    ax.legend(fontsize=8.5)
    fig.tight_layout()
    fig.savefig(out_dir / "ellipticity_comparison.png", bbox_inches="tight")
    plt.close(fig)

    report = f"""# D4σ 柱面镜椭圆率改善方案

## 结论

主推荐是在当前 z 坐标 **{standard_z:.2f} mm** 处加入一片 **+{args.standard_band_f_mm:g} mm 正柱面镜**。让镜片只在相机 X 方向提供光焦度：柱面母线/无光焦度轴沿 Y，曲率方向沿 X。对 10.6 µm 系统，应使用 10.6 µm 增透的 ZnSe（或系统已验证可用的中红外材料），有效口径建议至少 10 mm。

模型预测在 z={args.band_min_mm:g}–{args.band_max_mm:g} mm 内，最大椭圆率从 **{base_band_max:.3f}** 降至 **{robust_band_max:.3f}**，最低圆度从 **{1/base_band_max:.1%}** 提升到 **{1/robust_band_max:.1%}**。推荐标准件把新的 X 束腰置于 {standard_z0:.2f} mm，D4σ 束腰直径约 {standard_d0:.1f} µm；Y 束腰保持 {fit_y.d0_um:.1f} µm。

## 原始拟合

| 轴 | D4σ 束腰 (µm) | 束腰 z (mm) | D4σ 发散斜率 (µm/mm) | M² | RMSE (µm) | R² |
|---|---:|---:|---:|---:|---:|---:|
| X | {fit_x.d0_um:.2f} | {fit_x.z0_mm:.3f} | {fit_x.theta_um_per_mm:.3f} | {fit_x.m2:.4f} | {fit_x.rmse_um:.2f} | {fit_x.r2:.6f} |
| Y | {fit_y.d0_um:.2f} | {fit_y.z0_mm:.3f} | {fit_y.theta_um_per_mm:.3f} | {fit_y.m2:.4f} | {fit_y.rmse_um:.2f} | {fit_y.r2:.6f} |

两轴束腰位置只差 {abs(fit_x.z0_mm-fit_y.z0_mm):.3f} mm，主要问题是束腰尺寸与发散角不匹配，而非明显的焦点轴向分离。

## 为什么单片柱面镜不能让整段都达到 1.00

拟合采用 D4σ 二阶矩焦散：

`D(z)^2 = D0^2 + theta^2 (z-z0)^2`

并用二阶矩束矩阵通过薄透镜 `[[1,0],[-1/f,1]]` 传播。无损一阶光学系统保持每个轴的二阶矩行列式，即保持各轴 M²。原始 M² 比为 {fit_x.m2/fit_y.m2:.4f}，所以用一片柱面镜把两轴束腰位置、焦散形状尽量对齐时，整段恒定轴比的理论下限是：

`sqrt(M2_X/M2_Y) = {uniform_ratio:.4f}`

理论连续参数解为 f={ideal_f:.3f} mm、z={ideal_z:.3f} mm；工程上使用 +50 mm 标准焦距并把位置调整到 {standard_z:.2f} mm，性能只损失约 {robust_band_max-uniform_ratio:.4f} 的轴比。

## 备选：只追求焦点单点圆斑

若只要求 z≈{fit_y.z0_mm:.2f} mm 这一平面接近 1.00，可用 +{args.standard_focus_f_mm:g} mm 柱面镜放在 z≈{point_standard_z:.2f} mm。理论精确解为 f={point_ideal_f:.3f} mm、z={point_ideal_z:.3f} mm。该方案在中心点更圆，但 X 发散被提高，离开焦点后椭圆率增长更快；在焦点 ±10 mm 内预测最差轴比约 {point_window_max:.3f}。

## 安装与调试

1. z 坐标必须与原 CSV 文件夹名采用同一机械零点；镜片主平面放在推荐 z 位置。若实际可安装位置不同，运行脚本时修改目标焦区或在代码参数中重新求解。
2. 需要给较宽的 X 方向增加正光焦度。柱面镜的母线沿 Y、曲率方向沿 X。若设备的 X/Y 与光斑主轴不重合，先旋转镜片，以 D4σ 长轴下降且交叉二阶矩最小为准。
3. 先用 +50 mm 方案，在 z={args.band_min_mm:g}、{fit_y.z0_mm:.1f}、{args.band_max_mm:g} mm 三点复测；以“三区间最大轴比最小”为目标微调镜片 z。模型灵敏度表明，位置 ±1 mm 且焦距误差 ±2% 时，焦区最差轴比仍约不超过 1.30。
4. 镜片应有 10.6 µm AR 镀膜，避免普通可见光玻璃；有效口径 ≥10 mm 可明显高于镜片处约 1 mm 的 D4σ 光束，降低截光风险。
5. 本结果是基于新增镜片后的线性、薄透镜预测，最终需要用相同 D4σ 算法复测验证。CSV 只有 X/Y 边际二阶矩，没有 XY 协方差和主轴角；若光斑主轴有明显旋转，需补充完整二阶矩或旋转角数据后做耦合模型。

## 材料与器件参考

- LASER COMPONENTS：ZnSe 是 10.6 µm CO₂ 激光透射光学的常用材料，柱面镜可询价定制；10.6 µm AR 镀膜规格见 https://www.lasercomponents.com/en/product/znse-lenses-for-co2-lasers/
- 采购时仍需按实际连续/脉冲功率、峰值功率密度、镜片口径与冷却条件核对损伤阈值；本模型没有替代热设计和损伤阈值审核。

## 文件说明

- `source_d4sigma_consolidated.csv`：未改写的原始 D4σ 汇总与原始椭圆率。
- `fit_summary.csv`：两轴拟合、M² 与误差统计。
- `design_summary.csv`：理论解、推荐标准件与单点圆斑备选。
- `model_curves.csv`：绘图和复核用的连续 z 曲线。
- `sensitivity.csv`：推荐方案对镜片位置和焦距误差的灵敏度。
- `design_results.json`：供其他程序读取的结构化结论。
"""
    (out_dir / "方案说明.md").write_text(report, encoding="utf-8")

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
