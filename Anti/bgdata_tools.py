from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


def ensure_h5py() -> Any:
    try:
        import h5py  # type: ignore

        return h5py
    except ModuleNotFoundError:
        local_deps = Path(__file__).resolve().parents[1] / ".codex_deps"
        if local_deps.exists():
            sys.path.insert(0, str(local_deps))
            import h5py  # type: ignore

            return h5py
        raise SystemExit(
            "h5py is required to read .bgData HDF5 files. "
            "Install it with: python -m pip install h5py"
        )


def safe_name(hdf_path: str) -> str:
    text = hdf_path.strip("/") or "root"
    text = re.sub(r"[^A-Za-z0-9_.-]+", "__", text)
    return text[:180]


def sha256_array(values: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(values)
    return hashlib.sha256(contiguous.view(np.uint8)).hexdigest()


def to_jsonable(value: Any, max_array_values: int = 64) -> Any:
    if isinstance(value, bytes):
        for encoding in ("utf-8", "utf-16", "latin1"):
            try:
                return value.decode(encoding).rstrip("\x00")
            except UnicodeDecodeError:
                continue
        return {"__bytes_hex__": value.hex()}
    if isinstance(value, np.bytes_):
        return to_jsonable(bytes(value), max_array_values)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        if value.size <= max_array_values:
            return to_jsonable(value.tolist(), max_array_values)
        return {
            "__array__": True,
            "shape": list(value.shape),
            "dtype": str(value.dtype),
            "sha256": sha256_array(value),
        }
    if isinstance(value, (list, tuple)):
        return [to_jsonable(item, max_array_values) for item in value]
    if isinstance(value, dict):
        return {str(key): to_jsonable(item, max_array_values) for key, item in value.items()}
    return value


def dataset_stats(values: np.ndarray) -> dict[str, Any]:
    stats: dict[str, Any] = {
        "shape": list(values.shape),
        "dtype": str(values.dtype),
        "size": int(values.size),
        "sha256": sha256_array(values),
    }
    if values.size and np.issubdtype(values.dtype, np.number):
        numeric = values.astype("float64", copy=False)
        finite = numeric[np.isfinite(numeric)]
        if finite.size:
            stats.update(
                {
                    "min": float(finite.min()),
                    "max": float(finite.max()),
                    "mean": float(finite.mean()),
                    "sum": float(finite.sum()),
                }
            )
    return stats


def robust_quicklook(values: np.ndarray) -> Image.Image:
    arr = values.astype("float64", copy=False)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        scaled = np.zeros(arr.shape, dtype=np.uint8)
    else:
        lo, hi = np.percentile(finite, [1, 99])
        if not math.isfinite(lo) or not math.isfinite(hi) or hi <= lo:
            lo, hi = float(finite.min()), float(finite.max())
        if hi <= lo:
            scaled = np.zeros(arr.shape, dtype=np.uint8)
        else:
            scaled = np.clip((arr - lo) / (hi - lo), 0.0, 1.0)
            scaled = (scaled * 255.0).astype(np.uint8)
    return Image.fromarray(scaled, mode="L")


def scalar_dataset_value(dataset: Any) -> Any:
    values = dataset[()]
    arr = np.asarray(values)
    if arr.shape == ():
        return to_jsonable(arr.item())
    if arr.size == 1:
        return to_jsonable(arr.reshape(-1)[0])
    return None


def dataset_to_text(dataset: Any) -> str | None:
    values = dataset[()]
    arr = np.asarray(values)
    if arr.dtype.kind not in {"S", "U", "O"}:
        return None
    decoded = to_jsonable(arr)
    if isinstance(decoded, list):
        return "\n".join(str(item) for item in decoded)
    return str(decoded)


def get_scalar(group: Any, rel_path: str, default: Any = None) -> Any:
    if rel_path not in group:
        return default
    arr = np.asarray(group[rel_path][()])
    if arr.size == 0:
        return default
    value = arr.reshape(-1)[0]
    if isinstance(value, np.generic):
        return value.item()
    return value


def extract_frame_if_present(h5file: Any, dataset_path: str, values: np.ndarray, out_dir: Path) -> dict[str, Any] | None:
    parts = dataset_path.strip("/").split("/")
    if len(parts) < 3 or parts[-1] != "DATA" or parts[0] != "BG_DATA":
        return None
    group_path = "/".join(parts[:-1])
    rawframe_path = f"{group_path}/RAWFRAME"
    if rawframe_path not in h5file:
        return None
    rawframe = h5file[rawframe_path]
    width = get_scalar(rawframe, "WIDTH")
    height = get_scalar(rawframe, "HEIGHT")
    if not width or not height:
        return None
    width = int(width)
    height = int(height)
    if values.size != width * height:
        return None

    frame = values.reshape(height, width)
    frames_dir = out_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    stem = safe_name(group_path)
    npy_path = frames_dir / f"{stem}__DATA_raw.npy"
    csv_path = frames_dir / f"{stem}__DATA_raw.csv"
    png_path = frames_dir / f"{stem}__DATA_quicklook.png"
    np.save(npy_path, frame)
    np.savetxt(csv_path, frame, delimiter=",", fmt="%d" if np.issubdtype(frame.dtype, np.integer) else "%.10g")
    robust_quicklook(frame).save(png_path)

    return {
        "dataset": dataset_path,
        "width": width,
        "height": height,
        "pixel_scale_x_um": get_scalar(rawframe, "PIXELSCALEXUM"),
        "pixel_scale_y_um": get_scalar(rawframe, "PIXELSCALEYUM"),
        "energy_of_beam": get_scalar(rawframe, "ENERGY/ENERGYOFBEAM"),
        "energy_of_frame": get_scalar(rawframe, "ENERGY/ENERGYOFFRAME"),
        "exposure_stamp": get_scalar(rawframe, "EXPOSURESTAMP"),
        "gain_stamp": get_scalar(rawframe, "GAINSTAMP"),
        "npy": str(npy_path),
        "csv": str(csv_path),
        "quicklook_png": str(png_path),
        **dataset_stats(frame),
    }


def extract_bgdata_file(path: Path, out_root: Path, export_all_large_arrays: bool = True) -> dict[str, Any]:
    h5py = ensure_h5py()
    file_out = out_root / path.stem
    file_out.mkdir(parents=True, exist_ok=True)
    arrays_dir = file_out / "arrays"
    strings_dir = file_out / "strings"

    summary: dict[str, Any] = {
        "source": str(path),
        "root_attrs": {},
        "groups": [],
        "datasets": [],
        "frames": [],
    }

    with h5py.File(path, "r") as h5:
        summary["root_attrs"] = to_jsonable(dict(h5.attrs))

        def visit(name: str, obj: Any) -> None:
            attrs = to_jsonable(dict(obj.attrs)) if hasattr(obj, "attrs") else {}
            if isinstance(obj, h5py.Group):
                summary["groups"].append({"path": name, "attrs": attrs})
                return

            values = obj[()]
            arr = np.asarray(values)
            entry = {
                "path": name,
                "shape": list(obj.shape),
                "dtype": str(obj.dtype),
                "attrs": attrs,
                "stats": dataset_stats(arr),
            }

            scalar_value = scalar_dataset_value(obj)
            if scalar_value is not None:
                entry["value"] = scalar_value

            text = dataset_to_text(obj)
            if text is not None and len(text) > 200:
                strings_dir.mkdir(parents=True, exist_ok=True)
                text_path = strings_dir / f"{safe_name(name)}.txt"
                text_path.write_text(text, encoding="utf-8", errors="replace")
                entry["text_file"] = str(text_path)

            if arr.size > 64 and export_all_large_arrays:
                arrays_dir.mkdir(parents=True, exist_ok=True)
                array_path = arrays_dir / f"{safe_name(name)}.npy"
                np.save(array_path, arr)
                entry["array_file"] = str(array_path)

            frame_info = extract_frame_if_present(h5, name, arr, file_out)
            if frame_info is not None:
                summary["frames"].append(frame_info)

            summary["datasets"].append(entry)

        h5.visititems(visit)

    metadata_path = file_out / "metadata.json"
    metadata_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    dataset_csv_path = file_out / "datasets.csv"
    with dataset_csv_path.open("w", encoding="utf-8", newline="") as f:
        fieldnames = ["path", "shape", "dtype", "size", "min", "max", "mean", "sum", "sha256", "value", "array_file", "text_file"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for entry in summary["datasets"]:
            stats = entry["stats"]
            writer.writerow(
                {
                    "path": entry["path"],
                    "shape": json.dumps(entry["shape"]),
                    "dtype": entry["dtype"],
                    "size": stats.get("size"),
                    "min": stats.get("min"),
                    "max": stats.get("max"),
                    "mean": stats.get("mean"),
                    "sum": stats.get("sum"),
                    "sha256": stats.get("sha256"),
                    "value": json.dumps(entry.get("value"), ensure_ascii=False),
                    "array_file": entry.get("array_file", ""),
                    "text_file": entry.get("text_file", ""),
                }
            )

    return summary


def build_cross_file_summary(summaries: list[dict[str, Any]], out_root: Path) -> None:
    rows: list[dict[str, Any]] = []
    frame_rows: list[dict[str, Any]] = []
    for summary in summaries:
        source = summary["source"]
        for entry in summary["datasets"]:
            stats = entry["stats"]
            rows.append(
                {
                    "source": source,
                    "path": entry["path"],
                    "shape": json.dumps(entry["shape"]),
                    "dtype": entry["dtype"],
                    "size": stats.get("size"),
                    "min": stats.get("min"),
                    "max": stats.get("max"),
                    "mean": stats.get("mean"),
                    "sum": stats.get("sum"),
                    "sha256": stats.get("sha256"),
                }
            )
        for frame in summary["frames"]:
            frame_rows.append(
                {
                    "source": source,
                    "dataset": frame["dataset"],
                    "width": frame["width"],
                    "height": frame["height"],
                    "pixel_scale_x_um": frame.get("pixel_scale_x_um"),
                    "pixel_scale_y_um": frame.get("pixel_scale_y_um"),
                    "energy_of_beam": frame.get("energy_of_beam"),
                    "energy_of_frame": frame.get("energy_of_frame"),
                    "exposure_stamp": frame.get("exposure_stamp"),
                    "gain_stamp": frame.get("gain_stamp"),
                    "min": frame.get("min"),
                    "max": frame.get("max"),
                    "mean": frame.get("mean"),
                    "sum": frame.get("sum"),
                    "sha256": frame.get("sha256"),
                    "npy": frame.get("npy"),
                    "csv": frame.get("csv"),
                    "quicklook_png": frame.get("quicklook_png"),
                }
            )

    with (out_root / "all_datasets.csv").open("w", encoding="utf-8", newline="") as f:
        fieldnames = ["source", "path", "shape", "dtype", "size", "min", "max", "mean", "sum", "sha256"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    with (out_root / "all_frames.csv").open("w", encoding="utf-8", newline="") as f:
        fieldnames = [
            "source",
            "dataset",
            "width",
            "height",
            "pixel_scale_x_um",
            "pixel_scale_y_um",
            "energy_of_beam",
            "energy_of_frame",
            "exposure_stamp",
            "gain_stamp",
            "min",
            "max",
            "mean",
            "sum",
            "sha256",
            "npy",
            "csv",
            "quicklook_png",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(frame_rows)


def discover_inputs(input_path: Path) -> list[Path]:
    if input_path.is_file():
        return [input_path]
    return sorted(input_path.rglob("*.bgData"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract metadata and datasets from Spiricon/BeamGage .bgData HDF5 files.")
    parser.add_argument("--input", type=Path, default=DATA_DIR, help="A .bgData file or a directory to search recursively.")
    parser.add_argument("--out", type=Path, default=BASE_DIR / "bgdata_extract", help="Output directory.")
    parser.add_argument("--no-large-arrays", action="store_true", help="Skip generic .npy export for large non-frame datasets.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    inputs = discover_inputs(args.input)
    if not inputs:
        raise SystemExit(f"No .bgData files found under {args.input}")
    args.out.mkdir(parents=True, exist_ok=True)
    summaries = []
    for path in inputs:
        print(f"Extracting {path}")
        summaries.append(extract_bgdata_file(path, args.out, export_all_large_arrays=not args.no_large_arrays))
    build_cross_file_summary(summaries, args.out)
    print(f"Done. Wrote extraction to {args.out}")


if __name__ == "__main__":
    main()
