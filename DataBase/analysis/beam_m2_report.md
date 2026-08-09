# Beam M2 analysis

- Wavelength used for numeric M2: 10.6 um.
- Camera pixel size used for PNG geometry: 80 um/pixel.
- Folder names were treated as z positions in mm.
- Original D4sigma diameters and original M2 come from the CSV result files.
- PNG processing uses the screenshot colorbar, masks/fills yellow and white overlays, and evaluates 90% energy inside a 1.5x original D4sigma elliptical domain.
- The final 90% D4sigma uses a CSV-anchored shape ratio: CSV original D4sigma * (PNG 90% D4sigma / PNG original moment D4sigma).
- The cyan ellipse in annotated PNGs is the CSV-derived original D4sigma. The green dashed ellipse is the 90% energy boundary. The magenta ellipse is the CSV-anchored 90% D4sigma estimate.

## Key M2 results

| Dataset | X M2 | Y M2 | X change vs CSV | Y change vs CSV |
|---|---:|---:|---:|---:|
| csv_original | 2.6065 | 1.6552 | +0.0000 (+0.00%) | +0.0000 (+0.00%) |
| png90_shape_inferred | 2.0684 | 1.1971 | -0.5381 (-20.65%) | -0.4581 (-27.68%) |
| png90 | 1.4097 | 1.0542 | -1.1968 (-45.92%) | -0.6010 (-36.31%) |
| png_original_moment_check | 1.7775 | 1.3589 | -0.8290 (-31.81%) | -0.2963 (-17.90%) |
