import fs from "node:fs/promises";
import path from "node:path";
import { SpreadsheetFile, Workbook } from "@oai/artifact-tool";

const repoRoot = path.resolve("..");
const outputDir = path.join(repoRoot, "outputs", "cylindrical_lens_d4sigma_20260810");

function parseCsv(text) {
  const lines = text.replace(/^\uFEFF/, "").trim().split(/\r?\n/);
  const parseLine = (line) => {
    const out = [];
    let item = "";
    let quoted = false;
    for (let i = 0; i < line.length; i++) {
      const ch = line[i];
      if (ch === '"') {
        if (quoted && line[i + 1] === '"') {
          item += '"';
          i++;
        } else {
          quoted = !quoted;
        }
      } else if (ch === "," && !quoted) {
        out.push(item);
        item = "";
      } else {
        item += ch;
      }
    }
    out.push(item);
    return out;
  };
  const headers = parseLine(lines[0]);
  return lines.slice(1).map((line) => {
    const values = parseLine(line);
    return Object.fromEntries(headers.map((h, i) => [h, values[i] ?? ""]));
  });
}

function typed(value) {
  if (value === "" || value === undefined || value === null) return null;
  const number = Number(value);
  return Number.isFinite(number) ? number : value;
}

function colName(index) {
  let n = index + 1;
  let s = "";
  while (n) {
    const r = (n - 1) % 26;
    s = String.fromCharCode(65 + r) + s;
    n = Math.floor((n - 1) / 26);
  }
  return s;
}

const sourceRows = parseCsv(await fs.readFile(path.join(outputDir, "source_d4sigma_consolidated.csv"), "utf8"));
const fitRows = parseCsv(await fs.readFile(path.join(outputDir, "fit_summary.csv"), "utf8"));
const designRows = parseCsv(await fs.readFile(path.join(outputDir, "design_summary.csv"), "utf8"));
const curveRows = parseCsv(await fs.readFile(path.join(outputDir, "model_curves.csv"), "utf8"));
const sensitivityRows = parseCsv(await fs.readFile(path.join(outputDir, "sensitivity.csv"), "utf8"));
const results = JSON.parse(await fs.readFile(path.join(outputDir, "design_results.json"), "utf8"));

const workbook = Workbook.create();
const overview = workbook.worksheets.add("概览");
const raw = workbook.worksheets.add("原始数据");
const params = workbook.worksheets.add("拟合与方案");
const curves = workbook.worksheets.add("模型曲线");
const sensitivity = workbook.worksheets.add("灵敏度");

const navy = "#17324D";
const teal = "#008C7A";
const lightTeal = "#DDF3EF";
const lightBlue = "#E8F0F7";
const orange = "#E69F00";
const rose = "#CC79A7";
const gray = "#667085";
const lightGray = "#F2F4F7";

for (const sheet of [overview, raw, params, curves, sensitivity]) {
  sheet.showGridLines = false;
}

overview.getRange("A1:J1").merge();
overview.getRange("A1").values = [["D4σ 柱面镜椭圆率改善方案"]];
overview.getRange("A1:J1").format = {
  fill: navy,
  font: { bold: true, color: "#FFFFFF", size: 18 },
  horizontalAlignment: "center",
  verticalAlignment: "center",
};
overview.getRange("A1:J1").format.rowHeight = 34;

overview.getRange("A3:B3").values = [["原始状态", "数值"]];
overview.getRange("D3:E3").values = [["推荐标准件方案", "数值"]];
overview.getRange("G3:H3").values = [["物理边界 / 验证", "数值"]];
for (const range of ["A3:B3", "D3:E3", "G3:H3"]) {
  overview.getRange(range).format = { fill: navy, font: { bold: true, color: "#FFFFFF" } };
}
overview.getRange("A4:B8").values = [
  ["X 束腰 D4σ (µm)", results.fits.X.d0_um],
  ["Y 束腰 D4σ (µm)", results.fits.Y.d0_um],
  ["焦区最大椭圆率", results.baseline.max_ellipticity_in_band],
  ["焦区最低圆度", results.baseline.min_circularity_in_band],
  ["X/Y 束腰位置差 (mm)", Math.abs(results.fits.X.z0_mm - results.fits.Y.z0_mm)],
];
overview.getRange("D4:E10").values = [
  ["材料 / 镀膜", "ZnSe / 10.6 µm AR"],
  ["柱面焦距 (mm)", results.recommended_standard_design.focal_length_mm],
  ["镜片主平面 z (mm)", results.recommended_standard_design.lens_z_mm],
  ["有光焦度方向", "X（柱面母线沿 Y）"],
  ["220–250 mm 最大椭圆率", results.recommended_standard_design.max_ellipticity],
  ["220–250 mm 最低圆度", results.recommended_standard_design.min_circularity],
  ["建议有效口径", "≥10 mm"],
];
overview.getRange("G4:H9").values = [
  ["M²X / M²Y", results.baseline.m2_ratio_x_over_y],
  ["单柱面镜整段理论下限", results.baseline.one_cylindrical_lens_uniform_ellipticity_lower_bound],
  ["推荐方案 / 理论下限", results.recommended_standard_design.max_ellipticity / results.baseline.one_cylindrical_lens_uniform_ellipticity_lower_bound],
  ["位置 ±1 mm + 焦距 ±2%", "最差轴比约 ≤1.30"],
  ["焦点单点备选", "+25.4 mm @ z≈222.62 mm"],
  ["核心限制", "单片镜不改变两轴 M²"],
];
for (const range of ["A4:B8", "D4:E10", "G4:H9"]) {
  overview.getRange(range).format = { borders: { preset: "outside", style: "thin", color: "#C8D2DC" } };
}
overview.getRange("B4:B8").format.numberFormat = "0.000";
overview.getRange("E5:E9").format.numberFormat = "0.000";
overview.getRange("H4:H6").format.numberFormat = "0.000";
overview.getRange("B7:B7").format.numberFormat = "0.0%";
overview.getRange("E9:E9").format.numberFormat = "0.0%";
overview.getRange("D4:E10").format.fill = lightTeal;
overview.getRange("G4:H9").format.fill = lightBlue;
overview.getRange("A12:J13").merge();
overview.getRange("A12").values = [["结论：主方案采用 +50 mm 正柱面镜，在 z≈219.43 mm 只对 X 方向增加会聚。预测 220–250 mm 内最大轴比由 1.731 降至 1.256；该值已接近 M² 不变量给出的理论下限 1.255。"]];
overview.getRange("A12:J13").format = {
  fill: "#FFF4D6",
  font: { bold: true, color: "#5B4200" },
  wrapText: true,
  verticalAlignment: "center",
  borders: { preset: "outside", style: "thin", color: orange },
};

const rawHeaders = ["z_mm", "D4σX_µm", "D4σY_µm", "椭圆率", "圆度", "源文件"];
raw.getRange(`A1:${colName(rawHeaders.length - 1)}1`).values = [rawHeaders];
raw.getRange(`A2:F${sourceRows.length + 1}`).values = sourceRows.map((r) => [
  typed(r.z_mm), typed(r.d4x_um), typed(r.d4y_um), null, null, r.source_file,
]);
for (let row = 2; row <= sourceRows.length + 1; row++) {
  raw.getRange(`D${row}:E${row}`).formulas = [[`=MAX(B${row},C${row})/MIN(B${row},C${row})`, `=MIN(B${row},C${row})/MAX(B${row},C${row})`]];
}
raw.tables.add(`A1:F${sourceRows.length + 1}`, true, "RawD4SigmaTable").style = "TableStyleMedium2";
raw.freezePanes.freezeRows(1);
raw.getRange(`A2:E${sourceRows.length + 1}`).format.numberFormat = "0.000";
raw.getRange(`E2:E${sourceRows.length + 1}`).format.numberFormat = "0.0%";

params.getRange("A1:K1").merge();
params.getRange("A1").values = [["拟合参数与柱面镜候选方案"]];
params.getRange("A1:K1").format = { fill: navy, font: { bold: true, color: "#FFFFFF", size: 16 }, horizontalAlignment: "center" };
params.getRange("A3:B4").values = [["假设", "数值"], ["波长 (µm)", results.assumptions.wavelength_um]];
params.getRange("A3:B3").format = { fill: navy, font: { bold: true, color: "#FFFFFF" } };
const fitHeaders = ["轴", "D0_µm", "theta_µm/mm", "z0_mm", "M²", "RMSE_µm", "R²", "D0标准差", "theta标准差", "z0标准差"];
params.getRange("A6:J6").values = [fitHeaders];
params.getRange("A7:J8").values = fitRows.map((r) => [
  r.axis, typed(r.d0_um), typed(r.theta_um_per_mm), typed(r.z0_mm), typed(r.m2), typed(r.rmse_um), typed(r.r2), typed(r.d0_std_um), typed(r.theta_std_um_per_mm), typed(r.z0_std_mm),
]);
params.getRange("A6:J6").format = { fill: navy, font: { bold: true, color: "#FFFFFF" } };
params.getRange("B7:J8").format.numberFormat = "0.0000";
const designHeaders = ["方案", "f_mm", "镜片z_mm", "有光焦度方向", "柱面母线方向", "新X束腰_µm", "新X发散_µm/mm", "新X束腰z_mm", "评价窗口", "最大椭圆率", "最低圆度"];
params.getRange("A11:K11").values = [designHeaders];
params.getRange("A12:K15").values = designRows.map((r) => [
  r.design, typed(r.focal_length_mm), typed(r.lens_z_mm), r.powered_direction, r.cylinder_axis_direction,
  typed(r.new_x_waist_d4_um), typed(r.new_x_divergence_um_per_mm), typed(r.new_x_waist_z_mm), r.evaluation_window_mm,
  typed(r.max_ellipticity), typed(r.min_circularity),
]);
params.getRange("A11:K11").format = { fill: navy, font: { bold: true, color: "#FFFFFF" }, wrapText: true };
params.getRange("A13:K13").format.fill = lightTeal;
params.getRange("A15:K15").format.fill = "#F9E8F3";
params.getRange("B12:H15").format.numberFormat = "0.000";
params.getRange("J12:J15").format.numberFormat = "0.000";
params.getRange("K12:K15").format.numberFormat = "0.0%";
params.getRange("A17:K20").merge();
params.getRange("A17").values = [["模型说明：D(z)^2=D0^2+theta^2(z-z0)^2；薄柱面镜在 X 轴使用 [ [1,0],[-1/f,1] ] 变换。单片无损一阶元件保持每个轴的二阶矩行列式，因此无法改变 M²X/M²Y。整段恒定轴比下限为 SQRT(M²X/M²Y)。"]];
params.getRange("A17:K20").format = { fill: lightGray, wrapText: true, verticalAlignment: "center", borders: { preset: "outside", style: "thin", color: "#C8D2DC" } };
params.getRange("A22:B23").values = [
  ["外部材料参考", "说明 / URL"],
  ["ZnSe CO₂ 激光柱面镜与 10.6 µm AR", "https://www.lasercomponents.com/en/product/znse-lenses-for-co2-lasers/"],
];
params.getRange("B23:K23").merge();
params.getRange("A22:B22").format = { fill: navy, font: { bold: true, color: "#FFFFFF" } };
params.getRange("A23:K23").format = { fill: lightBlue, wrapText: true };

const curveHeaders = ["z_mm", "原始X_µm", "原始Y_µm", "原始椭圆率", "推荐X_µm", "推荐Y_µm", "推荐椭圆率", "单点X_µm", "单点Y_µm", "单点椭圆率"];
curves.getRange("A1:J1").values = [curveHeaders];
const nCurve = curveRows.length;
curves.getRange(`A2:A${nCurve + 1}`).values = curveRows.map((r) => [typed(r.z_mm)]);
for (let row = 2; row <= nCurve + 1; row++) {
  curves.getRange(`B${row}:J${row}`).formulas = [[
    `=SQRT('拟合与方案'!$B$7^2+('拟合与方案'!$C$7*(A${row}-'拟合与方案'!$D$7))^2)`,
    `=SQRT('拟合与方案'!$B$8^2+('拟合与方案'!$C$8*(A${row}-'拟合与方案'!$D$8))^2)`,
    `=MAX(B${row},C${row})/MIN(B${row},C${row})`,
    `=IF(A${row}<'拟合与方案'!$C$13,"",SQRT('拟合与方案'!$F$13^2+('拟合与方案'!$G$13*(A${row}-'拟合与方案'!$H$13))^2))`,
    `=C${row}`,
    `=IF(E${row}="","",MAX(E${row},F${row})/MIN(E${row},F${row}))`,
    `=IF(A${row}<'拟合与方案'!$C$15,"",SQRT('拟合与方案'!$F$15^2+('拟合与方案'!$G$15*(A${row}-'拟合与方案'!$H$15))^2))`,
    `=C${row}`,
    `=IF(H${row}="","",MAX(H${row},I${row})/MIN(H${row},I${row}))`,
  ]];
}
curves.tables.add(`A1:J${nCurve + 1}`, true, "ModelCurveTable").style = "TableStyleMedium2";
curves.freezePanes.freezeRows(1);
curves.getRange(`A2:J${nCurve + 1}`).format.numberFormat = "0.000";

const sensitivityHeaders = ["位置误差_mm", "焦距误差_%", "220–250最大椭圆率", "220–250平均椭圆率"];
sensitivity.getRange("A1:D1").values = [sensitivityHeaders];
sensitivity.getRange(`A2:D${sensitivityRows.length + 1}`).values = sensitivityRows.map((r) => [
  typed(r.lens_position_error_mm), typed(r.focal_length_error_pct), typed(r.max_ellipticity_220_250), typed(r.mean_ellipticity_220_250),
]);
sensitivity.tables.add(`A1:D${sensitivityRows.length + 1}`, true, "SensitivityTable").style = "TableStyleMedium2";
sensitivity.freezePanes.freezeRows(1);
sensitivity.getRange(`A2:D${sensitivityRows.length + 1}`).format.numberFormat = "0.000";
sensitivity.getRange(`C2:C${sensitivityRows.length + 1}`).conditionalFormats.add("colorScale", {
  colors: ["#DDF3EF", "#FFF4D6", "#F8D7DA"],
  thresholds: ["min", "50%", "max"],
});

for (const sheet of [raw, params, curves, sensitivity]) {
  const used = sheet.getUsedRange();
  used.format.font = { name: "Aptos", size: 10 };
  used.format.autofitColumns();
  used.format.autofitRows();
}
raw.getRange("A:A").format.columnWidth = 12;
raw.getRange("B:E").format.columnWidth = 14;
raw.getRange("F:F").format.columnWidth = 48;
params.getRange("A:A").format.columnWidth = 29;
params.getRange("B:B").format.columnWidth = 13;
params.getRange("C:C").format.columnWidth = 17;
params.getRange("D:E").format.columnWidth = 18;
params.getRange("F:H").format.columnWidth = 20;
params.getRange("I:I").format.columnWidth = 19;
params.getRange("J:K").format.columnWidth = 16;
params.getRange("B:B").format.columnWidth = 13;
curves.getRange("A:J").format.columnWidth = 15;
sensitivity.getRange("A:D").format.columnWidth = 20;
overview.getRange("A:J").format.columnWidth = 15;
overview.getRange("A:A").format.columnWidth = 24;
overview.getRange("D:D").format.columnWidth = 25;
overview.getRange("G:G").format.columnWidth = 26;
overview.getRange("E:E").format.columnWidth = 22;
overview.getRange("H:H").format.columnWidth = 23;

const caustic = overview.charts.add("line", { chartType: "line", title: "D4σ 焦散对比 (µm)", hasLegend: true });
for (const [name, column, color] of [["原始 X", "B", "#D55E00"], ["原始 Y", "C", "#0072B2"], ["推荐后 X", "E", teal]]) {
  const series = caustic.series.add(name);
  series.categoryFormula = `'模型曲线'!$A$2:$A$${nCurve + 1}`;
  series.formula = `'模型曲线'!$${column}$2:$${column}$${nCurve + 1}`;
  series.fill = color;
}
caustic.setPosition("A15", "E33");
caustic.xAxis = { axisType: "textAxis", tickLabelInterval: 20, textStyle: { fontSize: 9 } };
caustic.yAxis = { numberFormatCode: "0", textStyle: { fontSize: 9 } };

const ellip = overview.charts.add("line", { chartType: "line", title: "焦点附近椭圆率（长轴/短轴）", hasLegend: true });
const focusStartRow = 182; // z=210 mm on the 0.5 mm model grid
const focusEndRow = 302;   // z=270 mm
for (const [name, column, color] of [["原始", "D", gray], ["推荐 +50 mm", "G", teal], ["单点 +25.4 mm", "J", rose]]) {
  const series = ellip.series.add(name);
  series.categoryFormula = `'模型曲线'!$A$${focusStartRow}:$A$${focusEndRow}`;
  series.formula = `'模型曲线'!$${column}$${focusStartRow}:$${column}$${focusEndRow}`;
  series.fill = color;
}
ellip.setPosition("F15", "J33");
ellip.xAxis = { axisType: "textAxis", tickLabelInterval: 10, textStyle: { fontSize: 9 } };
ellip.yAxis = { numberFormatCode: "0.00", min: 1.0, max: 2.0, textStyle: { fontSize: 9 } };

overview.freezePanes.freezeRows(1);

await fs.mkdir(outputDir, { recursive: true });
for (const [sheetName, range, fileName] of [
  ["概览", "A1:J33", "preview_overview.png"],
  ["原始数据", `A1:F${sourceRows.length + 1}`, "preview_raw.png"],
  ["拟合与方案", "A1:K23", "preview_params.png"],
  ["模型曲线", "A1:J26", "preview_curves.png"],
  ["灵敏度", "A1:D26", "preview_sensitivity.png"],
]) {
  const preview = await workbook.render({ sheetName, range, scale: 1.2, format: "png" });
  await fs.writeFile(path.join(outputDir, fileName), new Uint8Array(await preview.arrayBuffer()));
}

const inspectSummary = await workbook.inspect({
  kind: "table",
  range: "概览!A1:J13",
  include: "values,formulas",
  tableMaxRows: 20,
  tableMaxCols: 12,
});
console.log(inspectSummary.ndjson);
const modelCheck = await workbook.inspect({
  kind: "table",
  range: "模型曲线!A202:J262",
  include: "values,formulas",
  tableMaxRows: 5,
  tableMaxCols: 10,
});
console.log(modelCheck.ndjson);
const errors = await workbook.inspect({
  kind: "match",
  searchTerm: "#REF!|#DIV/0!|#VALUE!|#NAME\\?|#N/A",
  options: { useRegex: true, maxResults: 300 },
  summary: "final formula error scan",
});
console.log(errors.ndjson);

const output = await SpreadsheetFile.exportXlsx(workbook);
await output.save(path.join(outputDir, "D4sigma柱面镜方案.xlsx"));
