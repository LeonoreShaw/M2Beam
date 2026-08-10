import fs from "node:fs/promises";
import path from "node:path";
import { Presentation, PresentationFile } from "@oai/artifact-tool";

const ROOT = "F:/CodexProjects/光斑图片";
const DATA = `${ROOT}/outputs/cylindrical_lens_d4sigma_20260810`;
const OUT = `${DATA}/D4sigma椭圆率与柱面镜实施方案.pptx`;
const QA = `${ROOT}/.pptx_work/rendered`;

const C = {
  ink: "#111111",
  gray: "#5F6670",
  panel: "#EDEDED",
  rule: "#B8BCC4",
  blue: "#3D8DFF",
  lightBlue: "#D0EDFA",
  cyan: "#6DCBF4",
  teal: "#009E73",
  orange: "#E69F00",
  magenta: "#CC79A7",
  red: "#D55E00",
  white: "#FFFFFF",
};
const FONT = "Microsoft YaHei";

function parseCsv(text) {
  const lines = text.replace(/^\uFEFF/, "").trim().split(/\r?\n/);
  const split = (line) => {
    const out = [];
    let item = "";
    let quoted = false;
    for (let i = 0; i < line.length; i++) {
      const ch = line[i];
      if (ch === '"') {
        if (quoted && line[i + 1] === '"') { item += '"'; i++; }
        else quoted = !quoted;
      } else if (ch === "," && !quoted) { out.push(item); item = ""; }
      else item += ch;
    }
    out.push(item);
    return out;
  };
  const headers = split(lines[0]);
  return lines.slice(1).map((line) => {
    const values = split(line);
    return Object.fromEntries(headers.map((h, i) => [h, values[i] ?? ""]));
  });
}

const results = JSON.parse(await fs.readFile(`${DATA}/design_results.json`, "utf8"));
const curves = parseCsv(await fs.readFile(`${DATA}/model_curves.csv`, "utf8"));
const fits = parseCsv(await fs.readFile(`${DATA}/fit_summary.csv`, "utf8"));
const designs = parseCsv(await fs.readFile(`${DATA}/design_summary.csv`, "utf8"));

const xFit = fits.find((r) => r.axis === "X");
const yFit = fits.find((r) => r.axis === "Y");
const robust = designs.find((r) => r.design.includes("推荐标准件_整段均衡"));

const n = (v) => Number(v);
const band = curves.filter((r) => {
  const z = n(r.z_mm);
  return z >= 220 && z <= 250 && Math.abs(z - Math.round(z / 2) * 2) < 1e-8;
});
const bandCategories = band.map((r) => `${n(r.z_mm).toFixed(0)}`);

const deck = Presentation.create({ slideSize: { width: 1280, height: 720 } });

function addText(slide, text, pos, size = 24, opts = {}) {
  const shape = slide.shapes.add({
    geometry: "textbox",
    name: opts.name,
    position: pos,
    fill: opts.fill ?? "none",
    line: opts.line ?? { style: "solid", fill: "none", width: 0 },
  });
  shape.text = text;
  shape.text.style = {
    typeface: FONT,
    fontSize: size,
    bold: opts.bold ?? false,
    color: opts.color ?? C.ink,
    alignment: opts.alignment ?? "left",
    verticalAlignment: opts.verticalAlignment ?? "top",
    autoFit: opts.autoFit ?? "shrinkText",
    insets: opts.insets ?? { top: 0, right: 0, bottom: 0, left: 0 },
  };
  return shape;
}

function addTitle(slide, title, number) {
  addText(slide, title, { left: 42, top: 34, width: 1185, height: 72 }, 48, { bold: true, name: `slide-${number}-title` });
  addText(slide, String(number), { left: 1178, top: 660, width: 52, height: 20 }, 16, { alignment: "right", color: C.gray, name: `slide-${number}-number` });
}

function addPanel(slide, pos, fill = C.panel, name) {
  return slide.shapes.add({
    geometry: "rect",
    name,
    position: pos,
    fill,
    line: { style: "solid", fill: "none", width: 0 },
  });
}

function addNotes(slide, lines, external = []) {
  const sources = [
    `${DATA}/source_d4sigma_consolidated.csv`,
    `${DATA}/fit_summary.csv`,
    `${DATA}/design_summary.csv`,
    `${DATA}/model_curves.csv`,
    ...external,
  ];
  slide.speakerNotes.textFrame.setText([
    ...lines,
    "",
    "[Sources]",
    ...sources.map((s) => `- ${s}`),
  ]);
}

// Slide 1 — sparse cover, based on Codex Grid layout 01.
{
  const slide = deck.slides.add();
  slide.background.fill = C.white;
  addText(slide, "D4σ / M² CAUSTIC ANALYSIS", { left: 42, top: 42, width: 650, height: 42 }, 24, { color: C.gray, bold: true, name: "cover-eyebrow" });
  addText(slide, "D4σ 椭圆率与\n柱面镜修正方案", { left: 42, top: 175, width: 900, height: 250 }, 76, { bold: true, verticalAlignment: "bottom", autoFit: "none", name: "cover-title" });
  slide.shapes.add({ geometry: "line", name: "cover-rule", position: { left: 42, top: 468, width: 420, height: 0 }, fill: "none", line: { style: "solid", fill: C.blue, width: 5 } });
  addText(slide, "+50 mm ZnSe 柱面镜 · z≈219.43 mm · 焦区最大轴比≈1.256", { left: 42, top: 500, width: 980, height: 52 }, 30, { color: C.ink, name: "cover-subtitle" });
  addText(slide, "基于原始 CSV 的二阶矩焦散拟合与薄透镜传播", { left: 42, top: 580, width: 760, height: 36 }, 22, { color: C.gray, name: "cover-method" });
  addNotes(slide, ["介绍汇报目标：总结椭圆率变化、M²图变化及光路实施方式。"]);
}

// Slide 2 — metric-led summary, based on layout 19.
{
  const slide = deck.slides.add();
  slide.background.fill = C.white;
  addTitle(slide, "柱面镜把焦区最大轴比从 1.731 降至 1.256", 2);
  addText(slide, "原始两轴束腰几乎在同一位置，主要矛盾是束腰尺寸和发散斜率不匹配；对较宽的 X 方向增加正光焦度即可显著改善焦区圆度。", { left: 42, top: 122, width: 1170, height: 92 }, 26, { color: C.gray, name: "summary-explanation" });

  const cards = [
    { left: 42, stat: "1.731 → 1.256", label: "220–250 mm 最大椭圆率", color: C.blue },
    { left: 453, stat: "57.8% → 79.6%", label: "焦区最低圆度", color: C.teal },
    { left: 864, stat: "≈27.4%", label: "最大轴比降幅", color: C.orange },
  ];
  for (const [i, card] of cards.entries()) {
    addPanel(slide, { left: card.left, top: 315, width: 375, height: 300 }, C.panel, `metric-panel-${i + 1}`);
    slide.shapes.add({ geometry: "rect", name: `metric-accent-${i + 1}`, position: { left: card.left, top: 315, width: 12, height: 300 }, fill: card.color, line: { style: "solid", fill: "none", width: 0 } });
    addText(slide, card.stat, { left: card.left + 30, top: 375, width: 320, height: 105 }, 48, { bold: true, color: card.color, verticalAlignment: "bottom", name: `metric-stat-${i + 1}` });
    addText(slide, card.label, { left: card.left + 30, top: 515, width: 320, height: 70 }, 24, { name: `metric-label-${i + 1}` });
  }
  addNotes(slide, ["区分拟合焦区最大值与离散测点：1.731 是连续焦散模型在焦点附近的最大轴比。"]);
}

// Slide 3 — chart-led evidence, based on layout 21.
{
  const slide = deck.slides.add();
  slide.background.fill = C.white;
  addTitle(slide, "焦区椭圆率从尖峰变为接近恒定的 1.25", 3);
  slide.charts.add("line", {
    position: { left: 42, top: 145, width: 800, height: 480 },
    categories: bandCategories,
    series: [
      { name: "原始拟合", values: band.map((r) => n(r.baseline_ellipticity)), line: { style: "solid", width: 4, fill: C.gray }, marker: { symbol: "none" } },
      { name: "+50 mm 柱面镜", values: band.map((r) => n(r.robust_ellipticity)), line: { style: "solid", width: 4, fill: C.teal }, marker: { symbol: "none" } },
    ],
    hasLegend: true,
    legend: { position: "bottom", overlay: false, textStyle: { fontSize: 18, fill: C.ink } },
    chartFill: C.white,
    chartLine: { style: "solid", fill: C.white, width: 0 },
    plotAreaFill: { type: "none" },
    plotAreaLine: { style: "solid", fill: C.white, width: 0 },
    xAxis: { visible: true, title: { text: "z (mm)", textStyle: { fontSize: 18, fill: C.ink } }, textStyle: { fontSize: 14, fill: C.gray }, line: { style: "solid", fill: C.rule, width: 1 } },
    yAxis: { visible: true, min: 1.0, max: 1.8, majorUnit: 0.2, title: { text: "椭圆率 = 长轴/短轴", textStyle: { fontSize: 18, fill: C.ink } }, numberFormatCode: "0.00", textStyle: { fontSize: 14, fill: C.gray }, majorGridlines: { style: "solid", fill: C.panel, width: 1 } },
    lineOptions: { grouping: "standard", smooth: true },
  });
  addText(slide, "为什么无法整段达到 1.00？", { left: 880, top: 165, width: 340, height: 48 }, 28, { bold: true, name: "limit-heading" });
  addText(slide, "单片无损柱面镜不改变两轴 M²。\n\n因此整段恒定轴比存在一个理论下限。", { left: 880, top: 235, width: 330, height: 132 }, 23, { color: C.gray, name: "limit-body" });
  addPanel(slide, { left: 880, top: 390, width: 330, height: 125 }, C.panel, "limit-panel");
  addText(slide, "√(M²X/M²Y)\n= 1.2549", { left: 905, top: 412, width: 280, height: 82 }, 36, { bold: true, color: C.blue, alignment: "center", name: "limit-formula" });
  addText(slide, "推荐方案最大轴比 1.2563，距理论下限仅 0.0014。", { left: 880, top: 548, width: 330, height: 66 }, 22, { color: C.ink, name: "limit-conclusion" });
  addNotes(slide, ["强调绿色曲线的平坦性：目标不是只在单点变圆，而是在220–250 mm焦区内控制最差轴比。"]);
}

// Slide 4 — M2 bar chart plus caustic evidence.
{
  const slide = deck.slides.add();
  slide.background.fill = C.white;
  addTitle(slide, "柱面镜改变焦散形状，但不会改善 M²", 4);
  addText(slide, "M² 图中前后柱高相同；变化发生在 X 轴束腰与发散的重新分配。", { left: 42, top: 115, width: 1180, height: 45 }, 24, { color: C.gray, name: "m2-subtitle" });

  slide.charts.add("bar", {
    position: { left: 42, top: 185, width: 390, height: 405 },
    categories: ["X", "Y"],
    series: [
      { name: "加镜前", values: [n(xFit.m2), n(yFit.m2)], fill: C.gray },
      { name: "加镜后", values: [n(xFit.m2), n(yFit.m2)], fill: C.blue },
    ],
    barOptions: { direction: "column", grouping: "clustered", gapWidth: 80 },
    hasLegend: true,
    legend: { position: "bottom", overlay: false, textStyle: { fontSize: 16, fill: C.ink } },
    dataLabels: { showValue: true, position: "outEnd", textStyle: { fontSize: 16, fill: C.ink, bold: true } },
    xAxis: { visible: true, textStyle: { fontSize: 18, fill: C.ink }, line: { style: "solid", fill: C.rule, width: 1 } },
    yAxis: { visible: true, min: 0, max: 3.0, majorUnit: 0.5, title: { text: "M²", textStyle: { fontSize: 18, fill: C.ink } }, numberFormatCode: "0.0", majorGridlines: { style: "solid", fill: C.panel, width: 1 }, textStyle: { fontSize: 14, fill: C.gray } },
    chartFill: C.white,
    chartLine: { style: "solid", fill: C.white, width: 0 },
    plotAreaFill: { type: "none" },
  });

  slide.charts.add("line", {
    position: { left: 470, top: 185, width: 755, height: 405 },
    categories: bandCategories,
    series: [
      { name: "原始 X", values: band.map((r) => n(r.baseline_d4x_um)), line: { style: "solid", width: 3, fill: C.red }, marker: { symbol: "none" } },
      { name: "Y（不变）", values: band.map((r) => n(r.baseline_d4y_um)), line: { style: "solid", width: 3, fill: C.blue }, marker: { symbol: "none" } },
      { name: "加镜后 X", values: band.map((r) => n(r.robust_d4x_um)), line: { style: "solid", width: 4, fill: C.teal }, marker: { symbol: "none" } },
    ],
    hasLegend: true,
    legend: { position: "bottom", overlay: false, textStyle: { fontSize: 15, fill: C.ink } },
    xAxis: { visible: true, title: { text: "z (mm)", textStyle: { fontSize: 17, fill: C.ink } }, textStyle: { fontSize: 13, fill: C.gray }, line: { style: "solid", fill: C.rule, width: 1 } },
    yAxis: { visible: true, min: 0, max: 1800, majorUnit: 300, title: { text: "D4σ (µm)", textStyle: { fontSize: 17, fill: C.ink } }, numberFormatCode: "0", majorGridlines: { style: "solid", fill: C.panel, width: 1 }, textStyle: { fontSize: 13, fill: C.gray } },
    chartFill: C.white,
    chartLine: { style: "solid", fill: C.white, width: 0 },
    plotAreaFill: { type: "none" },
    lineOptions: { grouping: "standard", smooth: true },
  });

  addText(slide, `X 束腰：${n(xFit.d0_um).toFixed(0)} → ${n(robust.new_x_waist_d4_um).toFixed(0)} µm`, { left: 470, top: 610, width: 345, height: 34 }, 21, { bold: true, color: C.teal, name: "waist-change" });
  addText(slide, `X 发散：${n(xFit.theta_um_per_mm).toFixed(2)} → ${n(robust.new_x_divergence_um_per_mm).toFixed(2)} µm/mm`, { left: 825, top: 610, width: 400, height: 34 }, 21, { bold: true, color: C.teal, name: "divergence-change" });
  addText(slide, "M²X=2.6065 / M²Y=1.6552\n前后相同", { left: 55, top: 595, width: 350, height: 52 }, 20, { bold: true, alignment: "center", color: C.gray, name: "m2-exact-values" });
  addNotes(slide, ["说明M²保持不变并不意味着光斑形状不变；薄柱面镜通过改变曲率重新分配束腰和发散。"]);
}

// Slide 5 — native PowerPoint optical implementation diagram.
{
  const slide = deck.slides.add();
  slide.background.fill = C.white;
  addTitle(slide, "在原焦点前 15.89 mm 加入 +50 mm 柱面镜", 5);

  // Connectors and beam envelope first so they remain behind nodes.
  const line = (name, left, top, width, height, color, dash = "solid", w = 2) => {
    const position = { left, top, width, height };
    if (position.width < 0) {
      position.left += position.width;
      position.width = Math.abs(position.width);
      position.horizontalFlip = true;
    }
    if (position.height < 0) {
      position.top += position.height;
      position.height = Math.abs(position.height);
      position.verticalFlip = true;
    }
    return slide.shapes.add({ geometry: "line", name, position, fill: "none", line: { style: dash, fill: color, width: w } });
  };
  line("optical-axis", 90, 325, 1080, 0, C.rule, "dashed", 2);
  line("beam-upper-1", 170, 245, 350, 50, C.blue, "solid", 4);
  line("beam-lower-1", 170, 405, 350, -50, C.blue, "solid", 4);
  line("beam-upper-2", 520, 295, 330, 30, C.teal, "solid", 4);
  line("beam-lower-2", 520, 355, 330, -30, C.teal, "solid", 4);
  line("beam-upper-3", 850, 325, 250, 65, C.teal, "solid", 4);
  line("beam-lower-3", 850, 325, 250, -65, C.teal, "solid", 4);
  line("distance-bracket", 520, 475, 330, 0, C.ink, "solid", 2);
  line("x-power-arrow", 965, 555, 140, 0, C.blue, "solid", 3);
  line("y-axis-line", 1035, 495, 0, 120, C.gray, "solid", 3);

  const existing = slide.shapes.add({ geometry: "ellipse", name: "existing-optics", position: { left: 154, top: 235, width: 32, height: 180 }, fill: C.lightBlue, line: { style: "solid", fill: C.blue, width: 2 } });
  const cylinder = slide.shapes.add({ geometry: "roundRect", name: "cylindrical-lens", position: { left: 508, top: 250, width: 24, height: 150 }, fill: C.cyan, line: { style: "solid", fill: C.blue, width: 2 }, borderRadius: 8 });
  const focus = slide.shapes.add({ geometry: "ellipse", name: "focus-point", position: { left: 839, top: 314, width: 22, height: 22 }, fill: C.orange, line: { style: "solid", fill: C.orange, width: 1 } });
  const camera = slide.shapes.add({ geometry: "rect", name: "camera-scan", position: { left: 1090, top: 240, width: 70, height: 170 }, fill: C.panel, line: { style: "solid", fill: C.ink, width: 2 } });

  addText(slide, "现有上游\n聚焦光学", { left: 95, top: 170, width: 155, height: 60 }, 22, { bold: true, alignment: "center", name: "existing-label" });
  addText(slide, "+50 mm ZnSe\n10.6 µm AR", { left: 420, top: 175, width: 210, height: 62 }, 24, { bold: true, alignment: "center", color: C.blue, name: "lens-label" });
  addText(slide, "镜片主平面\nz = 219.43 mm", { left: 420, top: 410, width: 210, height: 62 }, 21, { alignment: "center", name: "lens-z-label" });
  addText(slide, "新焦区中心\nz ≈ 235.32 mm", { left: 755, top: 175, width: 205, height: 62 }, 24, { bold: true, alignment: "center", color: C.orange, name: "focus-label" });
  addText(slide, "15.89 mm", { left: 650, top: 482, width: 95, height: 30 }, 20, { bold: true, alignment: "center", name: "distance-label" });
  addText(slide, "相机扫描\nz = 220–250 mm", { left: 1030, top: 165, width: 190, height: 62 }, 22, { bold: true, alignment: "center", name: "camera-label" });

  addPanel(slide, { left: 905, top: 485, width: 310, height: 165 }, "#F7F7F7", "orientation-panel");
  addText(slide, "镜片方向", { left: 925, top: 502, width: 120, height: 34 }, 24, { bold: true, name: "orientation-title" });
  line("orientation-x-visible", 940, 578, 85, 0, C.blue, "solid", 3);
  line("orientation-y-visible", 982, 542, 0, 78, C.gray, "solid", 3);
  slide.shapes.add({ geometry: "roundRect", name: "orientation-cylinder-icon", position: { left: 975, top: 550, width: 15, height: 55 }, fill: C.cyan, line: { style: "solid", fill: C.blue, width: 1 }, borderRadius: 5 });
  addText(slide, "X：有光焦度", { left: 1035, top: 558, width: 160, height: 30 }, 18, { color: C.blue, bold: true, name: "x-orientation-label" });
  addText(slide, "Y：柱面母线 / 无光焦度", { left: 930, top: 615, width: 260, height: 26 }, 17, { color: C.gray, alignment: "center", name: "y-orientation-label" });

  addText(slide, "坐标要求：z 必须沿用原 CSV 文件夹的机械零点；若装调平台零点不同，只保持镜片到焦区中心约 15.9 mm 的相对关系。", { left: 42, top: 610, width: 820, height: 58 }, 20, { color: C.gray, name: "coordinate-note" });
  addNotes(slide, ["光路实现：上游聚焦光学不变，在原焦点前加入正柱面镜，只给X方向增加会聚。"], ["https://www.lasercomponents.com/en/product/znse-lenses-for-co2-lasers/"]);
}

// Slide 6 — three-step implementation and acceptance plan, based on layout 17.
{
  const slide = deck.slides.add();
  slide.background.fill = C.white;
  addTitle(slide, "按三点复测和 ±1 mm 微调完成装调验收", 6);

  // Timeline connector first.
  slide.shapes.add({ geometry: "line", name: "timeline-line", position: { left: 105, top: 305, width: 1010, height: 0 }, fill: "none", line: { style: "solid", fill: C.rule, width: 3 } });
  const steps = [
    { x: 115, num: "01", title: "安装", body: "+50 mm ZnSe\n镜片 z=219.43 mm\n有效口径 ≥10 mm" },
    { x: 455, num: "02", title: "定向", body: "柱面母线沿 Y\n有光焦度方向沿 X\n旋转至长轴最小" },
    { x: 795, num: "03", title: "复测与微调", body: "复测 z=220 / 235 / 250\n镜片位置微调 ±1 mm\n最小化三区间最大轴比" },
  ];
  for (const [i, step] of steps.entries()) {
    slide.shapes.add({ geometry: "ellipse", name: `step-dot-${i + 1}`, position: { left: step.x, top: 292, width: 26, height: 26 }, fill: C.blue, line: { style: "solid", fill: C.blue, width: 1 } });
    addText(slide, step.num, { left: step.x, top: 240, width: 80, height: 32 }, 20, { bold: true, color: C.blue, name: `step-number-${i + 1}` });
    addText(slide, step.title, { left: step.x, top: 345, width: 300, height: 48 }, 32, { bold: true, name: `step-title-${i + 1}` });
    addText(slide, step.body, { left: step.x, top: 410, width: 295, height: 130 }, 22, { color: C.gray, name: `step-body-${i + 1}` });
  }
  addPanel(slide, { left: 42, top: 575, width: 1170, height: 72 }, C.panel, "acceptance-panel");
  addText(slide, "验收目标：220–250 mm 内最大椭圆率 ≤1.30；若明显超差，优先检查主轴旋转、z 零点、镜片焦距公差及截光。", { left: 68, top: 594, width: 1120, height: 38 }, 24, { bold: true, color: C.ink, name: "acceptance-text" });
  addNotes(slide, ["建议用与原始CSV完全相同的D4σ算法复测，以避免算法口径变化造成伪改善。"]);
}

await fs.mkdir(QA, { recursive: true });
for (const [i, slide] of deck.slides.items.entries()) {
  const stem = `slide-${String(i + 1).padStart(2, "0")}`;
  const png = await deck.export({ slide, format: "png", scale: 1.4 });
  await fs.writeFile(`${QA}/${stem}.png`, new Uint8Array(await png.arrayBuffer()));
  const layout = await slide.export({ format: "layout" });
  await fs.writeFile(`${QA}/${stem}.layout.json`, await layout.text());
}
const montage = await deck.export({ format: "webp", montage: true, scale: 1 });
await fs.writeFile(`${QA}/montage.webp`, new Uint8Array(await montage.arrayBuffer()));

const inspection = await deck.inspect({ kind: "slide,textbox,shape,chart,notes", maxChars: 12000 });
await fs.writeFile(`${QA}/deck-inspect.ndjson`, inspection.ndjson, "utf8");

const pptx = await PresentationFile.exportPptx(deck);
await pptx.save(OUT);
console.log(OUT);
