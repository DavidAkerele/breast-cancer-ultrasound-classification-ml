import fs from "node:fs/promises";
import path from "node:path";
import { pathToFileURL } from "node:url";

const workspaceDir = process.env.WORKSPACE_DIR;
const SKILL_DIR = process.env.SKILL_DIR;
const RUNTIME_PYTHON = process.env.RUNTIME_PYTHON;
const ARTIFACT_TOOL_DIR = process.env.ARTIFACT_TOOL_DIR;
if (![workspaceDir, SKILL_DIR, RUNTIME_PYTHON, ARTIFACT_TOOL_DIR].every((value) => path.isAbsolute(value ?? ""))) {
  throw new Error("WORKSPACE_DIR, SKILL_DIR, RUNTIME_PYTHON, and ARTIFACT_TOOL_DIR must be absolute paths");
}

const { Presentation, PresentationFile } = await import(
  pathToFileURL(path.join(ARTIFACT_TOOL_DIR, "dist/artifact_tool.mjs")).href
);

const utils = await import(pathToFileURL(path.join(SKILL_DIR, "container_tools/artifact_tool_utils.mjs")).href);
const {
  resolvePresentationFont,
  applyPresentationChartFont,
  makeNativeBulletParagraphs,
  finalizePresentation,
} = utils;

const family = resolvePresentationFont();
const presentation = Presentation.create({ slideSize: { width: 1280, height: 720 } });
const C = {
  navy: "#082F35",
  teal: "#0C746D",
  aqua: "#8DD8CE",
  ivory: "#F6F1E8",
  paper: "#FFFCF6",
  ink: "#153438",
  muted: "#5B706F",
  amber: "#C38A21",
  amberPale: "#F5E8C8",
  line: "#D6DDD8",
  white: "#FFFFFF",
  red: "#9B3F36",
  redPale: "#F3DEDA",
};

const figureDir = path.join(workspaceDir, "latex_university", "figures");
const FIG = {
  discrimination: await fs.readFile(path.join(figureDir, "fig12_discrimination_calibration.png")),
  sourcePerformance: await fs.readFile(path.join(figureDir, "fig13_source_stratified_performance.png")),
  webSingle: await fs.readFile(path.join(workspaceDir, "docs", "images", "dashboard", "01_single_image.png")),
  webNoise: await fs.readFile(path.join(workspaceDir, "docs", "images", "dashboard", "02_noise_experiment.png")),
  webBatch: await fs.readFile(path.join(workspaceDir, "docs", "images", "dashboard", "03_batch_evaluation.png")),
  webEvidence: await fs.readFile(path.join(workspaceDir, "docs", "images", "dashboard", "04_evidence_record.png")),
  webDataset: await fs.readFile(path.join(workspaceDir, "docs", "images", "dashboard", "05_dataset_explorer.png")),
  webDocs: await fs.readFile(path.join(workspaceDir, "docs", "images", "dashboard", "06_documentation.png")),
};

function addFigure(slide, bytes, alt, position, fit = "contain") {
  return slide.images.add({
    blob: bytes,
    contentType: "image/png",
    alt,
    fit,
    position,
    geometry: "roundRect",
    borderRadius: "rounded-xl",
  });
}

function box(slide, { left, top, width, height, fill = C.paper, line = C.line, radius = 18 }) {
  return slide.shapes.add({
    geometry: "roundRect",
    position: { left, top, width, height },
    fill,
    line: { style: "solid", fill: line, width: 1 },
    borderRadius: radius,
  });
}

function textBox(slide, text, { left, top, width, height, size = 24, color = C.ink,
  bold = false, align = "left", fill = "none", line = "none", radius = 0 } = {}) {
  const shape = slide.shapes.add({
    geometry: "textbox",
    position: { left, top, width, height },
    fill,
    line: line === "none" ? { fill: "none", width: 0 } : { style: "solid", fill: line, width: 1 },
    ...(radius ? { borderRadius: radius } : {}),
  });
  shape.text = text;
  shape.text.style = {
    typeface: family,
    fontSize: size,
    bold,
    color,
    alignment: align,
    autoFit: "none",
  };
  return shape;
}

function addHeader(slide, kicker, title, number) {
  textBox(slide, kicker.toUpperCase(), {
    left: 64, top: 34, width: 780, height: 28, size: 15, color: C.teal, bold: true,
  });
  textBox(slide, title, {
    left: 64, top: 68, width: 1110, height: 70, size: 39, color: C.navy, bold: true,
  });
  textBox(slide, String(number).padStart(2, "0"), {
    left: 1170, top: 42, width: 52, height: 28, size: 15, color: C.muted, bold: true, align: "right",
  });
  slide.shapes.add({
    geometry: "line",
    position: { left: 64, top: 146, width: 1152, height: 0 },
    fill: "none",
    line: { style: "solid", fill: C.line, width: 1 },
  });
}

function addFooter(slide, text = "MSc dissertation research prototype") {
  textBox(slide, text, { left: 64, top: 682, width: 1152, height: 20, size: 12, color: C.muted });
}

function addBulletList(slide, items, position, { size = 23, color = C.ink, spacing = 10 } = {}) {
  const shape = slide.shapes.add({
    geometry: "textbox",
    position,
    fill: "none",
    line: { fill: "none", width: 0 },
  });
  shape.text = makeNativeBulletParagraphs(items, {
    marginLeftPoints: 20,
    hangingPoints: 10,
    spaceAfterPoints: spacing,
  });
  shape.text.style = { typeface: family, fontSize: size, color, autoFit: "none" };
  return shape;
}

function styleTable(table, rows, columns, { headerFill = C.navy, bodyFill = C.paper, fontSize = 17 } = {}) {
  table.borders.assign({ style: "solid", fill: C.line, width: 1 });
  for (let row = 0; row < rows; row += 1) {
    for (let column = 0; column < columns; column += 1) {
      const cell = table.getCell(row, column);
      cell.fill = row === 0 ? headerFill : bodyFill;
      cell.text.style = {
        typeface: family,
        fontSize: row === 0 ? fontSize : fontSize - 1,
        bold: row === 0,
        color: row === 0 ? C.white : C.ink,
      };
    }
  }
}

// 1 — Cover
{
  const slide = presentation.slides.add();
  slide.background.fill = C.navy;
  slide.shapes.add({
    geometry: "rect",
    position: { left: 0, top: 0, width: 18, height: 720 },
    fill: C.aqua,
    line: { fill: "none", width: 0 },
  });
  textBox(slide, "MSc DISSERTATION, 2026", {
    left: 80, top: 64, width: 500, height: 32, size: 16, color: C.aqua, bold: true,
  });
  textBox(slide, "Breast Cancer Ultrasound\nClassification Using Machine\nLearning", {
    left: 76, top: 140, width: 920, height: 290, size: 58, color: C.white, bold: true,
  });
  textBox(slide, "Auditable evaluation and research dashboard", {
    left: 82, top: 465, width: 790, height: 70, size: 22, color: "#D7E8E3",
  });
  const pill = slide.shapes.add({
    geometry: "roundRect",
    position: { left: 910, top: 470, width: 282, height: 74 },
    fill: C.amberPale,
    line: { fill: "none", width: 0 },
    borderRadius: "rounded-full",
  });
  pill.text = "LOCAL COHORT EVIDENCE";
  pill.text.style = { typeface: family, fontSize: 18, bold: true, color: "#6E4C08", alignment: "center", autoFit: "none" };
  textBox(slide, "Akerele David Damilola, Student ID 25908322\nSupervisor: Professor Moi Hoon Yap, Manchester Metropolitan University", {
    left: 82, top: 610, width: 720, height: 62, size: 18, color: C.white,
  });
  slide.speakerNotes.textFrame.setText("Open with the evidence boundary. The repaired BrEaST and OASBUD cohort passes the subject-level split audit. The local BUSI derivative remains outside the validated experiment because its patient identifiers and transformation history cannot be reconstructed reliably.");
}

// 2 — Research problem and boundary
{
  const slide = presentation.slides.add();
  slide.background.fill = C.ivory;
  addHeader(slide, "Why this study matters", "Model scores and evidence strength are separate", 2);
  box(slide, { left: 64, top: 182, width: 540, height: 428, fill: C.paper });
  textBox(slide, "The modelling problem", { left: 96, top: 212, width: 450, height: 40, size: 26, bold: true });
  addBulletList(slide, [
    "Ultrasound varies with operator, device, acquisition settings, anatomy, and speckle.",
    "Rectangular images can be geometrically distorted by direct square resizing.",
    "Small public datasets make leakage and undocumented preprocessing unusually consequential.",
  ], { left: 92, top: 274, width: 464, height: 282 }, { size: 21, spacing: 12 });
  box(slide, { left: 636, top: 182, width: 580, height: 204, fill: C.navy, line: C.navy });
  textBox(slide, "Engineering evidence", { left: 670, top: 215, width: 500, height: 40, size: 28, color: C.white, bold: true });
  textBox(slide, "The code path executes consistently. Its artifacts are reproducible and auditable.", {
    left: 670, top: 274, width: 488, height: 78, size: 22, color: "#DDEAE7",
  });
  box(slide, { left: 636, top: 406, width: 580, height: 204, fill: C.redPale, line: "#D7AAA3" });
  textBox(slide, "Generalisation evidence", { left: 670, top: 438, width: 500, height: 40, size: 28, color: C.red, bold: true });
  textBox(slide, "The repaired local cohort has subject separation. External and clinical validation remain outside this study.", {
    left: 670, top: 497, width: 488, height: 78, size: 22, color: "#673E39",
  });
  addFooter(slide);
  slide.speakerNotes.textFrame.setText("Frame the dissertation around two evidence levels. The present result is patient separated within the local BrEaST and OASBUD cohort. It does not establish external or clinical performance.");
}

// 3 — Architecture
{
  const slide = presentation.slides.add();
  slide.background.fill = C.ivory;
  addHeader(slide, "System design", "One controlled path connects the data to every output", 3);
  const labels = [
    ["01", "DATA", "OASBUD and BrEaST"],
    ["02", "AUDIT", "subjects and provenance"],
    ["03", "PREPROCESS", "CLAHE, crop, pad"],
    ["04", "MODEL", "EfficientNet, ResNet, CNN"],
    ["05", "EVALUATE", "predictions and metrics"],
    ["06", "SURFACES", "CLI, API, web"],
  ];
  const nodes = [];
  for (let index = 0; index < labels.length; index += 1) {
    const [number, label, detail] = labels[index];
    const left = 58 + index * 202;
    const node = box(slide, { left, top: 248, width: 174, height: 180, fill: index === 1 ? C.amberPale : C.paper, line: index === 1 ? "#D6B56F" : C.line, radius: 16 });
    textBox(slide, number, { left: left + 20, top: 270, width: 44, height: 24, size: 15, color: C.teal, bold: true });
    textBox(slide, label, { left: left + 20, top: 311, width: 134, height: 32, size: index === 2 ? 16 : 21, color: C.navy, bold: true });
    textBox(slide, detail, { left: left + 20, top: 357, width: 134, height: 54, size: 16, color: C.muted });
    nodes.push(node);
  }
  for (let index = 0; index < nodes.length - 1; index += 1) {
    slide.shapes.connect(nodes[index], nodes[index + 1], {
      kind: "straight",
      fromSide: "right",
      toSide: "left",
      line: { style: "solid", fill: C.teal, width: 2 },
      tail: { type: "triangle", width: "sm", length: "sm" },
    });
  }
  box(slide, { left: 164, top: 490, width: 952, height: 104, fill: "#E6F2EF", line: "#B8D7D1" });
  textBox(slide, "Shared contract", { left: 194, top: 516, width: 180, height: 32, size: 21, color: C.teal, bold: true });
  textBox(slide, "Training, evaluation, command-line prediction, and API inference call the same preprocessing implementation.", {
    left: 382, top: 509, width: 694, height: 55, size: 20, color: C.ink,
  });
  addFooter(slide);
  slide.speakerNotes.textFrame.setText("Point out that the audit is a first-class stage. Every downstream surface shares the same preprocessing contract. BUSI is documented but cannot enter this path because the local derivative lacks recoverable patient provenance.");
}

// 4 — Audit table
{
  const slide = presentation.slides.add();
  slide.background.fill = C.ivory;
  addHeader(slide, "Current cohort", "Validated datasets pass the subject-level split audit", 4);
  const values = [
    ["Dataset", "Train", "Validation", "Test", "Audit finding"],
    ["BUSI derivative", "n/a", "n/a", "n/a", "Excluded: provenance cannot be reconstructed"],
    ["OASBUD", "138", "30", "30", "Pass: no subject crosses a partition"],
    ["BrEaST", "164", "33", "32", "Pass: no subject crosses a partition"],
    ["Validated", "302", "63", "62", "Pass: local cohort evidence"],
  ];
  const table = slide.tables.add({
    rows: values.length,
    columns: values[0].length,
    left: 64,
    top: 190,
    width: 1152,
    height: 324,
    columnWidths: [170, 120, 140, 110, 612],
    values,
  });
  styleTable(table, values.length, values[0].length, { fontSize: 18 });
  for (let column = 0; column < values[0].length; column += 1) {
    table.getCell(4, column).fill = "#D7EEE9";
    table.getCell(4, column).text.style = { typeface: family, fontSize: 17, bold: true, color: C.teal };
  }
  box(slide, { left: 64, top: 542, width: 1152, height: 90, fill: C.amberPale, line: "#D6B56F" });
  textBox(slide, "Release gate", { left: 92, top: 568, width: 155, height: 28, size: 21, color: C.teal, bold: true });
  textBox(slide, "The BUSI derivative cannot prove patient separation or reproduce published curation checks, so it remains outside the experiment.", {
    left: 260, top: 560, width: 916, height: 45, size: 20, color: "#5D4920",
  });
  addFooter(slide, "Source: outputs/data_audit.json, generated from the supplied local folders");
  slide.speakerNotes.textFrame.setText("Source: outputs/data_audit.json. BrEaST and OASBUD pass the subject-level audit. Exclusion concerns this local BUSI derivative, not the value of the original BUSI dataset published by Al-Dhabyani et al. The local copy was renamed and contains processed square derivatives, so its patient mapping, duplicate checks, overlays, and curation history cannot be reproduced.");
}

// 5 — Preprocessing
{
  const slide = presentation.slides.add();
  slide.background.fill = C.ivory;
  addHeader(slide, "Experimental method", "Preprocessing is configurable, shared, and testable", 5);
  const cards = [
    ["DIRECT RESIZE", "Rectangular image is resized directly to 224×224.", "Comparator. Scaling differs across the two axes."],
    ["CENTRE CROP", "A centred square is selected before resizing.", "Comparator. Peripheral context may be removed."],
    ["MASK / FALLBACK + PAD", "Mask box or central 80%, then margin and reflected square padding.", "Default research path. The crop retains its aspect ratio."],
  ];
  cards.forEach(([label, method, caveat], index) => {
    const left = 64 + index * 392;
    box(slide, { left, top: 198, width: 360, height: 288, fill: index === 2 ? "#E6F2EF" : C.paper, line: index === 2 ? "#98C8BF" : C.line });
    textBox(slide, `0${index + 1}`, { left: left + 28, top: 224, width: 45, height: 25, size: 15, color: C.teal, bold: true });
    textBox(slide, label, { left: left + 28, top: 267, width: 304, height: 48, size: 22, color: C.navy, bold: true });
    textBox(slide, method, { left: left + 28, top: 330, width: 304, height: 65, size: 20, color: C.ink });
    textBox(slide, caveat, { left: left + 28, top: 414, width: 304, height: 48, size: 16, color: C.muted });
  });
  textBox(slide, "Controlled ablation rule", { left: 64, top: 532, width: 280, height: 34, size: 22, color: C.teal, bold: true });
  textBox(slide, "Hold the subject partition, architecture, training budget, and random seeds fixed. Change one preprocessing factor at a time.", {
    left: 350, top: 525, width: 866, height: 62, size: 21, color: C.ink,
  });
  textBox(slide, "Training augmentation uses horizontal flips and rotations up to 15°. Vertical flips are excluded because image depth has acquisition meaning.", {
    left: 64, top: 610, width: 1152, height: 42, size: 17, color: C.muted,
  });
  addFooter(slide);
  slide.speakerNotes.textFrame.setText("The current implementation supports all three strategies. The submitted experiment uses the default path. A comparative preprocessing claim would require a matched ablation with the same subject partition, seed policy, and training budget. That ablation was not run for this revision.");
}

// 6 — Provisional result
{
  const slide = presentation.slides.add();
  slide.background.fill = C.ivory;
  addHeader(slide, "Executable checkpoint", "Audited test performance", 6);
  const chart = slide.charts.add("bar", {
    position: { left: 54, top: 196, width: 690, height: 382 },
    categories: ["Accuracy", "Macro F1", "Macro recall", "ROC-AUC"],
    series: [{ name: "Score", values: [0.6774, 0.6774, 0.6803, 0.7419], fill: C.teal }],
    barOptions: { direction: "bar", grouping: "clustered", gapWidth: 48 },
    hasLegend: false,
    xAxis: { min: 0, max: 1, numberFormatCode: "0%", majorGridlines: { style: "solid", fill: C.line, width: 1 } },
    yAxis: { line: { style: "solid", fill: C.line, width: 1 } },
    dataLabels: { showValue: true, position: "outEnd", textStyle: { fill: C.navy, fontSize: 16, bold: true } },
    chartFill: C.ivory,
    plotAreaFill: C.ivory,
  });
  applyPresentationChartFont(chart, { fontFamily: family });
  textBox(slide, "CONFUSION MATRIX", { left: 798, top: 202, width: 350, height: 30, size: 17, color: C.teal, bold: true });
  const matrixValues = [
    ["Actual / predicted", "Benign", "Malignant"],
    ["Benign", "21", "12"],
    ["Malignant", "8", "21"],
  ];
  const matrix = slide.tables.add({
    rows: 3,
    columns: 3,
    left: 794,
    top: 244,
    width: 422,
    height: 208,
    columnWidths: [182, 120, 120],
    values: matrixValues,
  });
  styleTable(matrix, 3, 3, { headerFill: C.navy, bodyFill: C.paper, fontSize: 17 });
  matrix.getCell(1, 1).fill = "#D7EEE9";
  matrix.getCell(2, 2).fill = "#D7EEE9";
  matrix.getCell(1, 2).fill = C.redPale;
  matrix.getCell(2, 1).fill = C.redPale;
  box(slide, { left: 794, top: 480, width: 422, height: 104, fill: C.amberPale, line: "#D6B56F" });
  textBox(slide, "62 test images, EfficientNet-B0", { left: 820, top: 502, width: 370, height: 28, size: 19, color: "#6E4C08", bold: true });
  textBox(slide, "Local cohort evidence only. Clinical validation is outside scope.", { left: 820, top: 540, width: 370, height: 34, size: 16, color: "#5D4920" });
  addFooter(slide, "Source: outputs/metrics.json and outputs/predictions.csv");
  slide.speakerNotes.textFrame.setText("Source: outputs/metrics.json and outputs/predictions.csv generated on the audited 62-image test folders. Accuracy 0.6774; macro F1 0.6774; macro recall 0.6803; ROC-AUC 0.7419. The confusion matrix contains 21 true negatives, 12 false positives, 8 false negatives, and 21 true positives. Audit status: passed for BrEaST and OASBUD.");
}

// 7 — Reproducibility controls
{
  const slide = presentation.slides.add();
  slide.background.fill = C.ivory;
  addHeader(slide, "What changed", "The project now leaves a verifiable evidence trail", 7);
  addBulletList(slide, [
    "Subject-level audit is regenerated before training. Failed audits stop by default.",
    "Python, NumPy, PyTorch, and CUDA seeds are set and checkpoint configuration is saved.",
    "Evaluation writes one row per image plus metrics, figures, audit status, and checksums.",
    "Mask discovery, crop dimensions, audit failure, and the non-clinical API boundary have regression tests.",
    "The dissertation, notebook, slides, README, and web interface use the same current evidence status.",
  ], { left: 74, top: 190, width: 748, height: 406 }, { size: 22, spacing: 11 });
  box(slide, { left: 864, top: 190, width: 352, height: 408, fill: C.navy, line: C.navy });
  textBox(slide, "RELEASE ARTIFACTS", { left: 898, top: 225, width: 286, height: 28, size: 16, color: C.aqua, bold: true });
  textBox(slide, "data_audit.json\npredictions.csv\nmetrics.json\nextended_evaluation.json\nreport figures", {
    left: 898, top: 281, width: 286, height: 208, size: 20, color: C.white, bold: true,
  });
  textBox(slide, "Generated, inspectable, and tied to one evaluation run.", {
    left: 898, top: 516, width: 276, height: 54, size: 17, color: "#D7E8E3",
  });
  addFooter(slide);
  slide.speakerNotes.textFrame.setText("Emphasise reproducibility as the dissertation's present contribution. The extended evaluation was calculated from the preserved EfficientNet probabilities. No model was retrained and no threshold was selected from the test set.");
}

// 8 — Internal model comparison
{
  const slide = presentation.slides.add();
  slide.background.fill = C.ivory;
  addHeader(slide, "Internal benchmark", "Architecture comparison on the audited test split", 8);
  const chart = slide.charts.add("bar", {
    position: { left: 52, top: 186, width: 666, height: 388 },
    categories: ["EfficientNet-B0", "Custom CNN", "ResNet-50"],
    series: [
      { name: "Accuracy", values: [0.6774, 0.6452, 0.5645], fill: C.teal },
      { name: "Macro F1", values: [0.6774, 0.6452, 0.5374], fill: C.aqua },
      { name: "ROC-AUC", values: [0.7419, 0.6708, 0.6364], fill: C.amber },
    ],
    barOptions: { direction: "column", grouping: "clustered", gapWidth: 48 },
    hasLegend: true,
    legend: { position: "bottom", textStyle: { fontSize: 14, fill: C.ink } },
    yAxis: { min: 0, max: 1, numberFormatCode: "0%", majorGridlines: { style: "solid", fill: C.line, width: 1 } },
    xAxis: { line: { style: "solid", fill: C.line, width: 1 } },
    dataLabels: { showValue: true, position: "outEnd", textStyle: { fill: C.navy, fontSize: 12, bold: true } },
    chartFill: C.ivory,
    plotAreaFill: C.ivory,
  });
  applyPresentationChartFont(chart, { fontFamily: family });
  const values = [
    ["Model", "Accuracy 95% CI", "ROC-AUC 95% CI"],
    ["EfficientNet-B0", "0.5645 to 0.7903", "0.6217 to 0.8616"],
    ["Custom CNN", "0.5323 to 0.7581", "0.5330 to 0.7979"],
    ["ResNet-50", "0.4355 to 0.6935", "0.5054 to 0.7701"],
  ];
  const table = slide.tables.add({
    rows: values.length,
    columns: values[0].length,
    left: 752,
    top: 210,
    width: 464,
    height: 256,
    columnWidths: [156, 154, 154],
    values,
  });
  styleTable(table, values.length, values[0].length, { fontSize: 15 });
  box(slide, { left: 752, top: 494, width: 464, height: 106, fill: C.amberPale, line: "#D6B56F" });
  textBox(slide, "Intervals overlap", { left: 780, top: 516, width: 408, height: 28, size: 21, color: "#6E4C08", bold: true });
  textBox(slide, "The ranking is descriptive. The data do not support a superiority or breakthrough claim.", {
    left: 780, top: 550, width: 408, height: 40, size: 16, color: "#5D4920",
  });
  addFooter(slide, "Source: outputs/model_benchmark.json, one training seed and 62 test images");
  slide.speakerNotes.textFrame.setText("All models use the same audited test split. EfficientNet-B0 has the highest accuracy, macro F1, and ROC-AUC point estimates. The bootstrap intervals overlap, and the stored benchmark has summaries rather than paired per-image outputs, so a paired significance test cannot be reconstructed. The custom CNN exceeding ResNet-50 in this run is useful evidence, but it is not a scientific breakthrough.");
}

// 9 — Discrimination and calibration
{
  const slide = presentation.slides.add();
  slide.background.fill = C.ivory;
  addHeader(slide, "Extended evaluation", "Discrimination and calibration metrics", 9);
  addFigure(slide, FIG.discrimination, "Precision-recall curve and reliability diagram for the EfficientNet-B0 test predictions", {
    left: 64, top: 178, width: 820, height: 424,
  });
  textBox(slide, "AVERAGE PRECISION", { left: 930, top: 196, width: 260, height: 24, size: 14, color: C.teal, bold: true });
  textBox(slide, "0.6803", { left: 924, top: 228, width: 270, height: 58, size: 42, color: C.navy, bold: true });
  textBox(slide, "BRIER SCORE", { left: 930, top: 324, width: 260, height: 24, size: 14, color: C.teal, bold: true });
  textBox(slide, "0.2144", { left: 924, top: 356, width: 270, height: 58, size: 42, color: C.navy, bold: true });
  textBox(slide, "EXPECTED CALIBRATION ERROR", { left: 930, top: 452, width: 280, height: 24, size: 14, color: C.teal, bold: true });
  textBox(slide, "0.0652", { left: 924, top: 484, width: 270, height: 58, size: 42, color: C.navy, bold: true });
  textBox(slide, "Five bins, descriptive test-set diagnostics", { left: 930, top: 558, width: 270, height: 36, size: 15, color: C.muted });
  addFooter(slide, "Source: outputs/extended_evaluation.json and outputs/predictions.csv");
  slide.speakerNotes.textFrame.setText("Average precision summarises malignant-class ranking under class imbalance. The Brier score measures probability error. Expected calibration error compares confidence with observed frequency in five bins. These are descriptive diagnostics on 62 images. No calibration model or decision threshold was fitted on the test set.");
}

// 10 — Source-specific error analysis
{
  const slide = presentation.slides.add();
  slide.background.fill = C.ivory;
  addHeader(slide, "Error analysis", "Errors differ between BrEaST and OASBUD", 10);
  addFigure(slide, FIG.sourcePerformance, "Source-stratified test performance for BrEaST and OASBUD", {
    left: 64, top: 178, width: 742, height: 404,
  });
  const values = [
    ["Source", "n", "Accuracy", "FP", "FN"],
    ["BrEaST", "32", "0.7813", "2", "5"],
    ["OASBUD", "30", "0.5667", "10", "3"],
  ];
  const table = slide.tables.add({
    rows: values.length,
    columns: values[0].length,
    left: 838,
    top: 210,
    width: 378,
    height: 196,
    columnWidths: [112, 48, 94, 62, 62],
    values,
  });
  styleTable(table, values.length, values[0].length, { fontSize: 15 });
  box(slide, { left: 838, top: 438, width: 378, height: 142, fill: C.redPale, line: "#D7AAA3" });
  textBox(slide, "Interpretation boundary", { left: 866, top: 464, width: 322, height: 28, size: 20, color: C.red, bold: true });
  textBox(slide, "Source, case mix, acquisition, reconstruction, and sample size are confounded. The plot cannot rank dataset quality.", {
    left: 866, top: 502, width: 322, height: 62, size: 16, color: "#673E39",
  });
  addFooter(slide, "Fixed threshold 0.5 with no source-specific threshold tuning");
  slide.speakerNotes.textFrame.setText("OASBUD contributes ten of the twelve false positives and three of the eight false negatives. BrEaST contributes two false positives and five false negatives. This shows why pooled accuracy is incomplete. It does not establish that one dataset is better because source and acquisition factors are confounded.");
}

// 11 — Web interface
{
  const slide = presentation.slides.add();
  slide.background.fill = C.ivory;
  addHeader(slide, "Research dashboard", "Six pages document the complete workflow", 11);
  const screens = [
    [FIG.webSingle, "Single-image inference"],
    [FIG.webNoise, "Noise robustness"],
    [FIG.webBatch, "Batch explorer"],
    [FIG.webEvidence, "Evidence page"],
    [FIG.webDataset, "Dataset documentation"],
    [FIG.webDocs, "Project documentation"],
  ];
  screens.forEach(([bytes, alt], index) => {
    const column = index % 3;
    const row = Math.floor(index / 3);
    addFigure(slide, bytes, alt, {
      left: 64 + column * 392,
      top: 168 + row * 220,
      width: 360,
      height: 202,
    }, "cover");
  });
  addFooter(slide, "The dashboard documents research outputs. It does not provide a clinical service");
  slide.speakerNotes.textFrame.setText("Walk through the six pages: single-image inference, noise robustness, batch explorer, evidence, dataset documentation, and project documentation. Explain that the same API and preprocessing code support every page. The interface includes appropriate-use and unsupported-use guidance. The screenshots document implementation only and do not add clinical evidence.");
}

// 12 — Conclusion
{
  const slide = presentation.slides.add();
  slide.background.fill = C.navy;
  textBox(slide, "CONCLUSION", { left: 68, top: 48, width: 400, height: 30, size: 16, color: C.aqua, bold: true });
  textBox(slide, "A complete research system\nwith an honest evidence boundary", {
    left: 64, top: 112, width: 820, height: 150, size: 48, color: C.white, bold: true,
  });
  box(slide, { left: 64, top: 320, width: 532, height: 236, fill: "#123E43", line: "#2D5C60" });
  textBox(slide, "Defensible now", { left: 96, top: 350, width: 444, height: 36, size: 26, color: C.aqua, bold: true });
  addBulletList(slide, [
    "Shared preprocessing and inference implementation",
    "Auditable evaluation artifacts and release checks",
    "Patient-separated local checkpoint with uncertainty estimates",
  ], { left: 92, top: 398, width: 450, height: 150 }, { size: 18, color: C.white, spacing: 6 });
  box(slide, { left: 628, top: 320, width: 588, height: 236, fill: C.amberPale, line: "#D6B56F" });
  textBox(slide, "Claim boundary", { left: 660, top: 350, width: 510, height: 60, size: 24, color: "#6E4C08", bold: true });
  addBulletList(slide, [
    "Clinical effectiveness or diagnostic safety",
    "External validation beyond the local combined cohort",
    "Statistically proven superiority over other architectures",
  ], { left: 654, top: 422, width: 520, height: 130 }, { size: 19, color: "#4F4022", spacing: 8 });
  textBox(slide, "Product repository: github.com/DavidAkerele/breast-cancer-ultrasound-classification-ml", {
    left: 64, top: 590, width: 900, height: 28, size: 17, color: C.aqua, bold: true,
  });
  textBox(slide, "Questions", { left: 64, top: 635, width: 400, height: 42, size: 29, color: C.white, bold: true });
  textBox(slide, "No diagnostic, BI-RADS, or management inference is authorised by this prototype.", {
    left: 570, top: 639, width: 646, height: 32, size: 15, color: "#C7DAD6", align: "right",
  });
  slide.speakerNotes.textFrame.setText("Close with the central contribution: a reproducible, patient-separated local experiment linked to a transparent research interface. Point the examiners to the repository before the product demonstration. The report does not use the BUSI derivative, does not claim clinical readiness, and does not claim model superiority. Future work can add multi-seed training, expert error review, and external validation without changing the submitted evidence.");
}

const buildDir = path.join(workspaceDir, ".qa", "pptx-build");
const finalDir = path.join(workspaceDir, ".qa", "pptx-final");
const outputPath = path.join(workspaceDir, "docs", "presentation", "Akerele_David_25908322_Presentation.pptx");
await fs.mkdir(buildDir, { recursive: true });
await fs.mkdir(finalDir, { recursive: true });
await fs.mkdir(path.dirname(outputPath), { recursive: true });

const revision = Date.now();
const candidatePath = path.join(buildDir, `candidate-${revision}.pptx`);
const finalPath = path.join(finalDir, `validated-${revision}.pptx`);
await (await PresentationFile.exportPptx(presentation)).save(candidatePath);

const requirements = {
  explicitTotalSlideCount: 12,
  requiredNativeTableOwnerSlides: [4, 6, 8, 10],
  requiredNativeChartOwnerSlides: [6, 8],
  requiredEmbeddedWorkbookChartOwnerSlides: [],
  materializeLiteralChartWorkbooks: true,
};
const fontPolicy = { basis: "design", families: [family] };
const result = await finalizePresentation({
  ...requirements,
  workspaceDir,
  candidatePath,
  finalPath,
  pythonExecutable: RUNTIME_PYTHON,
  integrityValidatorPath: path.join(SKILL_DIR, "container_tools", "inspect_presentation_package_integrity.py"),
  layoutValidatorPath: path.join(SKILL_DIR, "container_tools", "inspect_presentation_layout_geometry.py"),
  layoutArgs: [
    "--expected-slide-size-emu", "12192000,6858000",
    "--validate-bullet-geometry",
    "--validate-heading-fit",
    "--require-native-table-slide", "4",
    "--require-native-table-slide", "6",
    "--require-native-table-slide", "8",
    "--require-native-table-slide", "10",
  ],
  requiredNativeTableOwnerSlides: requirements.requiredNativeTableOwnerSlides,
  fontPolicy,
  verifyArtifactToolImport: true,
  receiptPath: path.join(buildDir, `validation-${revision}.json`),
});
await fs.copyFile(finalPath, outputPath);
console.log(JSON.stringify({ outputPath, finalPath, family, result }, null, 2));
