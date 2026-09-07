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

function addFooter(slide, text = "MSc dissertation · research prototype") {
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
  textBox(slide, "MSc DISSERTATION · 2026", {
    left: 80, top: 64, width: 500, height: 32, size: 16, color: C.aqua, bold: true,
  });
  textBox(slide, "Breast ultrasound\nclassification using\ndeep learning", {
    left: 76, top: 140, width: 920, height: 290, size: 58, color: C.white, bold: true,
  });
  textBox(slide, "A reproducibility-focused study of preprocessing, convolutional models, and evidence integrity", {
    left: 82, top: 465, width: 790, height: 70, size: 22, color: "#D7E8E3",
  });
  const pill = slide.shapes.add({
    geometry: "roundRect",
    position: { left: 910, top: 470, width: 282, height: 74 },
    fill: C.amberPale,
    line: { fill: "none", width: 0 },
    borderRadius: "rounded-full",
  });
  pill.text = "PROVISIONAL EVIDENCE";
  pill.text.style = { typeface: family, fontSize: 18, bold: true, color: "#6E4C08", alignment: "center", autoFit: "none" };
  textBox(slide, "Akerele David Damilola\nManchester Metropolitan University", {
    left: 82, top: 610, width: 720, height: 62, size: 18, color: C.white,
  });
  slide.speakerNotes.textFrame.setText("Open with the evidence boundary: this is a reproducible research prototype. The supplied cohort currently fails its subject-level split audit.");
}

// 2 — Research problem and boundary
{
  const slide = presentation.slides.add();
  slide.background.fill = C.ivory;
  addHeader(slide, "Why this study matters", "A strong score is not the same as strong evidence", 2);
  box(slide, { left: 64, top: 182, width: 540, height: 428, fill: C.paper });
  textBox(slide, "The modelling problem", { left: 96, top: 212, width: 450, height: 40, size: 26, bold: true });
  addBulletList(slide, [
    "Ultrasound varies with operator, device, acquisition settings, anatomy, and speckle.",
    "Rectangular images can be geometrically distorted by direct square resizing.",
    "Small public datasets make leakage and undocumented preprocessing unusually consequential.",
  ], { left: 92, top: 274, width: 464, height: 282 }, { size: 21, spacing: 12 });
  box(slide, { left: 636, top: 182, width: 580, height: 204, fill: C.navy, line: C.navy });
  textBox(slide, "Engineering evidence", { left: 670, top: 215, width: 500, height: 40, size: 28, color: C.white, bold: true });
  textBox(slide, "The code path executes consistently; artifacts are reproducible and auditable.", {
    left: 670, top: 274, width: 488, height: 78, size: 22, color: "#DDEAE7",
  });
  box(slide, { left: 636, top: 406, width: 580, height: 204, fill: C.redPale, line: "#D7AAA3" });
  textBox(slide, "Generalisation evidence", { left: 670, top: 438, width: 500, height: 40, size: 28, color: C.red, bold: true });
  textBox(slide, "Requires verified subject separation. The current local cohort does not meet this condition.", {
    left: 670, top: 497, width: 488, height: 78, size: 22, color: "#673E39",
  });
  addFooter(slide);
  slide.speakerNotes.textFrame.setText("Frame the dissertation around two evidence levels. The present result validates integration, not patient-independent or clinical performance.");
}

// 3 — Architecture
{
  const slide = presentation.slides.add();
  slide.background.fill = C.ivory;
  addHeader(slide, "System design", "One preprocessing path across every executable surface", 3);
  const labels = [
    ["01", "DATA", "BUSI · OASBUD · BrEaST"],
    ["02", "AUDIT", "subjects · provenance"],
    ["03", "PREPROCESS", "CLAHE · crop · pad"],
    ["04", "MODEL", "EfficientNet · ResNet · CNN"],
    ["05", "EVALUATE", "predictions · metrics"],
    ["06", "SURFACES", "CLI · API · web"],
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
  slide.speakerNotes.textFrame.setText("Point out that the audit is a first-class stage, not an appendix. Every downstream surface shares the same preprocessing contract.");
}

// 4 — Audit table
{
  const slide = presentation.slides.add();
  slide.background.fill = C.ivory;
  addHeader(slide, "Current cohort", "The data audit fails before model comparison begins", 4);
  const values = [
    ["Dataset", "Train", "Validation", "Test", "Audit finding"],
    ["BUSI", "100", "50", "50", "Subject IDs unverifiable after renaming"],
    ["OASBUD", "140", "30", "30", "30nh appears in train and validation"],
    ["BrEaST", "162", "27", "40", "case140 and case151 cross partitions"],
    ["Combined", "402", "107", "120", "FAILED — provisional metrics only"],
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
    table.getCell(4, column).fill = C.redPale;
    table.getCell(4, column).text.style = { typeface: family, fontSize: 17, bold: true, color: C.red };
  }
  box(slide, { left: 64, top: 542, width: 1152, height: 90, fill: C.amberPale, line: "#D6B56F" });
  textBox(slide, "Release gate", { left: 92, top: 568, width: 155, height: 28, size: 21, color: "#6E4C08", bold: true });
  textBox(slide, "Training now stops by default unless --allow-unaudited-data is supplied for an explicitly provisional run.", {
    left: 260, top: 560, width: 916, height: 45, size: 20, color: "#5D4920",
  });
  addFooter(slide, "Source: outputs/data_audit.json · generated from the supplied local folders");
  slide.speakerNotes.textFrame.setText("Source: outputs/data_audit.json. Counts are image-folder counts, not an audited patient cohort. Explain the three concrete audit failures.");
}

// 5 — Preprocessing
{
  const slide = presentation.slides.add();
  slide.background.fill = C.ivory;
  addHeader(slide, "Experimental method", "Preprocessing is configurable, shared, and testable", 5);
  const cards = [
    ["DIRECT RESIZE", "Rectangular image is resized directly to 224×224.", "Comparator; anisotropic when x/y scale factors differ."],
    ["CENTRE CROP", "A centred square is selected before resizing.", "Comparator; may remove peripheral context."],
    ["MASK / FALLBACK + PAD", "Mask bbox or central 80% → margin → reflected square pad.", "Default research path; preserves crop aspect ratio."],
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
  textBox(slide, "Hold the subject partition, architecture, training budget, and random seeds fixed; change one preprocessing factor at a time.", {
    left: 350, top: 525, width: 866, height: 62, size: 21, color: C.ink,
  });
  textBox(slide, "Training augmentation: horizontal flips + rotations ≤15°. Vertical flips are excluded because image depth has acquisition meaning.", {
    left: 64, top: 610, width: 1152, height: 42, size: 17, color: C.muted,
  });
  addFooter(slide);
  slide.speakerNotes.textFrame.setText("The current implementation supports all three strategies. No comparative preprocessing claim is made until the subject-level cohort is rebuilt.");
}

// 6 — Provisional result
{
  const slide = presentation.slides.add();
  slide.background.fill = C.ivory;
  addHeader(slide, "Executable checkpoint", "End-to-end execution, provisional evidence", 6);
  const chart = slide.charts.add("bar", {
    position: { left: 54, top: 196, width: 690, height: 382 },
    categories: ["Accuracy", "Macro F1", "Macro recall", "ROC-AUC"],
    series: [{ name: "Score", values: [0.7833, 0.7811, 0.7809, 0.8612], fill: C.teal }],
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
    ["Benign", "53", "9"],
    ["Malignant", "17", "41"],
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
  textBox(slide, "120 test images · EfficientNet-B0", { left: 820, top: 502, width: 370, height: 28, size: 19, color: "#6E4C08", bold: true });
  textBox(slide, "Software-regression evidence only; the data audit failed.", { left: 820, top: 540, width: 370, height: 28, size: 16, color: "#5D4920" });
  addFooter(slide, "Source: outputs/metrics.json and outputs/predictions.csv");
  slide.speakerNotes.textFrame.setText("Source: outputs/metrics.json generated on the 120-image test folders. Accuracy 0.7833; macro F1 0.7811; macro recall 0.7809; ROC-AUC 0.8612. Audit status: failed.");
}

// 7 — Reproducibility controls
{
  const slide = presentation.slides.add();
  slide.background.fill = C.ivory;
  addHeader(slide, "What changed", "The project now leaves a verifiable evidence trail", 7);
  addBulletList(slide, [
    "Subject-level audit is regenerated before training; failed audits stop by default.",
    "Python, NumPy, PyTorch, and CUDA seeds are set and checkpoint configuration is saved.",
    "Evaluation writes one row per image plus metrics, figures, audit status, and checksums.",
    "Mask discovery, crop dimensions, audit failure, and the non-clinical API boundary have regression tests.",
    "The dissertation, notebook, Word document, slides, and web interface use the same current evidence status.",
  ], { left: 74, top: 190, width: 748, height: 406 }, { size: 22, spacing: 11 });
  box(slide, { left: 864, top: 190, width: 352, height: 408, fill: C.navy, line: C.navy });
  textBox(slide, "RELEASE ARTIFACTS", { left: 898, top: 225, width: 286, height: 28, size: 16, color: C.aqua, bold: true });
  textBox(slide, "data_audit.json\npredictions.csv\nmetrics.json\nconfusion_matrix.png\nroc_curve.png", {
    left: 898, top: 281, width: 286, height: 208, size: 24, color: C.white, bold: true,
  });
  textBox(slide, "Generated, inspectable, and tied to one evaluation run.", {
    left: 898, top: 516, width: 276, height: 54, size: 17, color: "#D7E8E3",
  });
  addFooter(slide);
  slide.speakerNotes.textFrame.setText("Emphasise reproducibility as the dissertation's present contribution. These controls prevent illustrative UI values from being mistaken for measured findings.");
}

// 8 — Conclusion
{
  const slide = presentation.slides.add();
  slide.background.fill = C.navy;
  textBox(slide, "CONCLUSION", { left: 68, top: 48, width: 400, height: 30, size: 16, color: C.aqua, bold: true });
  textBox(slide, "The framework is ready.\nThe evidence is not finished.", {
    left: 64, top: 112, width: 820, height: 150, size: 48, color: C.white, bold: true,
  });
  box(slide, { left: 64, top: 320, width: 532, height: 236, fill: "#123E43", line: "#2D5C60" });
  textBox(slide, "Defensible now", { left: 96, top: 350, width: 444, height: 36, size: 26, color: C.aqua, bold: true });
  addBulletList(slide, [
    "Shared preprocessing and inference implementation",
    "Auditable evaluation artifacts and release checks",
    "Provisional checkpoint as engineering evidence",
  ], { left: 92, top: 398, width: 450, height: 150 }, { size: 18, color: C.white, spacing: 6 });
  box(slide, { left: 628, top: 320, width: 588, height: 236, fill: C.amberPale, line: "#D6B56F" });
  textBox(slide, "Required before a final performance claim", { left: 660, top: 350, width: 510, height: 60, size: 24, color: "#6E4C08", bold: true });
  addBulletList(slide, [
    "Rebuild subject-level partitions from original identifiers",
    "Freeze the test manifest before model selection",
    "Rerun matched multi-seed experiments and regenerate every result",
  ], { left: 654, top: 422, width: 520, height: 130 }, { size: 19, color: "#4F4022", spacing: 8 });
  textBox(slide, "Questions", { left: 64, top: 625, width: 400, height: 48, size: 31, color: C.white, bold: true });
  textBox(slide, "No diagnostic, BI-RADS, or management inference is authorised by this prototype.", {
    left: 570, top: 629, width: 646, height: 36, size: 16, color: "#C7DAD6", align: "right",
  });
  slide.speakerNotes.textFrame.setText("Close on the distinction between a release-ready research framework and an unfinished patient-independent experiment. Invite questions.");
}

const buildDir = path.join(workspaceDir, ".qa", "pptx-build");
const finalDir = path.join(workspaceDir, ".qa", "pptx-final");
const outputPath = path.join(workspaceDir, "docs", "presentation", "Breast_Cancer_Ultrasound_Dissertation_Presentation.pptx");
await fs.mkdir(buildDir, { recursive: true });
await fs.mkdir(finalDir, { recursive: true });
await fs.mkdir(path.dirname(outputPath), { recursive: true });

const revision = Date.now();
const candidatePath = path.join(buildDir, `candidate-${revision}.pptx`);
const finalPath = path.join(finalDir, `validated-${revision}.pptx`);
await (await PresentationFile.exportPptx(presentation)).save(candidatePath);

const requirements = {
  explicitTotalSlideCount: 8,
  requiredNativeTableOwnerSlides: [4, 6],
  requiredNativeChartOwnerSlides: [6],
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
  ],
  requiredNativeTableOwnerSlides: requirements.requiredNativeTableOwnerSlides,
  fontPolicy,
  verifyArtifactToolImport: true,
  receiptPath: path.join(buildDir, `validation-${revision}.json`),
});
await fs.copyFile(finalPath, outputPath);
console.log(JSON.stringify({ outputPath, finalPath, family, result }, null, 2));
