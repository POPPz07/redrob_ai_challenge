import fs from "node:fs/promises";
import path from "node:path";
import { SpreadsheetFile, Workbook } from "@oai/artifact-tool";

const inputPath = process.argv[2] ?? "outputs/annotation_audit/annotation_profiles.json";
const outputDir = process.argv[3] ?? "outputs/annotation_audit";
const profiles = JSON.parse(await fs.readFile(inputPath, "utf8"));
await fs.mkdir(outputDir, { recursive: true });

const colors = {
  ink: "#17202A",
  teal: "#0B6B69",
  tealLight: "#DDEFEA",
  blueLight: "#E8F1F8",
  yellow: "#FFF4CC",
  redLight: "#FCE8E6",
  greenLight: "#E3F2E7",
  white: "#FFFFFF",
  line: "#CBD5E1",
  muted: "#52606D",
};

const columns = [
  ["profile_key", "Profile Key"],
  ["current_title", "Current Title"],
  ["current_company", "Current Company"],
  ["years_experience", "Years Experience"],
  ["location", "Location"],
  ["industry", "Industry"],
  ["headline", "Headline"],
  ["summary", "Summary"],
  ["career_history", "Career History"],
  ["skills_and_evidence", "Skills and Evidence"],
  ["education", "Education"],
  ["availability", "Availability"],
  ["market_and_trust_signals", "Market and Trust Signals"],
];

const rubricRows = [
  [5, "Exceptional fit", "Direct, substantial production ownership of search, retrieval, ranking, recommendation, matching, or embeddings; strong evaluation evidence; recent hands-on Python; shipped to real users. Minor gaps are acceptable."],
  [4, "Strong fit", "Strong production ML or relevant backend/platform evidence with credible retrieval/ranking work. May lack one ideal dimension such as formal evaluation metrics, startup context, or exact location."],
  [3, "Plausible fit", "Solid applied ML, data, or backend engineer with some relevant search/recommendation/embedding evidence, but production ownership or depth is not fully proven."],
  [2, "Weak fit", "Adjacent technical profile, demos/tutorials, mostly data engineering, CV/speech/robotics, or generic LLM work with limited production information-retrieval relevance."],
  [1, "Poor fit", "Mostly irrelevant, nontechnical, services-only without relevant product evidence, stale hands-on coding, or little evidence beyond keywords."],
  [0, "Reject / misleading", "Severe cross-field inconsistency, implausible claims, obvious keyword stuffing, disqualifying availability/location evidence, or no credible support for the claimed expertise."],
];

function styleTitle(sheet, rangeAddress, title) {
  const range = sheet.getRange(rangeAddress);
  range.merge();
  range.values = [[title]];
  range.format = {
    fill: colors.teal,
    font: { bold: true, color: colors.white, size: 18 },
    verticalAlignment: "center",
    horizontalAlignment: "left",
  };
  range.format.rowHeight = 34;
}

function buildWorkbook(reviewerNumber) {
  const workbook = Workbook.create();
  const instructions = workbook.worksheets.add("Instructions");
  const rubric = workbook.worksheets.add("Rubric");
  const annotations = workbook.worksheets.add("Annotations");

  instructions.showGridLines = false;
  styleTitle(instructions, "A1:F2", `Bug Solvers Blind Relevance Review - Reviewer ${reviewerNumber}`);
  instructions.getRange("A4:B11").values = [
    ["Purpose", "Independently judge candidate relevance for Redrob's Senior AI Engineer - Founding Team role."],
    ["Profiles", profiles.length],
    ["Required fields", "Relevance (0-5), confidence (1-3), risk flag, and a brief evidence-based rationale."],
    ["Independence", "Do not discuss labels or inspect another reviewer's workbook until both reviews are complete."],
    ["Evidence rule", "Use only facts shown in the profile. Do not infer prestige, identity, gender, age, or protected characteristics."],
    ["Priority", "Production retrieval/ranking/recommendation, evaluation practice, Python engineering, and systems shipped to real users."],
    ["Behavior", "Availability, response, activity, location/relocation, and notice period modify technical relevance; they do not replace it."],
    ["Trap handling", "Flag unsupported expert claims, impossible duration, career/title mismatch, keyword stuffing, or inconsistent evidence."],
  ];
  instructions.getRange("A4:A11").format = {
    fill: colors.tealLight,
    font: { bold: true, color: colors.ink },
    verticalAlignment: "top",
  };
  instructions.getRange("B4:B11").format = {
    wrapText: true,
    verticalAlignment: "top",
    font: { color: colors.ink },
  };
  instructions.getRange("A4:B11").format.borders = { preset: "insideHorizontal", style: "thin", color: colors.line };
  instructions.getRange("A4:A11").format.columnWidth = 21;
  instructions.getRange("B4:B11").format.columnWidth = 92;
  instructions.getRange("A4:B11").format.rowHeight = 36;
  instructions.freezePanes.freezeRows(2);

  rubric.showGridLines = false;
  styleTitle(rubric, "A1:C2", "Relevance and Risk Rubric");
  rubric.getRange("A4:C10").values = [
    ["Score", "Label", "Decision standard"],
    ...rubricRows,
  ];
  rubric.getRange("A4:C4").format = {
    fill: colors.ink,
    font: { bold: true, color: colors.white },
    horizontalAlignment: "center",
  };
  rubric.getRange("A5:A10").format = { horizontalAlignment: "center", font: { bold: true } };
  rubric.getRange("A5:C10").format.wrapText = true;
  rubric.getRange("A5:C10").format.verticalAlignment = "top";
  rubric.getRange("A4:C10").format.borders = { preset: "insideHorizontal", style: "thin", color: colors.line };
  rubric.getRange("A4:A10").format.columnWidth = 10;
  rubric.getRange("B4:B10").format.columnWidth = 23;
  rubric.getRange("C4:C10").format.columnWidth = 100;
  rubric.getRange("A5:C10").format.rowHeight = 46;
  rubric.getRange("A12:C16").values = [
    ["Risk flag", "Meaning", "Examples"],
    ["No", "Evidence is reasonably coherent.", "Minor gaps or ordinary uncertainty are not a risk flag."],
    ["Unsure", "Something may be inconsistent but the visible evidence is insufficient.", "Overstated proficiency, unclear chronology, unsupported scale."],
    ["Yes", "Material evidence suggests misleading or internally inconsistent claims.", "Expert skill with zero use, nontechnical career plus dense AI keywords, impossible chronology."],
    ["Reminder", "Score job relevance separately from risk.", "A technically relevant profile can still carry risk; explain the interaction in the rationale."],
  ];
  rubric.getRange("A12:C12").format = { fill: colors.ink, font: { bold: true, color: colors.white } };
  rubric.getRange("A13:C16").format.wrapText = true;
  rubric.getRange("A12:C16").format.borders = { preset: "insideHorizontal", style: "thin", color: colors.line };
  rubric.freezePanes.freezeRows(2);

  annotations.showGridLines = false;
  const headers = [...columns.map(([, label]) => label), "Relevance 0-5", "Confidence 1-3", "Risk Flag", "Evidence-based Rationale"];
  const rows = profiles.map((profile) => [
    ...columns.map(([key]) => profile[key] ?? ""),
    null,
    null,
    null,
    null,
  ]);
  annotations.getRangeByIndexes(0, 0, 1, headers.length).values = [headers];
  annotations.getRangeByIndexes(1, 0, rows.length, headers.length).values = rows;
  const headerRange = annotations.getRangeByIndexes(0, 0, 1, headers.length);
  headerRange.format = {
    fill: colors.ink,
    font: { bold: true, color: colors.white },
    wrapText: true,
    verticalAlignment: "center",
    horizontalAlignment: "center",
  };
  headerRange.format.rowHeight = 38;

  const body = annotations.getRangeByIndexes(1, 0, rows.length, headers.length);
  body.format = {
    verticalAlignment: "top",
    wrapText: true,
    font: { color: colors.ink, size: 10 },
  };
  body.format.borders = { preset: "insideHorizontal", style: "thin", color: colors.line };
  body.format.rowHeight = 94;
  annotations.getRange(`A2:A${rows.length + 1}`).format = { fill: colors.blueLight, font: { bold: true, color: colors.teal } };
  annotations.getRange(`N2:Q${rows.length + 1}`).format = { fill: colors.yellow, verticalAlignment: "top", wrapText: true };
  annotations.getRange(`N2:N${rows.length + 1}`).dataValidation = { rule: { type: "whole", operator: "between", formula1: 0, formula2: 5 } };
  annotations.getRange(`O2:O${rows.length + 1}`).dataValidation = { rule: { type: "whole", operator: "between", formula1: 1, formula2: 3 } };
  annotations.getRange(`P2:P${rows.length + 1}`).dataValidation = { rule: { type: "list", values: ["No", "Unsure", "Yes"] } };
  annotations.getRange(`N2:N${rows.length + 1}`).conditionalFormats.add("colorScale", {
    colors: ["#F8D7DA", "#FFF4CC", "#D9EAD3"],
    thresholds: ["min", "50%", "max"],
  });
  annotations.getRange(`P2:P${rows.length + 1}`).conditionalFormats.add("containsText", { text: "Yes", format: { fill: colors.redLight, font: { bold: true, color: "#9B1C1C" } } });
  annotations.getRange(`P2:P${rows.length + 1}`).conditionalFormats.add("containsText", { text: "No", format: { fill: colors.greenLight, font: { color: "#166534" } } });

  const widths = [15, 24, 22, 12, 21, 18, 35, 58, 90, 72, 38, 48, 58, 13, 14, 13, 58];
  widths.forEach((width, index) => {
    annotations.getRangeByIndexes(0, index, rows.length + 1, 1).format.columnWidth = width;
  });
  annotations.freezePanes.freezeRows(1);
  annotations.freezePanes.freezeColumns(1);
  const table = annotations.tables.add(`A1:Q${rows.length + 1}`, true, `Reviewer${reviewerNumber}Annotations`);
  table.style = "TableStyleMedium2";
  table.showBandedRows = true;
  table.showFilterButton = true;

  return workbook;
}

for (const reviewerNumber of [1, 2]) {
  const workbook = buildWorkbook(reviewerNumber);
  const inspection = await workbook.inspect({
    kind: "sheet,table",
    maxChars: 5000,
    tableMaxRows: 3,
    tableMaxCols: 6,
  });
  await fs.writeFile(path.join(outputDir, `reviewer_${reviewerNumber}_inspect.txt`), inspection.ndjson ?? String(inspection), "utf8");

  for (const sheetName of ["Instructions", "Rubric", "Annotations"]) {
    const preview = await workbook.render({ sheetName, autoCrop: "all", scale: sheetName === "Annotations" ? 0.45 : 1, format: "png" });
    await fs.writeFile(
      path.join(outputDir, `reviewer_${reviewerNumber}_${sheetName.toLowerCase()}_preview.png`),
      new Uint8Array(await preview.arrayBuffer()),
    );
  }

  const output = await SpreadsheetFile.exportXlsx(workbook);
  await output.save(path.join(outputDir, `annotation_reviewer_${reviewerNumber}.xlsx`));
}

console.log(JSON.stringify({ reviewers: 2, profiles: profiles.length, outputDir: path.resolve(outputDir) }, null, 2));