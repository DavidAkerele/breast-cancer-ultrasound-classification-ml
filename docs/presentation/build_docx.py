"""Build the evidence-aware dissertation Word document from repository sources."""
import re
from pathlib import Path

from docx import Document
from docx.enum.table import WD_CELL_VERTICAL_ALIGNMENT, WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor


ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "docs/presentation/Breast_Cancer_Ultrasound_Dissertation.docx"
CHAPTERS = [
    "Chapter_1_Introduction.md",
    "Chapter_2_Literature_Review.md",
    "Chapter_3_Methodology.md",
    "Chapter_4_Implementation_And_Training.md",
    "Chapter_5_Results_And_Evaluation.md",
    "Chapter_6_Discussion_And_Conclusion.md",
]

ABSTRACT = (
    "Breast ultrasound is affected by speckle, acquisition settings, lesion context, and operator technique. "
    "This dissertation develops a reproducible research prototype for binary classification of breast-ultrasound "
    "images using EfficientNet-B0, ResNet-50, and a custom convolutional baseline. The implementation combines "
    "configurable CLAHE, aspect-ratio-preserving square padding, machine-readable evaluation artifacts, and a "
    "FastAPI research interface. Its principal contribution is methodological: training, evaluation, command-line "
    "prediction, and API inference share one preprocessing path, while a read-only audit checks subject-level split "
    "integrity before metrics are interpreted. The supplied OASBUD and BrEaST folders have been rebuilt into subject-level "
    "splits with no cross-partition identifiers. BUSI remains excluded because its renamed files do not retain enough "
    "provenance to verify patient separation. The fresh EfficientNet-B0 checkpoint achieved 67.74% accuracy, macro F1 "
    "0.6774, and ROC-AUC 0.7419 on 62 held-out BrEaST and OASBUD images. These are local-cohort research results only, "
    "not clinical or external validation."
)

DECLARATION = (
    "No part of this project has been submitted in support of an application for any other degree or qualification "
    "at this or any other institute of learning. Apart from those parts containing citations to the work of others, "
    "this project is my own unaided work. This work has been carried out in accordance with Manchester Metropolitan "
    "University research ethics procedures."
)

ABBREVIATIONS = [
    ("ACR", "American College of Radiology"),
    ("AUC-ROC", "Area Under the Receiver Operating Characteristic Curve"),
    ("BI-RADS", "Breast Imaging-Reporting and Data System"),
    ("BUSI", "Breast Ultrasound Images Dataset"),
    ("CLAHE", "Contrast Limited Adaptive Histogram Equalisation"),
    ("CNN", "Convolutional Neural Network"),
    ("Grad-CAM", "Gradient-weighted Class Activation Mapping"),
    ("OASBUD", "Open Access Ultrasound Database"),
    ("ROC-AUC", "Area Under the Receiver Operating Characteristic Curve"),
    ("ROI", "Region of Interest"),
]


def set_cell_fill(cell, colour):
    shading = OxmlElement("w:shd")
    shading.set(qn("w:fill"), colour)
    cell._tc.get_or_add_tcPr().append(shading)


def set_repeat_table_header(row):
    properties = row._tr.get_or_add_trPr()
    repeat = OxmlElement("w:tblHeader")
    repeat.set(qn("w:val"), "true")
    properties.append(repeat)


def add_field(paragraph, instruction):
    run = paragraph.add_run()
    begin = OxmlElement("w:fldChar")
    begin.set(qn("w:fldCharType"), "begin")
    text = OxmlElement("w:instrText")
    text.set(qn("xml:space"), "preserve")
    text.text = instruction
    separate = OxmlElement("w:fldChar")
    separate.set(qn("w:fldCharType"), "separate")
    end = OxmlElement("w:fldChar")
    end.set(qn("w:fldCharType"), "end")
    run._r.extend([begin, text, separate, end])


def configure_document(doc):
    styles = doc.styles
    normal = styles["Normal"]
    normal.font.name = "Times New Roman"
    normal.font.size = Pt(12)
    normal._element.rPr.rFonts.set(qn("w:eastAsia"), "Times New Roman")
    normal.paragraph_format.line_spacing = 1.5
    normal.paragraph_format.space_after = Pt(6)
    normal.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY

    for style_name, size, colour in [
        ("Title", 28, "062D2C"),
        ("Heading 1", 20, "062D2C"),
        ("Heading 2", 15, "0D6F66"),
        ("Heading 3", 12, "244D48"),
    ]:
        style = styles[style_name]
        style.font.name = "Aptos Display"
        style.font.size = Pt(size)
        style.font.bold = True
        style.font.color.rgb = RGBColor.from_string(colour)
        style.paragraph_format.keep_with_next = True
        style.paragraph_format.space_before = Pt(14)
        style.paragraph_format.space_after = Pt(7)

    # Explicit page breaks are more reliable than style-driven breaks when the
    # document is opened by both Word and LibreOffice.
    styles["Heading 1"].paragraph_format.page_break_before = False
    for section in doc.sections:
        section.top_margin = Inches(0.9)
        section.bottom_margin = Inches(0.85)
        section.left_margin = Inches(1.05)
        section.right_margin = Inches(0.9)
        section.header_distance = Inches(0.35)
        section.footer_distance = Inches(0.35)
        section.different_first_page_header_footer = True

        header = section.header.paragraphs[0]
        header.text = "Breast Ultrasound Classification · MSc Dissertation"
        header.style = styles["Caption"]
        header.alignment = WD_ALIGN_PARAGRAPH.RIGHT
        header.runs[0].font.color.rgb = RGBColor(82, 99, 95)

        footer = section.footer.paragraphs[0]
        footer.alignment = WD_ALIGN_PARAGRAPH.CENTER
        add_field(footer, " PAGE ")
        section.first_page_header.paragraphs[0].text = ""
        section.first_page_footer.paragraphs[0].text = ""

    settings = doc.settings.element
    update = OxmlElement("w:updateFields")
    update.set(qn("w:val"), "true")
    settings.append(update)


def add_title_page(doc):
    paragraph = doc.add_paragraph()
    paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
    paragraph.paragraph_format.space_before = Pt(44)
    run = paragraph.add_run("BREAST CANCER ULTRASOUND CLASSIFICATION\nUSING MACHINE LEARNING")
    run.font.name = "Aptos Display"
    run.font.size = Pt(27)
    run.font.bold = True
    run.font.color.rgb = RGBColor(6, 45, 44)

    subtitle = doc.add_paragraph()
    subtitle.alignment = WD_ALIGN_PARAGRAPH.CENTER
    subtitle.paragraph_format.space_before = Pt(20)
    run = subtitle.add_run("An auditable study of preprocessing, convolutional models,\nand evidence integrity")
    run.font.name = "Times New Roman"
    run.font.size = Pt(15)
    run.font.italic = True

    metadata = doc.add_paragraph()
    metadata.alignment = WD_ALIGN_PARAGRAPH.CENTER
    metadata.paragraph_format.space_before = Pt(56)
    metadata.add_run(
        "A dissertation submitted to Manchester Metropolitan University\n"
        "for the degree of Master of Science\n\n"
        "Akerele David Damilola\n"
        "Department of Computing and Mathematics\n\n"
        "2026"
    )

    notice = doc.add_table(rows=1, cols=1)
    notice.alignment = WD_TABLE_ALIGNMENT.CENTER
    notice.autofit = False
    notice.columns[0].width = Inches(5.9)
    cell = notice.cell(0, 0)
    set_cell_fill(cell, "F4E8C8")
    cell.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER
    p = cell.paragraphs[0]
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_before = Pt(7)
    p.paragraph_format.space_after = Pt(7)
    r = p.add_run("EVIDENCE STATUS · PROVISIONAL\n")
    r.bold = True
    r.font.color.rgb = RGBColor(101, 71, 5)
    p.add_run("BrEaST and OASBUD pass the subject-level split audit; BUSI is excluded because its patient identifiers are unavailable.\nThis document makes no clinical-performance claim.")
    doc.add_page_break()


def add_front_matter(doc):
    doc.add_heading("Abstract", level=1)
    doc.add_paragraph(ABSTRACT)

    doc.add_page_break()
    doc.add_heading("Declaration", level=1)
    doc.add_paragraph(DECLARATION)
    doc.add_paragraph("\nSigned: ____________________________________")
    doc.add_paragraph("Date: ______________________________________")

    doc.add_page_break()
    doc.add_heading("Acknowledgements", level=1)
    doc.add_paragraph(
        "I am grateful to my academic supervisor, department faculty, and fellow researchers at Manchester "
        "Metropolitan University for their guidance, encouragement, and technical insight during this MSc project."
    )

    doc.add_page_break()
    doc.add_heading("Abbreviations", level=1)
    table = doc.add_table(rows=1, cols=2)
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    table.style = "Light Shading Accent 1"
    table.autofit = False
    table.columns[0].width = Inches(1.55)
    table.columns[1].width = Inches(5.05)
    table.rows[0].cells[0].text = "Abbreviation"
    table.rows[0].cells[1].text = "Meaning"
    set_repeat_table_header(table.rows[0])
    for short, meaning in ABBREVIATIONS:
        cells = table.add_row().cells
        cells[0].text = short
        cells[1].text = meaning
    for row in table.rows:
        row.cells[0].width = Inches(1.55)
        row.cells[1].width = Inches(5.05)
        for cell in row.cells:
            cell.paragraphs[0].alignment = WD_ALIGN_PARAGRAPH.LEFT

    doc.add_page_break()
    doc.add_heading("Contents", level=1)
    for title in [
        "Abstract", "Declaration", "Acknowledgements", "Abbreviations",
        "Chapter 1: Introduction", "Chapter 2: Background and Literature Review",
        "Chapter 3: Methodology", "Chapter 4: Implementation and Training Controls",
        "Chapter 5: Results and Evaluation Status", "Chapter 6: Discussion and Conclusion",
        "References", "Appendix A: Reproducibility checklist",
    ]:
        paragraph = doc.add_paragraph(title)
        paragraph.paragraph_format.left_indent = Inches(0.2)
        paragraph.paragraph_format.space_after = Pt(9)
        paragraph.alignment = WD_ALIGN_PARAGRAPH.LEFT
    doc.add_page_break()


def clean_text(text):
    replacements = {
        "\\times": "×", "\\neq": "≠", "\\max": "max", "\\%": "%",
        "\\texttt": "", "\\text": "", "\\operatorname": "", "\\mid": "|",
        "\\left": "", "\\right": "", "\\,": " ", "\\!": "",
        "\\eta_m": "ηm", "\\eta_a": "ηa", "10^-4": "10⁻⁴",
    }
    for source, target in replacements.items():
        text = text.replace(source, target)
    text = text.replace("$", "").replace("{", "").replace("}", "")
    return text


def add_inline(paragraph, text):
    text = clean_text(text)
    for part in re.split(r"(\*\*.*?\*\*|`.*?`)", text):
        if not part:
            continue
        if part.startswith("**") and part.endswith("**"):
            run = paragraph.add_run(part[2:-2])
            run.bold = True
        elif part.startswith("`") and part.endswith("`"):
            run = paragraph.add_run(part[1:-1])
            run.font.name = "Aptos Mono"
            run.font.size = Pt(10)
        else:
            paragraph.add_run(part)


def add_markdown_table(doc, lines):
    rows = [[clean_text(cell.strip()) for cell in line.strip().strip("|").split("|")] for line in lines if "---" not in line]
    if not rows:
        return
    width = max(len(row) for row in rows)
    table = doc.add_table(rows=1, cols=width)
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    table.style = "Light Shading Accent 1"
    for index, value in enumerate(rows[0]):
        table.rows[0].cells[index].text = value
        set_cell_fill(table.rows[0].cells[index], "0D6F66")
        for run in table.rows[0].cells[index].paragraphs[0].runs:
            run.font.color.rgb = RGBColor(255, 255, 255)
            run.bold = True
        table.rows[0].cells[index].paragraphs[0].alignment = WD_ALIGN_PARAGRAPH.LEFT
    set_repeat_table_header(table.rows[0])
    for values in rows[1:]:
        cells = table.add_row().cells
        for index, value in enumerate(values):
            cells[index].text = value
            for run in cells[index].paragraphs[0].runs:
                run.font.size = Pt(9)
            cells[index].paragraphs[0].alignment = WD_ALIGN_PARAGRAPH.LEFT
    doc.add_paragraph()


def add_markdown_file(doc, path):
    lines = path.read_text(encoding="utf-8").splitlines()
    table_lines = []

    def flush_table():
        nonlocal table_lines
        if table_lines:
            add_markdown_table(doc, table_lines)
            table_lines = []

    for raw in lines:
        line = raw.strip()
        if line.startswith("|"):
            table_lines.append(line)
            continue
        flush_table()
        if not line or line == "---":
            continue
        if line.startswith("### "):
            doc.add_heading(clean_text(line[4:]), level=3)
        elif line.startswith("## "):
            doc.add_heading(clean_text(line[3:]), level=2)
        elif line.startswith("# "):
            doc.add_heading(clean_text(line[2:]), level=1)
        elif line.startswith("- "):
            p = doc.add_paragraph(style="List Bullet")
            add_inline(p, line[2:])
        elif re.match(r"^\d+\.\s", line):
            p = doc.add_paragraph(style="List Number")
            add_inline(p, re.sub(r"^\d+\.\s+", "", line))
        else:
            p = doc.add_paragraph()
            add_inline(p, line)
    flush_table()


def add_figures(doc):
    doc.add_heading("Generated evaluation figures", level=2)
    for filename, caption in [
        ("confusion_matrix.png", "Figure 1. Confusion matrix for the audited 62-image checkpoint run."),
        ("roc_curve.png", "Figure 2. Receiver operating characteristic for the same audited run."),
    ]:
        path = ROOT / "outputs" / filename
        if not path.exists():
            continue
        paragraph = doc.add_paragraph()
        paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
        paragraph.add_run().add_picture(str(path), width=Inches(5.6))
        caption_p = doc.add_paragraph(caption, style="Caption")
        caption_p.alignment = WD_ALIGN_PARAGRAPH.CENTER


def tex_to_text(text):
    for source, target in {
        r'{\\"o}': "ö", r'{\\"a}': "ä", r"{\\'e}": "é", r"{\\'a}": "á",
    }.items():
        text = text.replace(source, target)
    return text.replace("{", "").replace("}", "").replace("\\", "")


def parse_bibtex(path, cited_keys):
    entries = []
    for block in re.findall(r"@\w+\{.*?\n\}", path.read_text(encoding="utf-8"), re.S):
        key_match = re.match(r"@\w+\{([^,]+),", block)
        if not key_match or key_match.group(1) not in cited_keys:
            continue
        fields = {key.lower(): value.strip().strip("{}") for key, value in re.findall(r"(\w+)\s*=\s*\{(.*?)\}\s*,?\n", block, re.S)}
        author = tex_to_text(fields.get("author", "Unknown author")).replace(" and ", "; ")
        year = fields.get("year", "n.d.")
        title = tex_to_text(fields.get("title", "Untitled"))
        source = tex_to_text(fields.get("journal") or fields.get("booktitle") or fields.get("publisher", ""))
        doi = fields.get("doi", "")
        entries.append((author.split(";")[0], f"{author} ({year}). {title}. {source}." + (f" https://doi.org/{doi}" if doi else "")))
    return [entry for _, entry in sorted(entries, key=lambda item: item[0].lower())]


def build():
    doc = Document()
    configure_document(doc)
    add_title_page(doc)
    add_front_matter(doc)
    for index, filename in enumerate(CHAPTERS):
        if index:
            doc.add_page_break()
        add_markdown_file(doc, ROOT / "docs/chapters" / filename)
        if filename == "Chapter_5_Results_And_Evaluation.md":
            add_figures(doc)

    doc.add_page_break()
    doc.add_heading("References", level=1)
    citation_text = "\n".join(path.read_text(encoding="utf-8") for path in (ROOT / "latex").rglob("*.tex"))
    cited_keys = {key.strip() for group in re.findall(r"\\citep\{([^}]+)\}", citation_text) for key in group.split(",")}
    for reference in parse_bibtex(ROOT / "latex/references.bib", cited_keys):
        paragraph = doc.add_paragraph(reference)
        paragraph.paragraph_format.first_line_indent = Inches(-0.3)
        paragraph.paragraph_format.left_indent = Inches(0.3)
        paragraph.paragraph_format.line_spacing = 1.0
        paragraph.paragraph_format.space_after = Pt(8)
        paragraph.alignment = WD_ALIGN_PARAGRAPH.LEFT

    doc.add_page_break()
    doc.add_heading("Appendix A: Reproducibility checklist", level=1)
    for item in [
        "Dataset licences, versions, checksums, exclusions, and label mapping recorded.",
        "Every subject and all associated views and masks assigned to one partition.",
        "Test manifest frozen before model selection.",
        "Seeds, environment, configuration, and checkpoint hashes saved.",
        "Per-image predictions retained for every reported run.",
        "Uncertainty, calibration, failure cases, and subgroup or site limitations reported.",
        "All written and visual artifacts regenerated from the final evidence files.",
    ]:
        paragraph = doc.add_paragraph()
        paragraph.alignment = WD_ALIGN_PARAGRAPH.LEFT
        paragraph.paragraph_format.left_indent = Inches(0.28)
        paragraph.paragraph_format.first_line_indent = Inches(-0.2)
        paragraph.add_run("• ").bold = True
        paragraph.add_run(item)

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    doc.save(OUTPUT)
    print(f"Wrote {OUTPUT}")


if __name__ == "__main__":
    build()
