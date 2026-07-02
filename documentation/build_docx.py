import os
import re
from docx import Document
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement, parse_xml
from docx.oxml.ns import nsdecls, qn

def create_element(name):
    return OxmlElement(name)

def set_cell_background(cell, hex_color):
    shading_xml = f'<w:shd {nsdecls("w")} w:fill="{hex_color}"/>'
    cell._tc.get_or_add_tcPr().append(parse_xml(shading_xml))

def build_dissertation_docx():
    print("Compiling Dissertation Chapters into Word (.docx)...")
    
    doc = Document()
    
    # Page Margins Configuration
    for section in doc.sections:
        section.top_margin = Inches(1)
        section.bottom_margin = Inches(1)
        section.left_margin = Inches(1)
        section.right_margin = Inches(1)
        
    # Styles Configuration
    styles = doc.styles
    
    # Title Page Styles
    title_p = doc.add_paragraph()
    title_p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    title_run = title_p.add_run("UNIVERSITY DISSERTATION THESIS\n\n\n")
    title_run.font.name = "Georgia"
    title_run.font.size = Pt(14)
    title_run.font.bold = True
    
    h1_p = doc.add_paragraph()
    h1_p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    h1_run = h1_p.add_run("Deep Convolutional Neural Networks and Contrast Limited Adaptive Histogram Equalization (CLAHE) for Automated Breast Cancer Mammography Classification\n\n\n\n\n")
    h1_run.font.name = "Georgia"
    h1_run.font.size = Pt(22)
    h1_run.font.bold = True
    h1_run.font.color.rgb = RGBColor(11, 25, 44)
    
    author_p = doc.add_paragraph()
    author_p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    author_run = author_p.add_run("Author: David Akerele\n\n\n\n")
    author_run.font.name = "Georgia"
    author_run.font.size = Pt(12)
    author_run.font.bold = True
    
    dept_p = doc.add_paragraph()
    dept_p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    dept_run = dept_p.add_run("Department of Computer Science & Artificial Intelligence\n")
    dept_run.font.name = "Georgia"
    dept_run.font.size = Pt(11)
    dept_run.font.italic = True
    
    doc.add_page_break()
    
    # Chapters List in order
    chapter_files = [
        "Chapter_1_Introduction.md",
        "Chapter_2_Literature_Review.md",
        "Chapter_3_Methodology.md",
        "Chapter_4_Implementation_And_Training.md",
        "Chapter_5_Results_And_Evaluation.md",
        "Chapter_6_Discussion_And_Conclusion.md"
    ]
    
    doc_dir = os.path.dirname(os.path.abspath(__file__))
    
    for filename in chapter_files:
        filepath = os.path.join(doc_dir, filename)
        if not os.path.exists(filepath):
            print(f"Skipping missing chapter: {filename}")
            continue
            
        print(f"Adding: {filename}...")
        
        with open(filepath, "r", encoding="utf-8") as f:
            lines = f.readlines()
            
        in_table = False
        table_headers = []
        table_rows = []
        
        for line in lines:
            stripped = line.strip()
            
            # Skip divider bars
            if stripped == "---":
                continue
                
            # Detect Markdown Table
            if stripped.startswith("|"):
                # Handle separator rows e.g. |:---|:---:|
                if "---" in stripped:
                    continue
                cells = [c.strip() for c in stripped.split("|")[1:-1]]
                if not in_table:
                    in_table = True
                    table_headers = cells
                else:
                    table_rows.append(cells)
                continue
            elif in_table:
                # We left the table, write it to Word document now
                create_word_table(doc, table_headers, table_rows)
                in_table = False
                table_headers = []
                table_rows = []
                
            # Headings
            if stripped.startswith("# "):
                title_text = stripped[2:]
                h = doc.add_paragraph()
                h.paragraph_format.space_before = Pt(24)
                h.paragraph_format.space_after = Pt(12)
                h.paragraph_format.keep_with_next = True
                
                run = h.add_run(title_text)
                run.font.name = "Georgia"
                run.font.size = Pt(18)
                run.font.bold = True
                run.font.color.rgb = RGBColor(11, 25, 44)
                
            elif stripped.startswith("## "):
                subtitle_text = stripped[3:]
                h = doc.add_paragraph()
                h.paragraph_format.space_before = Pt(16)
                h.paragraph_format.space_after = Pt(8)
                h.paragraph_format.keep_with_next = True
                
                run = h.add_run(subtitle_text)
                run.font.name = "Georgia"
                run.font.size = Pt(14)
                run.font.bold = True
                run.font.color.rgb = RGBColor(27, 38, 59)
                
            elif stripped.startswith("### "):
                sub_text = stripped[4:]
                h = doc.add_paragraph()
                h.paragraph_format.space_before = Pt(12)
                h.paragraph_format.space_after = Pt(6)
                h.paragraph_format.keep_with_next = True
                
                run = h.add_run(sub_text)
                run.font.name = "Georgia"
                run.font.size = Pt(12)
                run.font.bold = True
                run.font.color.rgb = RGBColor(65, 90, 119)
                
            # Lists
            elif stripped.startswith("- "):
                list_text = stripped[2:]
                p = doc.add_paragraph(style='List Bullet')
                p.paragraph_format.space_after = Pt(4)
                
                add_formatted_runs(p, list_text)
                
            elif re.match(r"^\d+\.\s+", stripped):
                list_text = re.sub(r"^\d+\.\s+", "", stripped)
                p = doc.add_paragraph(style='List Number')
                p.paragraph_format.space_after = Pt(4)
                
                add_formatted_runs(p, list_text)
                
            # Paragraph text
            elif stripped:
                p = doc.add_paragraph()
                p.paragraph_format.space_after = Pt(10)
                p.paragraph_format.line_spacing = 1.15
                
                add_formatted_runs(p, stripped)
                
        # Handle trailing table if file ends
        if in_table:
            create_word_table(doc, table_headers, table_rows)
            
        # Add Page Break between Chapters
        doc.add_page_break()
        
    out_path = os.path.join(doc_dir, "Breast_Cancer_Mammography_Dissertation.docx")
    doc.save(out_path)
    print(f"Success! Saved dissertation draft to {out_path}")

def add_formatted_runs(paragraph, text):
    """Parses simple inline markdown tags (bold **, italics *, LaTeX math $$ / $) into docx runs"""
    # Clean LaTeX math signs for readable Word text format
    text = text.replace("$$", "").replace("$", "")
    
    # Parse bold indicators (**)
    pattern = r"(\*\*[^*]+\*\*)"
    parts = re.split(pattern, text)
    
    for part in parts:
        if part.startswith("**") and part.endswith("**"):
            run = paragraph.add_run(part[2:-2])
            run.font.bold = True
        else:
            run = paragraph.add_run(part)
            
        run.font.name = "Arial"
        run.font.size = Pt(11)

def create_word_table(doc, headers, rows):
    """Generates a beautifully styled MS Word table for quantitative dissertation matrices"""
    table = doc.add_table(rows=1, cols=len(headers))
    table.autofit = False
    
    # Style Header Row
    hdr_cells = table.rows[0].cells
    for i, header in enumerate(headers):
        hdr_cells[i].text = header
        set_cell_background(hdr_cells[i], "0B192C")
        
        # Header text styling
        p = hdr_cells[i].paragraphs[0]
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        for run in p.runs:
            run.font.name = "Georgia"
            run.font.size = Pt(10)
            run.font.bold = True
            run.font.color.rgb = RGBColor(255, 255, 255)
            
    # Add Data Rows
    for r_idx, row_cells in enumerate(rows):
        row = table.add_row()
        cells = row.cells
        for col_idx, text in enumerate(row_cells):
            cells[col_idx].text = text
            
            # Zebra striping backgrounds
            if r_idx % 2 == 1:
                set_cell_background(cells[col_idx], "F8F9FA")
            else:
                set_cell_background(cells[col_idx], "FFFFFF")
                
            p = cells[col_idx].paragraphs[0]
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            for run in p.runs:
                run.font.name = "Arial"
                run.font.size = Pt(9.5)
                run.font.color.rgb = RGBColor(50, 50, 50)
                
    # Add spacing after tables
    doc.add_paragraph().paragraph_format.space_after = Pt(12)

if __name__ == "__main__":
    build_dissertation_docx()
