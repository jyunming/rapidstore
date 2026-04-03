import pymupdf4llm
import os

pdf_path = "2504.19874v1.pdf"
md_path = "2504.19874v1.md"

if not os.path.exists(pdf_path):
    print(f"Error: {pdf_path} not found.")
    exit(1)

print(f"Converting {pdf_path} to {md_path}...")
md_text = pymupdf4llm.to_markdown(pdf_path)

with open(md_path, "w", encoding="utf-8") as f:
    f.write(md_text)

print("Conversion successful.")
