"""Merge README + Code PDF, and README + Presentation PDF."""
from pypdf import PdfReader, PdfWriter
import os

def merge(parts, out_path):
    writer = PdfWriter()
    for path in parts:
        reader = PdfReader(path)
        for page in reader.pages:
            writer.add_page(page)
    with open(out_path, 'wb') as f:
        writer.write(f)
    total = len(PdfReader(out_path).pages)
    print(f"  {os.path.basename(out_path)}: {total} pages")

print("Merging Code README + Code PDF...")
merge(
    [
        '/home/ubuntu/project2/code/Code_README.pdf',
        '/home/ubuntu/upload/Group18_Code.pdf',
    ],
    '/home/ubuntu/project2/code/Group18_Code_Final.pdf'
)

print("Merging Presentation README + Presentation PDF...")
merge(
    [
        '/home/ubuntu/project2/slides/Presentation_README.pdf',
        '/home/ubuntu/project2/slides/Group18_UNT_Presentation.pdf',
    ],
    '/home/ubuntu/project2/slides/Group18_Presentation_Final.pdf'
)

print("\nDone.")
