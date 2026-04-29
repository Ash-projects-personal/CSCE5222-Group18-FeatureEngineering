"""Merge README PDF + Code PDF into a single file."""
from pypdf import PdfReader, PdfWriter

writer = PdfWriter()

for path in [
    '/home/ubuntu/project2/code/Colab_README.pdf',
    '/home/ubuntu/upload/Group18_Code.pdf',
]:
    reader = PdfReader(path)
    for page in reader.pages:
        writer.add_page(page)

out_path = '/home/ubuntu/project2/code/Group18_Code_Final.pdf'
with open(out_path, 'wb') as f:
    writer.write(f)

reader = PdfReader(out_path)
print(f"Merged PDF: {out_path}")
print(f"Total pages: {len(reader.pages)}")
