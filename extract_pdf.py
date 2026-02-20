
import pypdf
import os

pdf_files = ["ML Engineer Assignment.pdf", "ML Final Assignment.pdf"]

for pdf_file in pdf_files:
    if os.path.exists(pdf_file):
        print(f"--- START OF {pdf_file} ---")
        try:
            reader = pypdf.PdfReader(pdf_file)
            for page in reader.pages:
                print(page.extract_text())
        except Exception as e:
            print(f"Error reading {pdf_file}: {e}")
        print(f"--- END OF {pdf_file} ---")
    else:
        print(f"File not found: {pdf_file}")
