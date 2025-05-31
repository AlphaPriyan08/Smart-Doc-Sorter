import pdfplumber
import os 

def extract_text_from_pdf(file_path):
    text = ""
    if not os.path.exists(file_path):
        print(f"PDF Parser: File not found at {file_path}")
        return None

    try:
        with pdfplumber.open(file_path) as pdf:
            if not pdf.pages:
                print(f"PDF Parser: No pages found in PDF: {file_path}")
                return "" 
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
        if not text.strip():
            print(f"PDF Parser: No text extracted from PDF (pages might be images or empty): {file_path}")
        return text
    except Exception as e: 
        print(f"PDF Parser: Error opening or parsing PDF {file_path}: {e}")
        raise 