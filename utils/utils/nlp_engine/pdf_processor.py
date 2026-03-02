import pdfplumber
from utils.helpers import clean_text

class PDFProcessor:
    @staticmethod
    def extract_text(pdf_path):
        """استخراج النص من PDF"""
        text = ""
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
        return clean_text(text)
