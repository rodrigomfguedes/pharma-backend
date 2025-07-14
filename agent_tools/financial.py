from langchain.tools import StructuredTool
import fitz
from dotenv import load_dotenv

load_dotenv()

def extract_text_from_pages(pdf_path:str, pages:list):
    """
    Extracts plain text from specific pages in a PDF.

    :param pdf_path: str – path to your PDF file
    :param pages: list of ints – page numbers you want (1-based)
    :return: dict mapping page number → extracted text
    """
    doc = fitz.open(pdf_path)
    result = {}

    for human_page in pages:
        idx = human_page - 1  # convert to zero-based index
        if idx < 0 or idx >= doc.page_count:
            print(f"Page out of range. Skipping.")
            continue

        page = doc.load_page(idx)  # get the page
        text = page.get_text("text")  # extract plain text
        result[human_page] = text

    doc.close()
    return result

def financial_report():
    # Extract text from main financial analysis pages in the 24/25 annual report (contains P&L)
    financial_context = extract_text_from_pages("docs/grunenthal-annual-report24-25.pdf", [30,31,32])
    return financial_context


financial_report_tool = StructuredTool.from_function(
    func=financial_report,
    name="FinancialReportRetriever",
    description="Gets information about Grunenthal's financial report 24/25",
)