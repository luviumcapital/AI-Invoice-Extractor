import os
import io
import re
import json
import csv
from dotenv import load_dotenv
from PIL import Image
import streamlit as st
import google.generativeai as genai
import fitz  # PyMuPDF
import pytesseract
import pandas as pd
import itables
from itables.streamlit import interactive_table as show_itable


# --- Settings & Presets ---
PROMPT_TEMPLATES = {
    "General Invoice": """
You are an expert invoice extraction assistant. Analyze the provided invoice (image or PDF) and extract the following fields. Output ONLY a valid JSON object, with no extra text, comments, explanations, markdown, or code blocks. Do NOT mention the word 'json' anywhere. The JSON should have the following structure:

{
  "total_amount": "",
  "invoice_number": "",
  "invoice_date": "",
  "vendor": {
    "name": "",
    "address": "",
    "contact": ""
  },
  "customer": {
    "name": "",
    "address": "",
    "contact": ""
  },
  "line_items": [
    {
      "description": "",
      "quantity": "",
      "unit_price": "",
      "total": ""
    }
  ],
  "payment_terms": "",
  "due_date": ""
}

For each line item, ensure that the fields 'description', 'quantity', 'unit_price', and 'total' are extracted as separate values. Do NOT combine multiple details (such as color, country, etc.) into the description field. If any of these fields are missing or not present in the invoice, use an empty string for that field. Do not add explanations. Do not include any text before or after the JSON. Do not use markdown or code blocks. Only output the JSON object, nothing else. Any deviation will break the downstream system.
""",
    "Minimal": """
Extract the following fields from this invoice and return ONLY a valid JSON object (no extra text, markdown, explanations, or code blocks). Do NOT mention the word 'json' anywhere:
{
  "total_amount": "",
  "invoice_number": "",
  "invoice_date": "",
  "vendor": ""
}
If a field is missing, leave it blank. Do not include any text before or after the JSON. Do not use markdown or code blocks. Only output the JSON object, nothing else. Any deviation will break the downstream system.
""",
    "Line Items Only": """
List all line items from this invoice in a JSON array. Output ONLY the JSON array, with no extra text, markdown, explanations, or code blocks. Do NOT mention the word 'json' anywhere. Each item should include:
{
  "description": "",
  "quantity": "",
  "unit_price": "",
  "total": ""
}
For each line item, ensure that the fields 'description', 'quantity', 'unit_price', and 'total' are extracted as separate values. Do NOT combine multiple details (such as color, country, etc.) into the description field. If any of these fields are missing or not present in the invoice, use an empty string for that field. Return only the JSON array. If no line items are found, return an empty array. Do not include any text before or after the JSON. Do not use markdown or code blocks. Only output the JSON array, nothing else. Any deviation will break the downstream system.
"""
}

SUPPORTED_TYPES = {
    "application/pdf": "PDF",
    "image/jpeg": "Image",
    "image/png": "Image",
    "image/jpg": "Image",
}

MAX_FILE_SIZE_MB = 10

# --- Helper Functions (extractors.py) ---

@st.cache_data(show_spinner=False)
def extract_text_from_pdf(pdf_bytes: bytes, max_pages: int = 10) -> list:
    """Extract text from a PDF file."""
    pdf_document = fitz.open(stream=pdf_bytes, filetype="pdf")
    texts = []
    for page_num in range(min(len(pdf_document), max_pages)):
        page = pdf_document.load_page(page_num)
        texts.append(page.get_text())
    return texts

@st.cache_data(show_spinner=False)
def extract_text_from_image(image_bytes: bytes) -> list:
    """Extract text from an image file."""
    image = Image.open(io.BytesIO(image_bytes))
    text = pytesseract.image_to_string(image)
    return [text]

def parse_invoice_fields(text: str) -> dict:
    """Parse invoice fields from text using regex."""
    # Simple regex-based extraction for demo purposes
    fields = {
        "Total Amount": re.search(r"(Total\s*[:\-]?\s*\$?\s*[\d,]+\.\d{2})", text, re.I),
        "Invoice Number": re.search(r"(Invoice\s*#?\s*[:\-]?\s*\w+)", text, re.I),
        "Date": re.search(r"(Date\s*[:\-]?\s*\d{1,2}[\/\-]\d{1,2}[\/\-]\d{2,4})", text, re.I),
        "Vendor": re.search(r"(From\s*[:\-]?\s*.+)", text, re.I),
    }
    return {k: (v.group(1) if v else "") for k, v in fields.items()}

def to_csv(data_dict: dict) -> str:
    """Convert a dictionary to a CSV string."""
    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=data_dict.keys())
    writer.writeheader()
    writer.writerow(data_dict)
    return output.getvalue()

# --- Helper Functions (ai.py) ---

def load_api_key() -> str:
    """Load the Google API key from environment."""
    load_dotenv()
    return os.getenv("GOOGLE_API_KEY")

def configure_gemini(api_key: str):
    """Configure the Gemini AI model."""
    genai.configure(api_key=api_key)
    return genai.GenerativeModel('gemini-2.0-flash')

def get_gemini_response(model, input_text: str, prompt: str) -> str:
    """Get a response from the Gemini model."""
    try:
        parts = []
        if prompt:
            parts.append({"text": prompt})
        if input_text:
            parts.append({"text": input_text})
        response = model.generate_content({"parts": parts})
        return response.text
    except Exception as e:
        return f"An error occurred: {str(e)}"

# --- UI Functions (ui.py) ---

def sidebar_settings():
    """Display sidebar settings and return prompt type and uploaded file."""
    st.sidebar.header("Settings")
    prompt_type = st.sidebar.selectbox("Prompt Template", list(PROMPT_TEMPLATES.keys()))
    uploaded_file = st.sidebar.file_uploader(
        "Upload Invoice (PDF, JPG, PNG)",
        type=["pdf", "jpg", "jpeg", "png"],
        label_visibility="collapsed"
    )
    if uploaded_file:
        # Validate file size
        if uploaded_file.size > MAX_FILE_SIZE_MB * 1024 * 1024:
            st.sidebar.error(f"File too large! Please upload a file smaller than {MAX_FILE_SIZE_MB} MB.")
        else:
            st.sidebar.markdown(f"**File Name:** {uploaded_file.name}")
            st.sidebar.markdown(f"**File Size:** {uploaded_file.size / (1024 * 1024):.2f} MB")
    else:
        st.sidebar.info("Upload an invoice to get started.")
    return prompt_type, uploaded_file

def show_extracted_text(texts: list) -> None:
    """Show extracted text in the UI."""
    st.subheader("Extracted Text")
    if len(texts) > 1:
        page = st.number_input("Page", min_value=1, max_value=len(texts), value=1)
        st.text_area("Text", value=texts[page-1], height=200)
    else:
        st.text_area("Text", value=texts[0], height=200)

def show_invoice_table(fields: dict) -> None:
    """Show detected invoice fields as a table."""
    st.subheader("Detected Invoice Fields")
    st.table([fields])

def show_download_buttons(fields: dict) -> None:
    """Show download buttons for CSV and JSON."""
    csv_data = to_csv(fields)
    json_data = json.dumps(fields, indent=2)
    st.download_button("Download as CSV", csv_data, file_name="invoice.csv", mime="text/csv")
    st.download_button("Download as JSON", json_data, file_name="invoice.json", mime="application/json")

def flatten_json(y, parent_key: str = '', sep: str = '.') -> dict:
    """Flatten nested JSON for table display and CSV export."""
    items = []
    if isinstance(y, list):
        for i, v in enumerate(y):
            items.extend(flatten_json(v, f"{parent_key}{sep}{i}" if parent_key else str(i), sep=sep).items())
    elif isinstance(y, dict):
        for k, v in y.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            items.extend(flatten_json(v, new_key, sep=sep).items())
    else:
        items.append((parent_key, y))
    return dict(items)

def show_json_table(json_data):
    """Show JSON as a fully exploded interactive table using itables: each line item is a row, with invoice fields repeated. Enhanced presentation."""
    if isinstance(json_data, list):
        # For each invoice, create a row for each line item (or a single row if no line items)
        rows = []
        for invoice in json_data:
            base = {k: v for k, v in invoice.items() if k not in ["line_items"]}
            line_items = invoice.get("line_items", [])
            if line_items:
                for item in line_items:
                    row = base.copy()
                    for k, v in item.items():
                        row[f"line_{k}"] = v
                    rows.append(row)
            else:
                rows.append(base)
        df = pd.DataFrame(rows)
        st.markdown("""
        <div style='font-size:1.5rem;font-weight:700;color:#2d6cdf;margin-bottom:0.5em;'>Line Items</div>
        """, unsafe_allow_html=True)
        # Warn if most line items are missing key fields
        if not df.empty and (df[['quantity', 'unit_price', 'total']].isnull() | (df[['quantity', 'unit_price', 'total']] == '')).all(axis=None):
            st.warning("Most line items are missing 'quantity', 'unit_price', or 'total'. The extraction model may not be parsing these fields correctly. Check your prompt or invoice format.")
        show_itable(df, maxBytes=0, lengthMenu=[[10, 25, 50, 100, -1], [10, 25, 50, 100, 'All']])
    else:
        # Single invoice dict (previous logic)
        if "line_items" in json_data and isinstance(json_data["line_items"], list):
            st.markdown("""
            <div style='font-size:1.5rem;font-weight:700;color:#2d6cdf;margin-bottom:0.5em;'>Line Items</div>
            """, unsafe_allow_html=True)
            if json_data["line_items"]:
                df = pd.DataFrame(json_data["line_items"])
                # Warn if most line items are missing key fields
                if not df.empty and (df[['quantity', 'unit_price', 'total']].isnull() | (df[['quantity', 'unit_price', 'total']] == '')).all(axis=None):
                    st.warning("Most line items are missing 'quantity', 'unit_price', or 'total'. The extraction model may not be parsing these fields correctly. Check your prompt or invoice format.")
                show_itable(df, maxBytes=0, lengthMenu=[[10, 25, 50, 100, -1], [10, 25, 50, 100, 'All']])
            else:
                st.info("No line items found.")
            data = {k: v for k, v in json_data.items() if k != "line_items"}
        else:
            data = json_data
        st.markdown("""
        <div style='font-size:1.3rem;font-weight:600;color:#2d6cdf;margin-bottom:0.3em;'>Invoice Fields</div>
        """, unsafe_allow_html=True)
        flat = flatten_json(data)
        df = pd.DataFrame([flat])
        show_itable(df, maxBytes=0)

# --- Main App ---

st.set_page_config(page_title="AI Invoice Extractor", page_icon="🧾", layout="wide", initial_sidebar_state="expanded")

# Add a custom header with color and emoji
st.markdown("""
<style>
.big-title {
    font-size: 2.5rem;
    font-weight: 700;
    color: #2d6cdf;
    margin-bottom: 0.2em;
}
.subtext {
    font-size: 1.1rem;
    color: #444;
    margin-bottom: 1.5em;
}
.stButton>button {
    background-color: #2d6cdf;
    color: white;
    font-weight: 600;
    border-radius: 6px;
    padding: 0.5em 1.5em;
    margin-top: 0.5em;
}
.stDownloadButton>button {
    background-color: #e0e7ff;
    color: #2d6cdf;
    font-weight: 600;
    border-radius: 6px;
    margin-right: 0.5em;
}
.stTable, .stDataFrame, .itables-container {
    background: #f8fafc;
    border-radius: 8px;
    padding: 1em;
    margin-bottom: 1.5em;
}
</style>
<div class="big-title">🧾 AI Invoice Extractor</div>
<div class="subtext">Extract invoice data from PDFs and images using AI. Enjoy interactive tables and easy downloads!</div>
""", unsafe_allow_html=True)

with st.sidebar:
    st.markdown("""
    <div style='font-size:1.2rem;font-weight:600;color:#2d6cdf;'>Quick Start</div>
    <ol style='margin-top:0.5em;'>
      <li>Upload an invoice (PDF or image).</li>
      <li>Select the extraction prompt.</li>
      <li>Click <b>Analyze Invoice</b>.</li>
    </ol>
    <hr style='margin:0.7em 0;'>
    <div style='color:#444;'>
    <b>About:</b> This app uses Google Gemini AI to extract invoice data and display it in interactive tables. Download your results as CSV or JSON.
    </div>
    """, unsafe_allow_html=True)

prompt_type, uploaded_file = sidebar_settings()
input_prompt = PROMPT_TEMPLATES[prompt_type]
api_key = load_api_key()

if uploaded_file:
    # Validate file type
    if uploaded_file.type not in SUPPORTED_TYPES:
        st.error("Unsupported file type. Please upload a PDF or image (JPG, PNG).")
        st.stop()
    # Validate file size (again, for main area)
    if uploaded_file.size > MAX_FILE_SIZE_MB * 1024 * 1024:
        st.error(f"File too large! Max size is {MAX_FILE_SIZE_MB} MB. Please upload a smaller file.")
        st.stop()
    file_type = uploaded_file.type
    st.info(f"Detected file type: {SUPPORTED_TYPES.get(file_type, 'Unknown')}")
    with st.spinner("Extracting text from your invoice. Please wait..."):
        try:
            if file_type == "application/pdf":
                texts = extract_text_from_pdf(uploaded_file.read())
            elif file_type in ["image/jpeg", "image/png", "image/jpg"]:
                texts = extract_text_from_image(uploaded_file.read())
            else:
                st.error("Unsupported file type. Please upload a PDF or image (JPG, PNG).")
                st.stop()
        except Exception as e:
            st.error(f"Sorry, we couldn't extract text from your file. Please check the file format or try another file.\nError: {str(e)}")
            st.stop()
    if not texts or all(not t.strip() for t in texts):
        st.warning("No text could be extracted from your file. Please check the file quality or try another document.")
        st.stop()
    show_extracted_text(texts)
    # Highlight fields
    fields = parse_invoice_fields(" ".join(texts))
    show_invoice_table(fields)
    show_download_buttons(fields)
    st.session_state["extracted_text"] = " ".join(texts)
    st.session_state["fields"] = fields

def extract_json_from_response(response_text: str) -> str:
    """Extract the first JSON object or array from a string, ignoring markdown/code blocks and extra text."""
    # Remove markdown code block markers if present
    cleaned = re.sub(r'^\s*```[a-zA-Z]*', '', response_text.strip())
    cleaned = re.sub(r'```\s*$', '', cleaned)
    # Find the first JSON object or array
    match = re.search(r'({[\s\S]*}|\[[\s\S]*\])', cleaned)
    if match:
        return match.group(1)
    return response_text  # fallback: return as-is

if st.button("Analyze Invoice", disabled=not (uploaded_file and api_key)):
    if not api_key:
        st.error("API key required. Please set your Google API key in the .env file.")
    elif not uploaded_file:
        st.error("Please upload an invoice file to analyze.")
    else:
        model = configure_gemini(api_key)
        with st.spinner("Analyzing your invoice with Gemini AI. This may take a few seconds..."):
            try:
                response = get_gemini_response(model, st.session_state.get("extracted_text", ""), input_prompt)
            except Exception as e:
                st.error(f"Sorry, there was an error communicating with the AI model. Please try again later.\nError: {str(e)}")
                st.stop()
        try:
            cleaned_response = extract_json_from_response(response)
            processed_json = json.loads(cleaned_response)
            st.session_state['gemini_result'] = processed_json  # Store result in session state
        except Exception:
            st.session_state['gemini_result'] = None
            st.session_state['gemini_raw'] = response
            st.error("The AI response could not be parsed as valid JSON. Please check the raw output below or try a different prompt.")

# --- Always show the latest Gemini result if available ---
if 'gemini_result' in st.session_state and st.session_state['gemini_result'] is not None:
    show_json_table(st.session_state['gemini_result'])
    # --- Improved Download Buttons for Gemini Output ---
    processed_json = st.session_state['gemini_result']
    if isinstance(processed_json, list):
        rows = []
        for invoice in processed_json:
            base = {k: v for k, v in invoice.items() if k not in ["line_items"]}
            line_items = invoice.get("line_items", [])
            if line_items:
                for item in line_items:
                    row = base.copy()
                    for k, v in item.items():
                        row[f"line_{k}"] = v
                    rows.append(row)
            else:
                rows.append(base)
        csv_data = pd.DataFrame(rows).to_csv(index=False)
        json_data = json.dumps(processed_json, indent=2)
    else:
        if "line_items" in processed_json and isinstance(processed_json["line_items"], list):
            rows = []
            base = {k: v for k, v in processed_json.items() if k != "line_items"}
            for item in processed_json["line_items"]:
                row = base.copy()
                for k, v in item.items():
                    row[f"line_{k}"] = v
                rows.append(row)
            csv_data = pd.DataFrame(rows).to_csv(index=False)
        else:
            csv_data = pd.DataFrame([flatten_json(processed_json)]).to_csv(index=False)
        json_data = json.dumps(processed_json, indent=2)
    st.download_button("Download as CSV", csv_data, file_name="invoice_gemini.csv", mime="text/csv")
    st.download_button("Download as JSON", json_data, file_name="invoice_gemini.json", mime="application/json")
elif 'gemini_result' in st.session_state and st.session_state['gemini_result'] is None and 'gemini_raw' in st.session_state:
    st.info("Gemini output is not valid JSON. See raw output below for debugging.")
    st.code(st.session_state['gemini_raw'], language="json")

# --- Footer: Reference to GitHub Repo ---
st.markdown("""
<hr style='margin-top:2em;'>
<div style='text-align:center;font-size:1rem;color:#888;'>
  View source or contribute on <a href='https://github.com/pratstick/AI-Invoice-Extractor' target='_blank' style='color:#2d6cdf;text-decoration:underline;'>GitHub</a>.<br>
  <span style='font-size:0.95rem;color:#aaa;'>Powered by <span style='color:#4285F4;font-weight:600;'>Google Gemini</span></span>
</div>
""", unsafe_allow_html=True)


# Streamlit runs as a script, so no need for __main__ guard.
# This block is intentionally left empty.

