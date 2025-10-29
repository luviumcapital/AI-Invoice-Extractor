import os
import io
import re
import json
import csv
import requests
from dotenv import load_dotenv
from PIL import Image
import streamlit as st
import google.generativeai as genai
import fitz  # PyMuPDF
import pytesseract
import pandas as pd
import itables
from itables.streamlit import interactive_table as show_itable

# Load environment variables
load_dotenv()

# --- Dolibarr Integration Function ---
def send_to_dolibarr(invoice_json):
    """Send extracted invoice JSON to Dolibarr API."""
    try:
        url = os.environ.get("DOLIBARR_API_URL")
        api_key = os.environ.get("DOLIBARR_API_KEY")
        
        if not url or not api_key:
            return {"error": "Missing DOLIBARR_API_URL or DOLIBARR_API_KEY environment variables"}
        
        headers = {
            "DOLAPIKEY": api_key,
            "Content-Type": "application/json"
        }
        
        # Prepare invoice data for Dolibarr API
        dolibarr_payload = {
            "ref": invoice_json.get("invoice_number", ""),
            "date": invoice_json.get("invoice_date", ""),
            "amount": invoice_json.get("total_amount", 0),
            "notes": f"Vendor: {invoice_json.get('vendor', {}).get('name', '')}\nCustomer: {invoice_json.get('customer', {}).get('name', '')}"
        }
        
        resp = requests.post(
            f"{url}/api/index.php/invoices",
            headers=headers,
            json=dolibarr_payload,
            timeout=10
        )
        
        return resp.json() if resp.text else {"status": "success", "message": "Invoice sent to Dolibarr"}
    except Exception as e:
        return {"error": str(e)}

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
If a field is missing, leave it blank. Do not include any text before or after the JSON. Do not use markdown or code blocks. Only output the JSON array, nothing else. Any deviation will break the downstream system.
"""
}

# --- Initialize Session State ---
if "api_key" not in st.session_state:
    st.session_state.api_key = os.environ.get("GEMINI_API_KEY", "")
if "extracted_data" not in st.session_state:
    st.session_state.extracted_data = None
if "extraction_history" not in st.session_state:
    st.session_state.extraction_history = []

load_dotenv()

# Configure the Gemini API
if st.session_state.api_key:
    genai.configure(api_key=st.session_state.api_key)
else:
    genai.configure(api_key=os.environ.get("GEMINI_API_KEY"))

# --- Streamlit Page Config ---
st.set_page_config(
    page_title="AI Invoice Extractor + Dolibarr",
    page_icon="🧾",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("🧾 AI Invoice Extractor + Dolibarr Integration")
st.write("Extract invoice data using AI and automatically send to Dolibarr")

# --- Sidebar Configuration ---
with st.sidebar:
    st.header("⚙️ Configuration")
    st.session_state.api_key = st.text_input("Enter Gemini API Key:", value=st.session_state.api_key, type="password")
    prompt_template = st.selectbox("Select Extraction Template:", list(PROMPT_TEMPLATES.keys()))
    st.divider()
    st.subheader("📋 Dolibarr Settings")
    dolibarr_url = st.text_input("Dolibarr API URL:", value=os.environ.get("DOLIBARR_API_URL", ""))
    dolibarr_key = st.text_input("Dolibarr API Key:", value=os.environ.get("DOLIBARR_API_KEY", ""), type="password")

# --- Main Content Area ---
tab1, tab2, tab3 = st.tabs(["📤 Upload & Extract", "📊 View Results", "📜 History"])

with tab1:
    st.subheader("Upload Invoice Document")
    uploaded_file = st.file_uploader(
        "Upload PDF or Image:",
        type=["pdf", "jpg", "jpeg", "png", "bmp"],
        help="Supported formats: PDF, JPG, PNG, BMP"
    )
    
    if uploaded_file:
        st.success(f"✅ File uploaded: {uploaded_file.name}")
        
        # Extract text from uploaded file
        if st.button("🚀 Extract Invoice Data", use_container_width=True):
            if not st.session_state.api_key:
                st.error("❌ Please enter your Gemini API Key")
            else:
                try:
                    with st.spinner("🔄 Extracting invoice data..."):
                        # Process file based on type
                        if uploaded_file.type == "application/pdf":
                            pdf_document = fitz.open(stream=uploaded_file.read(), filetype="pdf")
                            images = []
                            for page_num in range(pdf_document.page_count):
                                page = pdf_document[page_num]
                                pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
                                images.append(Image.frombytes("RGB", [pix.width, pix.height], pix.samples))
                        else:
                            images = [Image.open(uploaded_file)]
                        
                        # Extract using Gemini
                        model = genai.GenerativeModel("gemini-1.5-flash")
                        prompt = PROMPT_TEMPLATES.get(prompt_template)
                        
                        extracted_json_str = ""
                        for image in images:
                            response = model.generate_content([prompt, image])
                            extracted_json_str += response.text
                        
                        # Parse JSON
                        try:
                            extracted_data = json.loads(extracted_json_str)
                            st.session_state.extracted_data = extracted_data
                            st.session_state.extraction_history.append(extracted_data)
                            st.success("✅ Invoice data extracted successfully!")
                        except json.JSONDecodeError as e:
                            st.error(f"❌ Failed to parse extracted JSON: {e}")
                            st.text_area("Raw Extraction:", extracted_json_str)
                except Exception as e:
                    st.error(f"❌ Error during extraction: {e}")

with tab2:
    st.subheader("Extracted Invoice Data")
    
    if st.session_state.extracted_data:
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.json(st.session_state.extracted_data)
        
        with col2:
            st.write("**Quick Actions:**")
            # Download as JSON
            json_str = json.dumps(st.session_state.extracted_data, indent=2)
            st.download_button(
                label="📥 Download JSON",
                data=json_str,
                file_name="extracted_invoice.json",
                mime="application/json",
                use_container_width=True
            )
            
            # Download as CSV
            if "line_items" in st.session_state.extracted_data:
                df = pd.DataFrame(st.session_state.extracted_data["line_items"])
                csv_str = df.to_csv(index=False)
                st.download_button(
                    label="📥 Download CSV",
                    data=csv_str,
                    file_name="invoice_items.csv",
                    mime="text/csv",
                    use_container_width=True
                )
            
            # Send to Dolibarr
            st.divider()
            st.write("**Send to Dolibarr:**")
            if st.button("🚀 Send to Dolibarr", use_container_width=True):
                if not dolibarr_url or not dolibarr_key:
                    st.error("❌ Please configure Dolibarr API credentials in the sidebar")
                else:
                    with st.spinner("📤 Sending to Dolibarr..."):
                        # Temporarily set environment variables for send_to_dolibarr function
                        os.environ["DOLIBARR_API_URL"] = dolibarr_url
                        os.environ["DOLIBARR_API_KEY"] = dolibarr_key
                        
                        response = send_to_dolibarr(st.session_state.extracted_data)
                        
                        if "error" in response:
                            st.error(f"❌ Error: {response['error']}")
                        else:
                            st.success("✅ Invoice sent to Dolibarr successfully!")
                            st.json(response)
        
        st.divider()
        st.subheader("📋 Line Items Table")
        if "line_items" in st.session_state.extracted_data and st.session_state.extracted_data["line_items"]:
            df = pd.DataFrame(st.session_state.extracted_data["line_items"])
            show_itable(df)
        else:
            st.info("No line items found in extraction")
    else:
        st.info("👆 Upload and extract invoice data from the 'Upload & Extract' tab")

with tab3:
    st.subheader("Extraction History")
    if st.session_state.extraction_history:
        for idx, data in enumerate(reversed(st.session_state.extraction_history), 1):
            with st.expander(f"📌 Extraction #{len(st.session_state.extraction_history) - idx + 1} - Invoice: {data.get('invoice_number', 'N/A')}"):
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.json(data)
                with col2:
                    if st.button("🗑️ Delete", key=f"delete_{idx}"):
                        st.session_state.extraction_history.pop(len(st.session_state.extraction_history) - idx)
                        st.rerun()
    else:
        st.info("No extraction history yet")

st.divider()
st.caption("🔐 Your API keys are never stored. Environment variables are used for security.")
