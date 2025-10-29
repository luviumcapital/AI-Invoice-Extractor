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
load_dotenv()
# Read defaults from environment with persistent fallbacks
DEFAULT_GEMINI_API_KEY = os.getenv('GEMINI_API_KEY', 'AIzaSyCXUx4IJan48xLgAoEoturQWuQY8wSnELE')
DEFAULT_BYTEZ_API_URL = os.getenv('BYTEZ_API_URL', 'https://api.bytez.example/v1/extract_invoice')
DEFAULT_BYTEZ_API_KEY = os.getenv('BYTEZ_API_KEY', 'bytez_sample_key')
DEFAULT_DOLIBARR_API_URL = os.getenv('DOLIBARR_API_URL', 'http://localhost/dolibarr/api/index.php')
DEFAULT_DOLIBARR_API_KEY = os.getenv('DOLIBARR_API_KEY', 'api_key_admin_2025')
# Init session state with defaults so sidebar is prefilled and persists across reruns
if 'gemini_api_key' not in st.session_state:
    st.session_state.gemini_api_key = DEFAULT_GEMINI_API_KEY
if 'bytez_api_url' not in st.session_state:
    st.session_state.bytez_api_url = DEFAULT_BYTEZ_API_URL
if 'bytez_api_key' not in st.session_state:
    st.session_state.bytez_api_key = DEFAULT_BYTEZ_API_KEY
if 'dolibarr_url' not in st.session_state:
    st.session_state.dolibarr_url = DEFAULT_DOLIBARR_API_URL
if 'dolibarr_key' not in st.session_state:
    st.session_state.dolibarr_key = DEFAULT_DOLIBARR_API_KEY
# --- Dolibarr Integration Function ---
def send_to_dolibarr(invoice_json):
    """Send extracted invoice JSON to Dolibarr API."""
    try:
        url = os.environ.get("DOLIBARR_API_URL")
        api_key = os.environ.get("DOLIBARR_API_KEY")
        if not url or not api_key:
            return {"error": "Missing DOLIBARR_API_URL or DOLIBARR_API_KEY environment variables"}
        headers = {"DOLAPIKEY": api_key, "Content-Type": "application/json"}
        dolibarr_payload = {
            "ref": invoice_json.get("invoice_number", ""),
            "date": invoice_json.get("invoice_date", ""),
            "amount": invoice_json.get("total_amount", 0),
            "notes": f"Vendor: {invoice_json.get('vendor', {}).get('name', '')}\nCustomer: {invoice_json.get('customer', {}).get('name', '')}"
        }
        resp = requests.post(
            f"{url}/invoices" if url.endswith('/api/index.php') else f"{url}/api/index.php/invoices",
            headers=headers,
            json=dolibarr_payload,
            timeout=10
        )
        return resp.json() if resp.text else {"status": "success", "message": "Invoice sent to Dolibarr"}
    except Exception as e:
        return {"error": str(e)}
# --- Settings & Presets ---
# Sidebar for API keys/config
with st.sidebar:
    st.header("🔑 API Keys & Endpoints")
    st.text_input("Gemini API Key", key="gemini_api_key")
    st.text_input("Bytez API Key", key="bytez_api_key")
    st.text_input("Bytez API URL", key="bytez_api_url")
    st.text_input("Dolibarr URL", key="dolibarr_url")
    st.text_input("Dolibarr API Key", key="dolibarr_key")
    st.caption("Keep your API keys secret! They are never stored by the app.")
# --- Extraction Logic ---
def extract_invoice_api(invoice_data):
    gemini_error = None
    bytez_error = None
    extracted = None
    with st.spinner('Extracting with Gemini...'):
        try:
            # Gemini extraction logic here (simplified example):
            gemini_response = requests.post(
                'https://generativelanguage.googleapis.com/v1beta/models/gemini-pro:generateContent',
                headers={"Authorization": f"Bearer {st.session_state.gemini_api_key}", "Content-Type": "application/json"},
                json={"contents": [{"parts": [{"text": invoice_data}]}]},
                timeout=35
            )
            if gemini_response.status_code == 200 and 'error' not in gemini_response.text:
                extracted = gemini_response.json()
            else:
                gemini_error = gemini_response.text
        except Exception as e:
            gemini_error = str(e)
    if extracted is None:
        with st.spinner('Retrying with Bytez...'):
            try:
                bytez_response = requests.post(
                    st.session_state.bytez_api_url,
                    headers={"Authorization": f"Bearer {st.session_state.bytez_api_key}", "Content-Type": "application/json"},
                    json={"invoice": invoice_data},
                    timeout=35
                )
                if bytez_response.status_code == 200 and 'error' not in bytez_response.text:
                    extracted = bytez_response.json()
                else:
                    bytez_error = bytez_response.text
            except Exception as e:
                bytez_error = str(e)
    return extracted, gemini_error, bytez_error
# --- Main App Logic ---
tab1, tab2, tab3 = st.tabs(["Upload & Extract", "Results", "History"])
with tab1:
    st.subheader("Upload and extract invoice")
    uploaded_file = st.file_uploader("Upload your invoice (PDF/Image)")
    if uploaded_file:
        file_bytes = uploaded_file.read()
        # Example file-to-text extraction (PDF/Image)
        invoice_text = "...EXTRACTED_TEXT..."  # Replace with proper OCR or PDF text extraction logic
        if st.button("Extract Invoice Data", use_container_width=True):
            extracted, gemini_error, bytez_error = extract_invoice_api(invoice_text)
            if extracted:
                st.session_state.extracted_data = extracted
                st.success("✅ Extraction successful!")
            else:
                st.error("Extraction failed.")
                if gemini_error:
                    st.error(f"Gemini error: {gemini_error}")
                if bytez_error:
                    st.error(f"Bytez error: {bytez_error}")
with tab2:
    st.subheader("Extracted Invoice Data")
    if 'extracted_data' in st.session_state:
        st.download_button(
            label="Download JSON",
            data=json.dumps(st.session_state.extracted_data, indent=2),
            file_name="invoice.json",
            mime="application/json",
            use_container_width=True,
        )
        if isinstance(st.session_state.extracted_data, dict) and st.session_state.extracted_data.get('line_items'):
            try:
                df = pd.DataFrame(st.session_state.extracted_data['line_items'])
                st.download_button(
                    label="Download CSV",
                    data=df.to_csv(index=False),
                    file_name="invoice_items.csv",
                    mime="text/csv",
                    use_container_width=True,
                )
            except Exception:
                pass
        st.divider()
        st.write("Send to Dolibarr")
        if st.button("Send to Dolibarr", use_container_width=True):
            with st.spinner("Sending..."):
                os.environ["DOLIBARR_API_URL"] = st.session_state.dolibarr_url
                os.environ["DOLIBARR_API_KEY"] = st.session_state.dolibarr_key
                resp = send_to_dolibarr(st.session_state.extracted_data)
                if 'error' in resp:
                    st.error(f"Dolibarr error: {resp['error']}")
                    if resp.get('body'):
                        with st.expander('Dolibarr response body'):
                            st.code(resp['body'])
                else:
                    st.success("Invoice sent to Dolibarr")
                    try:
                        st.json(resp)
                    except Exception:
                        pass
        st.divider()
        st.subheader("Line Items Table")
        try:
            items = st.session_state.extracted_data.get('line_items', []) if isinstance(st.session_state.extracted_data, dict) else []
            if items:
                df = pd.DataFrame(items)
                show_itable(df)
            else:
                st.info("No line items found")
        except Exception:
            pass
    else:
        st.info("Upload and extract an invoice in the first tab")
with tab3:
    st.subheader("History")
    if st.session_state.get('extraction_history'):
        for i, data in enumerate(reversed(st.session_state.extraction_history), 1):
            with st.expander(f"Extraction #{len(st.session_state.extraction_history) - i + 1}"):
                st.json(data)
    else:
        st.info("No history yet")
st.caption("Transparent fallback: Gemini → Bytez with detailed error logs if both fail.")
