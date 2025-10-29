import os
import io
import re
import json
import csv
import requests
from dotenv import load_dotenv
from PIL import Image
import streamlit as st
import fitz  # PyMuPDF
import pandas as pd
import itables
from itables.streamlit import interactive_table as show_itable

# Load environment variables
load_dotenv()

# Read defaults from environment with persistent fallbacks
DEFAULT_BYTEZ_API_KEY = os.getenv('BYTEZ_API_KEY', '')
DEFAULT_BYTEZ_API_URL = os.getenv('BYTEZ_API_URL', 'https://api.bytez.com/v1/extract')
DEFAULT_DOLIBARR_API_URL = os.getenv('DOLIBARR_API_URL', 'http://localhost/dolibarr/api/index.php')
DEFAULT_DOLIBARR_API_KEY = os.getenv('DOLIBARR_API_KEY', 'api_key_admin_2025')

# Init session state with defaults so sidebar is prefilled and persists across reruns
if 'bytez_api_key' not in st.session_state:
    st.session_state.bytez_api_key = DEFAULT_BYTEZ_API_KEY
if 'bytez_api_url' not in st.session_state:
    st.session_state.bytez_api_url = DEFAULT_BYTEZ_API_URL
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
            f"{url}/invoices" if url.endswith('/api/index.php') else f"{url}/api/index.php/invoices",
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
  "tax": "",
  "subtotal": "",
  "due_date": "",
  "payment_terms": "",
  "currency": ""
}
Validation rules:
- Ensure numbers are numbers where applicable
- Dates in ISO format YYYY-MM-DD if possible
- If a field is missing on the invoice, return an empty string or empty list
"""
}

# --- Streamlit App Layout ---
st.set_page_config(page_title="AI Invoice Extractor", layout="wide")

with st.sidebar:
    st.header("Configuration")
    st.text_input("Bytez API Key", value=st.session_state.bytez_api_key, key="bytez_api_key", type="password")
    st.text_input("Bytez API URL", value=st.session_state.bytez_api_url, key="bytez_api_url")
    st.text_input("OpenAI API Key", value=os.getenv('OPENAI_API_KEY', ''), key="openai_api_key", type="password")
    st.text_input("Anthropic API Key", value=os.getenv('ANTHROPIC_API_KEY', ''), key="anthropic_api_key", type="password")
    st.text_input("Dolibarr API URL", value=st.session_state.dolibarr_url, key="dolibarr_url")
    st.text_input("Dolibarr API Key", value=st.session_state.dolibarr_key, key="dolibarr_key", type="password")
    st.caption("These values are loaded from environment variables or defaults and persist during your session.")

# Initialize history holders
if 'extraction_history' not in st.session_state:
    st.session_state.extraction_history = []
if 'extracted_data' not in st.session_state:
    st.session_state.extracted_data = {}

st.title("AI Invoice Extractor")

# Tabs
tab1, tab2, tab3 = st.tabs(["Upload & Extract", "Review & Export", "History"])


def parse_json_from_text(t: str):
    t = (t or "").strip()
    # Try direct JSON
    try:
        return json.loads(t)
    except Exception:
        pass
    # Try to extract JSON block
    match = re.search(r"\{[\s\S]*\}$", t)
    if match:
        try:
            return json.loads(match.group(0))
        except Exception:
            pass
    raise ValueError("Model did not return valid JSON")


def extract_with_bytez(prompt: str, file_bytes: bytes, filename: str):
    """Call Bytez API for extraction. Assumes Bytez accepts multipart file + prompt."""
    if not st.session_state.bytez_api_key:
        raise RuntimeError("BYTEZ_API_KEY not set")
    headers = {"Authorization": f"Bearer {st.session_state.bytez_api_key}"}
    files = {"file": (filename, file_bytes)}
    data = {"prompt": prompt}
    url = st.session_state.bytez_api_url or DEFAULT_BYTEZ_API_URL
    r = requests.post(url, headers=headers, files=files, data=data, timeout=60)
    r.raise_for_status()
    # Bytez may already return JSON; if string, try to parse
    try:
        j = r.json()
        if isinstance(j, dict):
            return j, "Bytez"
    except Exception:
        pass
    return parse_json_from_text(r.text), "Bytez"


def extract_with_openai(prompt: str):
    if not st.session_state.openai_api_key and not os.getenv('OPENAI_API_KEY'):
        raise RuntimeError("OPENAI_API_KEY not set")
    key = st.session_state.openai_api_key or os.getenv('OPENAI_API_KEY')
    headers = {
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
    }
    # Use a low-cost/free-tier model
    payload = {
        "model": "gpt-4o-mini",
        "messages": [
            {"role": "system", "content": "Return only a valid JSON object."},
            {"role": "user", "content": prompt}
        ]
    }
    r = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload, timeout=60)
    r.raise_for_status()
    text = r.json()["choices"][0]["message"]["content"]
    return parse_json_from_text(text), "OpenAI (gpt-4o-mini)"


def extract_with_claude(prompt: str):
    if not st.session_state.anthropic_api_key and not os.getenv('ANTHROPIC_API_KEY'):
        raise RuntimeError("ANTHROPIC_API_KEY not set")
    key = st.session_state.anthropic_api_key or os.getenv('ANTHROPIC_API_KEY')
    headers = {
        "x-api-key": key,
        "anthropic-version": "2023-06-01",
        "content-type": "application/json",
    }
    # Use free/entry model
    payload = {
        "model": "claude-3-haiku-20240307",
        "max_tokens": 2000,
        "messages": [
            {"role": "user", "content": prompt + "\nReturn only the JSON object."}
        ]
    }
    r = requests.post("https://api.anthropic.com/v1/messages", headers=headers, json=payload, timeout=60)
    r.raise_for_status()
    text = r.json().get("content", [{}])[0].get("text", "")
    return parse_json_from_text(text), "Claude (haiku)"


with tab1:
    st.subheader("Upload Invoice (PDF/Image)")
    uploaded_file = st.file_uploader("Upload an invoice file", type=["pdf", "png", "jpg", "jpeg"], accept_multiple_files=False)

    # Provider selection
    provider = st.radio(
        "Select extraction provider",
        options=["Bytez", "OpenAI", "Claude"],
        index=0,
        horizontal=True,
    )

    model_name = st.selectbox("Prompt preset", list(PROMPT_TEMPLATES.keys()))

    if uploaded_file:
        file_bytes = uploaded_file.read()
        content = None
        if uploaded_file.type == 'application/pdf':
            # Use PyMuPDF to render first page as image for OCR if needed
            try:
                pdf = fitz.open(stream=file_bytes, filetype='pdf')
                page = pdf[0]
                pix = page.get_pixmap()
                img_bytes = pix.tobytes("png")
                content = Image.open(io.BytesIO(img_bytes))
            except Exception:
                content = None
        else:
            content = Image.open(io.BytesIO(file_bytes))

        if content:
            st.image(content, caption="Preview", use_container_width=True)

        # Action button changes label based on provider
        btn_label = f"Extract with {provider}"
        if st.button(btn_label, use_container_width=True):
            with st.spinner(f"Extracting with {provider}..."):
                prompt = PROMPT_TEMPLATES[model_name]
                data = None
                provider_used = None
                errors = []

                try:
                    if provider == "Bytez":
                        data, provider_used = extract_with_bytez(prompt, file_bytes, uploaded_file.name)
                    elif provider == "OpenAI":
                        data, provider_used = extract_with_openai(prompt)
                    elif provider == "Claude":
                        data, provider_used = extract_with_claude(prompt)
                except Exception as e:
                    errors.append(f"{provider} failed: {e}")

                # If selected provider failed, offer fallbacks automatically
                if data is None:
                    fallback_order = [p for p in ["Bytez", "OpenAI", "Claude"] if p != provider]
                    for fb in fallback_order:
                        try:
                            if fb == "Bytez":
                                data, provider_used = extract_with_bytez(prompt, file_bytes, uploaded_file.name)
                            elif fb == "OpenAI":
                                data, provider_used = extract_with_openai(prompt)
                            elif fb == "Claude":
                                data, provider_used = extract_with_claude(prompt)
                            if data is not None:
                                break
                        except Exception as e:
                            errors.append(f"{fb} fallback failed: {e}")

                if data is not None:
                    st.session_state.extracted_data = data
                    st.session_state.extraction_history.append(data)
                    st.success(f"Extraction complete via {provider_used}.")
                    st.json(data)
                else:
                    st.error("Extraction failed across all providers.")
                    with st.expander("View error details"):
                        for err in errors:
                            st.write(f"- {err}")

with tab2:
    st.subheader("Review Extracted Data")
    if st.session_state.extracted_data:
        st.json(st.session_state.extracted_data)
        # Download as JSON
        st.download_button(
            label="💾 Download JSON",
            data=json.dumps(st.session_state.extracted_data, indent=2),
            file_name="invoice.json",
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
            if not st.session_state.dolibarr_url or not st.session_state.dolibarr_key:
                st.error("❌ Please configure Dolibarr API credentials in the sidebar")
            else:
                with st.spinner("📤 Sending to Dolibarr..."):
                    # Set environment variables so send_to_dolibarr uses them
                    os.environ["DOLIBARR_API_URL"] = st.session_state.dolibarr_url
                    os.environ["DOLIBARR_API_KEY"] = st.session_state.dolibarr_key
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
st.caption("🔐 Your API keys are never stored. Environment variables or defaults are used for convenience.")
