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
load_dotenv()
# Read defaults from environment with persistent fallbacks
DEFAULT_GEMINI_API_KEY = os.getenv('GEMINI_API_KEY', 'AIzaSyCXUx4IJan48xLgAoEoturQWuQY8wSnELE')
DEFAULT_BYTEZ_API_URL = os.getenv('BYTEZ_API_URL', 'https://api.bytez.com/v1/extract')
DEFAULT_BYTEZ_API_KEY = os.getenv('BYTEZ_API_KEY', '')
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
        headers = {
            "DOLAPIKEY": api_key,
            "Content-Type": "application/json"
        }
        dolibarr_payload = {
            "ref": invoice_json.get("invoice_number", ""),