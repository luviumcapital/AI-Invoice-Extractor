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
      "amount": ""
    }
  ],
  "subtotal": "",
  "tax_amount": "",
  "discount": "",
  "payment_terms": "",
  "due_date": ""
}

NOTE: Always return valid JSON only. If you cannot determine a field, use an empty string. Ensure all JSON syntax is correct.
""",
}

# --- Load Environment Variables ---
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
CLAUDE_API_KEY = os.getenv("CLAUDE_API_KEY")
BYTEZ_API_KEY = os.getenv("BYTEZ_API_KEY")

# Initialize Gemini
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)

# --- Helper Functions ---
def image_to_base64(image_path: str) -> str:
    """Convert image to base64 string."""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

def pdf_to_images(pdf_path: str) -> list:
    """Convert PDF to list of PIL Images."""
    images = []
    pdf_document = fitz.open(pdf_path)
    for page_num in range(len(pdf_document)):
        page = pdf_document[page_num]
        pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        images.append(img)
    return images

def extract_with_gemini(image_data, prompt: str) -> dict:
    """
    Extract invoice data using Google Gemini API.
    
    Args:
        image_data: PIL Image object or file path
        prompt: Extraction prompt template
    
    Returns:
        Extracted data as dictionary
    """
    try:
        model = genai.GenerativeModel("gemini-1.5-flash")
        
        if isinstance(image_data, str):
            # File path provided
            response = model.generate_content([
                prompt,
                Image.open(image_data)
            ])
        else:
            # PIL Image provided
            response = model.generate_content([
                prompt,
                image_data
            ])
        
        # Parse response
        response_text = response.text.strip()
        # Remove markdown code blocks if present
        if response_text.startswith("```"):
            response_text = response_text.split("```")[1]
            if response_text.startswith("json"):
                response_text = response_text[4:]
        response_text = response_text.strip()
        
        extracted_data = json.loads(response_text)
        return extracted_data
    except Exception as e:
        st.error(f"Gemini extraction failed: {str(e)}")
        return None

def extract_with_bytez(image_data, prompt: str = None) -> dict:
    """
    Extract invoice data using Bytez Document QA API as fallback provider.
    
    This function serves as a fallback extraction provider after Gemini and Claude.
    It uses the Bytez Visual Question Answering model to extract structured invoice data.
    
    Args:
        image_data: PIL Image object or file path
        prompt: Optional prompt (for consistency with other extraction functions)
    
    Returns:
        Extracted data as dictionary
    """
    if not BYTEZ_API_KEY:
        return None
    
    try:
        # Convert image to base64 if needed
        if isinstance(image_data, str):
            # File path provided
            with open(image_data, "rb") as img_file:
                import base64
                image_base64 = base64.b64encode(img_file.read()).decode("utf-8")
        else:
            # PIL Image provided
            import base64
            img_bytes = io.BytesIO()
            image_data.save(img_bytes, format="PNG")
            image_base64 = base64.b64encode(img_bytes.getvalue()).decode("utf-8")
        
        # Define questions for each required invoice field
        questions = {
            "invoice_number": "What is the invoice number?",
            "invoice_date": "What is the invoice date?",
            "total_amount": "What is the total amount?",
            "subtotal": "What is the subtotal?",
            "tax_amount": "What is the tax amount?",
            "discount": "What is the discount?",
            "payment_terms": "What are the payment terms?",
            "due_date": "What is the due date?",
            "vendor_name": "What is the vendor name?",
            "vendor_address": "What is the vendor address?",
            "vendor_contact": "What is the vendor contact information?",
            "customer_name": "What is the customer name?",
            "customer_address": "What is the customer address?",
            "customer_contact": "What is the customer contact information?",
        }
        
        # Bytez API endpoint
        api_url = "https://api.bytez.com/models/v2/cloudqi/CQI_Visual_Question_Awnser_PT_v0"
        headers = {
            "Authorization": f"Bearer {BYTEZ_API_KEY}",
            "Content-Type": "application/json"
        }
        
        # Extract answers for all questions
        extracted_data = {
            "vendor": {},
            "customer": {},
            "line_items": []
        }
        
        for field_key, question in questions.items():
            payload = {
                "image": image_base64,
                "question": question
            }
            
            response = requests.post(api_url, json=payload, headers=headers, timeout=30)
            
            if response.status_code == 200:
                response_data = response.json()
                answer = response_data.get("answer", "")
                
                # Map answer to appropriate field
                if field_key == "invoice_number":
                    extracted_data["invoice_number"] = answer
                elif field_key == "invoice_date":
                    extracted_data["invoice_date"] = answer
                elif field_key == "total_amount":
                    extracted_data["total_amount"] = answer
                elif field_key == "subtotal":
                    extracted_data["subtotal"] = answer
                elif field_key == "tax_amount":
                    extracted_data["tax_amount"] = answer
                elif field_key == "discount":
                    extracted_data["discount"] = answer
                elif field_key == "payment_terms":
                    extracted_data["payment_terms"] = answer
                elif field_key == "due_date":
                    extracted_data["due_date"] = answer
                elif field_key == "vendor_name":
                    extracted_data["vendor"]["name"] = answer
                elif field_key == "vendor_address":
                    extracted_data["vendor"]["address"] = answer
                elif field_key == "vendor_contact":
                    extracted_data["vendor"]["contact"] = answer
                elif field_key == "customer_name":
                    extracted_data["customer"]["name"] = answer
                elif field_key == "customer_address":
                    extracted_data["customer"]["address"] = answer
                elif field_key == "customer_contact":
                    extracted_data["customer"]["contact"] = answer
        
        return extracted_data
    except Exception as e:
        st.error(f"Bytez extraction failed: {str(e)}")
        return None

def extract_invoice_data(image_data, invoice_type: str = "General Invoice"):
    """
    Extract invoice data using multiple providers as fallbacks.
    
    Extraction order:
    1. Gemini API (primary)
    2. Bytez Document QA API (fallback)
    
    Args:
        image_data: PIL Image object or file path
        invoice_type: Type of invoice for template selection
    
    Returns:
        Extracted data as dictionary
    """
    prompt = PROMPT_TEMPLATES.get(invoice_type, PROMPT_TEMPLATES["General Invoice"])
    
    # Try Gemini first (primary provider)
    st.info("Attempting extraction with Gemini...")
    result = extract_with_gemini(image_data, prompt)
    if result:
        st.success("Successfully extracted with Gemini!")
        return result
    
    # Fallback to Bytez Document QA API
    st.warning("Gemini extraction failed, attempting Bytez Document QA...")
    result = extract_with_bytez(image_data, prompt)
    if result:
        st.success("Successfully extracted with Bytez Document QA!")
        return result
    
    # If all extraction methods fail
    st.error("All extraction methods failed. Please try again or upload a different document.")
    return None

# --- Streamlit App ---
def main():
    st.set_page_config(page_title="AI Invoice Extractor", layout="wide")
    st.title("🧾 AI Invoice Extractor")
    st.write("Upload an invoice (PDF or image) to extract structured data.")
    
    # File upload
    uploaded_file = st.file_uploader(
        "Choose an invoice file",
        type=["pdf", "png", "jpg", "jpeg", "bmp", "tiff"]
    )
    
    if uploaded_file:
        # Process file
        file_extension = uploaded_file.name.split(".")[-1].lower()
        
        # Save uploaded file temporarily
        temp_file_path = f"/tmp/{uploaded_file.name}"
        with open(temp_file_path, "wb") as temp_file:
            temp_file.write(uploaded_file.getbuffer())
        
        # Extract based on file type
        if file_extension == "pdf":
            images = pdf_to_images(temp_file_path)
            if images:
                # Use first page for extraction
                extracted_data = extract_invoice_data(images[0])
        else:
            # Image file
            extracted_data = extract_invoice_data(temp_file_path)
        
        # Display results
        if extracted_data:
            st.success("✅ Extraction completed successfully!")
            st.subheader("Extracted Invoice Data")
            st.json(extracted_data)
            
            # Option to export
            csv_data = pd.json_normalize(extracted_data)
            csv_buffer = io.StringIO()
            csv_data.to_csv(csv_buffer, index=False)
            st.download_button(
                label="Download as CSV",
                data=csv_buffer.getvalue(),
                file_name=f"invoice_{extracted_data.get('invoice_number', 'unknown')}.csv",
                mime="text/csv"
            )

if __name__ == "__main__":
    main()
