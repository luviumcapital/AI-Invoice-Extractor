# AI Invoice Extractor
=====================

## Overview

AI-Invoice-Extractor is a professional web application built with Streamlit that enables users to upload invoices in various formats (PDF, JPEG, PNG) and extract structured data using advanced OCR and Google Gemini AI. The application provides a modern, interactive interface for extracting, analyzing, and downloading invoice data.

## Features

- **Text Extraction:** Extracts text from images (using Tesseract OCR) and PDF documents.
- **AI-Powered Invoice Analysis:** Utilizes Google Gemini AI to extract structured invoice fields and line items.
- **Interactive Web Interface:** Clean, user-friendly UI built with Streamlit, including dynamic tables and download options.
- **Download Options:** Export extracted data as CSV or JSON.
- **Robust Error Handling:** User-friendly error messages and input validation throughout the app.
- **Deployment Ready:** Easily deployable on Streamlit Community Cloud or other platforms.

## Installation

### Prerequisites

- Python 3.7 or higher
- Tesseract OCR engine (for image extraction)

### Setup Instructions

1. **Clone the Repository**

   ```bash
   git clone https://github.com/pratstick/AI-Invoice-Extractor.git
   cd AI-Invoice-Extractor
   ```

2. **Create and Activate a Virtual Environment**

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Dependencies**

   ```bash
   pip install -r requirements.txt
   ```

4. **Install Tesseract OCR Engine**

   - **Debian/Ubuntu:**
     ```bash
     sudo apt-get install tesseract-ocr
     ```
   - **macOS (Homebrew):**
     ```bash
     brew install tesseract
     ```
   - **Windows:**
     Download and install from [Tesseract Releases](https://github.com/tesseract-ocr/tesseract/releases). Ensure the Tesseract executable is in your system PATH.

5. **Set Up Environment Variables**

   Create a `.env` file in the project root and add your Google API key:
   ```
   GOOGLE_API_KEY=your_google_api_key_here
   ```

## Running the Application

1. **Start the Streamlit App**

   ```bash
   streamlit run app.py
   ```

2. **Access the Application**

   Open your browser and navigate to [http://localhost:8501](http://localhost:8501).

## Deployment

### Streamlit Community Cloud

1. **Create a New App**
   - Log in to Streamlit Community Cloud and click "New app".
   - Link your GitHub repository.
2. **Configure Deployment**
   - Set the entry point to `app.py`.
   - Add your Google API key as a secret or in the `.env` file.
3. **Deploy**
   - Click "Deploy". Streamlit Cloud will install dependencies and launch your app.

### Other Platforms

For platforms such as Heroku or AWS, follow their respective deployment guides for Python web applications. Ensure environment variables and dependencies are configured appropriately.

## Usage

1. **Upload Invoice:** Select a PDF or image (JPEG, PNG) of your invoice.
2. **Select Prompt:** Choose or customize the extraction prompt as needed.
3. **Analyze:** Click the "Analyze Invoice" button to extract and view structured data.
4. **Download:** Export results as CSV or JSON for further processing.

## Testing

- Unit tests are located in the `tests/` directory. To run tests:
  ```bash
  pytest tests/
  ```

## Contributing

Contributions are welcome! Please fork the repository and submit a pull request. Ensure your code is well-documented and tested.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.