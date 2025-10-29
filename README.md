# AI Invoice Extractor
=====================

## Overview

AI-Invoice-Extractor is a professional web application built with Streamlit that enables users to upload invoices in various formats (PDF, JPEG, PNG) and extract structured data using advanced OCR and AI models. The application automatically uses the best-available free tier API provider (Bytez, OpenAI, or Claude) to analyze invoices with high accuracy.

## Features

- **Text Extraction:** Extracts text from images (using Tesseract OCR) and PDF documents
- **Multi-Provider AI Analysis:** Supports Bytez (default), OpenAI, and Claude (Anthropic) with automatic fallback
- **Free Tier Support:** All providers configured to work with free tier APIs
- **Interactive Web Interface:** Clean, user-friendly UI built with Streamlit
- **Download Options:** Export extracted data as CSV or JSON
- **Robust Error Handling:** User-friendly error messages and input validation
- **Deployment Ready:** Easy deployment on Railway, Heroku, or other platforms

## Installation

### Prerequisites

- Python 3.8 or higher
- Tesseract OCR engine (for image extraction)
- API keys for at least one of: Bytez, OpenAI, or Claude

### Setup Instructions

#### 1. Clone the Repository

```bash
git clone https://github.com/luviumcapital/AI-Invoice-Extractor.git
cd AI-Invoice-Extractor
```

#### 2. Create and Activate a Virtual Environment

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

#### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

#### 4. Install Tesseract OCR Engine

**Debian/Ubuntu:**
```bash
sudo apt-get install tesseract-ocr
```

**macOS (Homebrew):**
```bash
brew install tesseract
```

**Windows:**
Download and install from [Tesseract Releases](https://github.com/tesseract-ocr/tesseract/releases). Ensure the Tesseract executable is in your system PATH.

#### 5. Set Up Environment Variables

Create a `.env` file in the project root and add your API keys for the providers you want to use:

### Free Tier API Setup

#### Option 1: Bytez (Recommended - Default)

1. Visit [https://bytez.com/](https://bytez.com/)
2. Sign up for a free account
3. Go to your dashboard and copy your API key
4. Add to `.env`:
   ```
   BYTEZ_API_KEY=your_bytez_api_key_here
   ```

**Free Tier Benefits:**
- Unlimited tokens, images, and videos
- Access to 141k+ small models
- 1 concurrent request
- No credit card required

#### Option 2: OpenAI (Free Trial)

1. Visit [https://platform.openai.com/](https://platform.openai.com/)
2. Sign up and create an API key
3. Note: OpenAI provides $5 free credits for the first 3 months
4. Add to `.env`:
   ```
   OPENAI_API_KEY=your_openai_api_key_here
   ```

#### Option 3: Claude (Anthropic - Free Trial)

1. Visit [https://console.anthropic.com/](https://console.anthropic.com/)
2. Sign up and create an API key
3. Note: Anthropic provides free trial credits for new users
4. Add to `.env`:
   ```
   ANTHROPIC_API_KEY=your_anthropic_api_key_here
   ```

### Complete `.env` File Example

```
# Primary: Bytez (Recommended)
BYTEZ_API_KEY=your_bytez_api_key_here

# Fallback Options
OPENAI_API_KEY=your_openai_api_key_here
ANTHROPIC_API_KEY=your_anthropic_api_key_here
```

## Running the Application

```bash
streamlit run app.py
```

The application will open in your browser at `http://localhost:8501`

## API Provider Priority

The application automatically selects the best available provider:

1. **Bytez** (Primary) - Recommended for production use
2. **OpenAI** (Fallback 1) - If Bytez is unavailable
3. **Claude** (Fallback 2) - If both above are unavailable

The system checks for valid API keys and automatically switches providers if issues occur.

## Deployment

### Railway Deployment

1. Push your code to GitHub
2. Connect your repository to Railway
3. Set the following environment variables in Railway:
   - `BYTEZ_API_KEY`
   - `OPENAI_API_KEY` (optional)
   - `ANTHROPIC_API_KEY` (optional)
4. Deploy with your custom domain

### Environment Variables

All API keys should be set as environment variables:
- `BYTEZ_API_KEY` - Your Bytez API key
- `OPENAI_API_KEY` - Your OpenAI API key
- `ANTHROPIC_API_KEY` - Your Anthropic API key

## Project Structure

```
.
├── app.py                 # Main Streamlit application
├── requirements.txt       # Python dependencies
├── packages.txt          # System packages (for deployment)
├── README.md             # This file
└── .env                  # Environment variables (not in git)
```

## Troubleshooting

### "No API key found" Error
- Ensure at least one API key is configured in your `.env` file
- Restart the application after adding environment variables

### OCR Not Working
- Verify Tesseract OCR is installed: `tesseract --version`
- On Linux: `sudo apt-get install tesseract-ocr`
- On macOS: `brew install tesseract`

### Invoice Extraction Issues
- Ensure uploaded PDF/images are clear and readable
- Try a different AI provider if one fails
- Check that API keys are valid and have remaining quota

## Contributing

Contributions are welcome! Please feel free to submit issues and pull requests.

## License

This project is licensed under the MIT License. See LICENSE file for details.

## Support

For issues and support:
- Check the GitHub Issues page
- Review API provider documentation:
  - [Bytez Docs](https://docs.bytez.com/)
  - [OpenAI Docs](https://platform.openai.com/docs/)
  - [Anthropic Docs](https://docs.anthropic.com/)
