# Document Intelligence Tool

Extract action items from business documents, classify the document type, log tasks to Google Sheets, and email a clean summary automatically.

## Tech Stack

- Python
- PyMuPDF
- pytesseract, pdf2image, and Pillow
- python-docx
- Groq API
- ChromaDB
- sentence-transformers
- Google Sheets API with gspread and oauth2client
- Gmail SMTP
- Gradio
- Flask
- n8n
- python-dotenv

## Setup

1. Create and activate a virtual environment:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

2. Install requirements:

```powershell
pip install -r requirements.txt
```

3. Create a `.env` file from the example:

```powershell
Copy-Item .env.example .env
```

4. Fill in the `.env` values:

```env
GROQ_API_KEY=your_groq_api_key_here
GMAIL_SENDER=your_email@gmail.com
GMAIL_APP_PASSWORD=your_gmail_app_password_here
GOOGLE_SHEET_ID=your_google_sheet_id_here
```

5. Add Google service account credentials:

- Create or select a Google Cloud service account with Sheets access.
- Download its JSON key.
- Save it in the project root as `credentials.json`.
- Share the target Google Sheet with the service account email address.
- Make sure the sheet has these headers: `Task | Owner | Deadline | Priority | Logged At`.

6. RAG setup:

- The app uses `sentence-transformers/all-MiniLM-L6-v2` for local embeddings.
- The first RAG run downloads and caches the embedding model from Hugging Face.
- Document chunks are stored locally in `.rag_store/`, which is ignored by Git.

## How to Run

Start the app:

```powershell
python main.py
```

This starts:

- Gradio UI on the first free port from `http://127.0.0.1:7860` to `http://127.0.0.1:7899`
- Flask API at `http://127.0.0.1:5000`

Health check:

```powershell
Invoke-RestMethod http://localhost:5000/health
```

Webhook test:

```powershell
python test_webhook.py
```

## Sample Use Case

Upload meeting notes PDF -> classify the document -> extract action items -> log tasks to Google Sheet -> email the summary automatically.

## Project Architecture

```text
PDF / TXT / DOCX File
   |
[n8n Webhook] -> triggers ->
   |
[Flask API /webhook]
   |
[PyMuPDF / OCR / TXT / DOCX Extraction] -> extracts raw text
   |
[RAG Layer] -> chunks text, creates embeddings, stores in ChromaDB
   |
[Groq API] -> document classification
   |
[RAG Layer] -> retrieves relevant chunks for the document type
   |
[Groq API] -> specialized extraction using retrieved context
   |
[Google Sheets] <- logs rows
   |
[Gmail SMTP] -> sends summary email
   |
[Gradio UI] -> shows classification, retrieved evidence, summary, structured data, and tasks
```

## API Usage

Send a POST request to the Flask webhook:

```json
{
  "pdf_path": "path/to/your/document.pdf",
  "recipient_email": "yourteam@company.com"
}
```

Successful response:

```json
{
  "status": "success",
  "classification": {
    "document_type": "meeting_notes",
    "confidence": 0.91,
    "summary": "Weekly planning meeting notes with follow-up tasks."
  },
  "action_items": []
}
```

## n8n Integration

### Setup

1. Install n8n: `npx n8n`
2. Open n8n at `http://localhost:5678`
3. Go to Settings -> Import Workflow -> upload `n8n_workflow.json`
4. Activate the workflow
5. Make sure `main.py` is running on port 5000

### How it works

- n8n listens for a POST request at its webhook URL
- On trigger, it calls the Document Intelligence Tool API at `localhost:5000/webhook`
- The tool extracts raw text, classifies the document, extracts action items, logs to Google Sheets, and sends the summary email
- The RAG layer indexes the document and retrieves the most relevant chunks before extraction and chat
- n8n returns a success/error response

### Trigger the workflow

Send a POST request to your n8n webhook URL:

```json
{
  "pdf_path": "path/to/your/document.pdf",
  "recipient_email": "yourteam@company.com"
}
```

### Why n8n?

n8n acts as the orchestration layer. Instead of running the script manually,
any external system such as Slack, email, file upload, or a scheduler can
trigger the pipeline automatically via webhook.

## Project Files

- `main.py` - Text extraction, document classification, RAG-aware extraction, Google Sheets logging, email sending, Flask API, and Gradio UI
- `rag.py` - Chunking, embeddings, ChromaDB indexing, retrieval, and evidence formatting
- `test_webhook.py` - Simple local webhook test client
- `n8n_workflow.json` - Importable n8n workflow
- `.env.example` - Environment variable template
- `credentials.json` - Local Google service account file, not committed
