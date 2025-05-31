# Smart Doc Sorter

**Smart Doc Sorter** is a multi-agent AI system designed to intelligently process and route various types of documents. It accepts inputs in PDF, JSON, or Email (text) format, automatically classifies the document's format and intent (e.g., Invoice, RFQ, Complaint), and then routes it to the appropriate specialized agent for further processing and data extraction. The system maintains a shared context using Redis, enabling traceability and potential for chained operations.

## System Overview

The system is orchestrated by a central **Classifier Agent** which coordinates the workflow:

1.  **Input Reception:** The system takes a file path or raw text string as input.
2.  **Classifier Agent:**
    *   Determines the **format** of the input (PDF, JSON, Email, or general Text).
    *   Utilizes a Large Language Model (Google Gemini Pro) to classify the **intent** of the document (e.g., Invoice, RFQ, Complaint, Regulation, Other).
    *   Logs the initial classification (format, intent, conversation ID) to shared memory (Redis).
    *   Routes the input to the appropriate specialized agent based on the determined format.
3.  **Specialized Agents:**
    *   **JSON Agent:** Accepts structured JSON payloads, extracts data according to a target schema, and flags anomalies or missing fields.
    *   **Email Agent:** Accepts email content, extracts sender, subject, body, and determines urgency.
    *   *(PDFs and general text are currently processed by extracting text and logging it with their classified intent. A dedicated PDF agent could be added for more specific PDF data extraction.)*
4.  **Shared Memory (Redis):**
    *   Stores a history for each processed document under a unique `conversation_id`.
    *   Logs include source format, classified intent, extracted values from specialized agents, anomalies (if any), and timestamps. This provides traceability and a context for each interaction.

**Example Flow:**
`User provides an email file -> ClassifierAgent detects "Email" format + "RFQ" intent -> Routes to EmailAgent -> EmailAgent extracts sender, subject, etc. -> ClassifierAgent logs all details to Redis.`

## Features

*   **Multi-Format Input:** Handles PDF, JSON, and Email (text) inputs.
*   **Automated Format Detection:** Intelligently determines the input's format.
*   **LLM-Powered Intent Classification:** Uses Google Gemini Pro for robust intent understanding.
*   **Modular Agent-Based Architecture:** Clear separation of concerns with a central classifier and specialized processing agents.
*   **Shared Context & Traceability:** Leverages Redis for logging conversation history and extracted data.
*   **Extensible:** Designed to easily incorporate new specialized agents or enhance existing ones.

## Tech Stack

*   **Python 3.9+**
*   **Google Gemini Pro:** For Natural Language Understanding and Intent Classification.
    *   Accessed via the `google-generativeai` SDK.
*   **Redis:** For lightweight shared memory and logging.
    *   Accessed via the `redis` Python library.
*   **PDFPlumber (`pdfplumber`):** For text extraction from PDF documents.
*   **Python Standard Libraries:** `json`, `os`, `argparse`, `uuid`.
*   **`python-dotenv`:** For managing environment variables (API keys).
*   **`reportlab`:** (Optional, used in `main.py` for generating sample PDF for testing).
