import os
import smtplib
import imaplib
import email
import ssl
import json
import re
import requests
import time
from email.message import EmailMessage

# --- Configuration & Secrets (Loaded from GitHub Environment Variables) ---
from google import genai
from google.genai import types

# Load the API key from the environment variable (set via GitHub Secrets)
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY") 
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash") 

EMAIL_ADDRESS = os.environ.get("EMAIL_ADDRESS")
EMAIL_PASSWORD = os.environ.get("EMAIL_PASSWORD") 
SMTP_SERVER = "smtp.gmail.com"
SMTP_PORT = 465
IMAP_SERVER = "imap.gmail.com"

# --- LangSmith Configuration for Tracing ---
langsmith_key = os.environ.get("LANGCHAIN_API_KEY")

if langsmith_key:
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_API_KEY"] = langsmith_key
    os.environ["LANGCHAIN_PROJECT"] = "Email_automation_schedule"
    print("STATUS: LangSmith tracing configured.")
else:
    os.environ["LANGCHAIN_TRACING_V2"] = "false"
    print("STATUS: LANGCHAIN_API_KEY not found. LangSmith tracing is disabled.")

# Initialize Gemini Client
gemini_client = None
if GEMINI_API_KEY:
    try:
        # Pass the key explicitly to the Client (though the SDK often auto-loads if the name is correct)
        gemini_client = genai.Client(api_key=GEMINI_API_KEY)
        print("STATUS: Gemini client initialized successfully using API key from environment.")
    except Exception as e:
        print(f"CRITICAL ERROR: Failed to initialize Gemini client. Error: {e}")
else:
    print("CRITICAL ERROR: GEMINI_API_KEY environment variable not found. Gemini client not initialized.")


# --- Knowledge Base & Persona Configuration ---
DATA_SCIENCE_KNOWLEDGE = """
# Data Science Project & Service Knowledge Base (Updated for Time Series and Project Inquiries)
#
# --------------------------------------------------------------------------------
## 1. Core Services Offered:
- **Predictive Modeling:** Advanced Regression, Time Series Forecasting (including **ARIMA**, SARIMA, Prophet, and **LSTM** for complex sequences). We handle complex **problem statements** across finance, logistics, and retail.
- **Machine Learning (ML) Projects:** Full lifecycle development for all ML tasks (classification, clustering, reinforcement learning).
- **Natural Language Processing (NLP):** Sentiment Analysis, Topic Modeling, Text Summarization, and custom Named Entity Recognition (NER).
- **Computer Vision:** Object Detection, Image Segmentation, and OCR solutions using CNNs (YOLO, ResNet).
- **MLOps and Deployment:** Model containerization (Docker), CI/CD pipelines, and hosting on AWS SageMaker, Azure ML, or GCP Vertex AI.
- **Data Engineering:** ETL pipeline development using Python/Pandas, Spark, and SQL optimization for large datasets.
- **Data Visualization & Reporting:** Interactive dashboards built with Streamlit, Tableau, and Power BI for executive summaries.

## 2. Guidance for Time Series Forecasting (Specific Reply Content):
- **ARIMA (and variations like SARIMA):** Excellent baseline for linear relationships and stationary data. Great for **short-term, stable predictions** and when interpretability is key. Requires data stationarity.
- **LSTM (Long Short-Term Memory Networks):** A type of recurrent neural network (RNN) superior for capturing complex, non-linear relationships, long-term dependencies, and memory in the data. Ideal for **long-term predictions** or highly volatile, non-stationary data (e.g., stock prices, sensor data). Requires more data and computation.
- **Recommendation:** If the data is complex, non-linear, or the prediction horizon is long, **prioritize LSTM**. For simple, short-term, or highly interpretable needs, **ARIMA is better**.

## 3. Standard Client Engagement Process:
1. **Initial Discovery Call (45 minutes):** Define the business problem, available data sources, and establish success metrics.
2. **Data Audit and Preparation (Phase 1):** Comprehensive review of data quality, feature engineering, and cleaning.
3. **Model Prototyping and Validation (Phase 2):** Iterative development, hyperparameter tuning, and cross-validation.
4. **Deployment and Handoff (Phase 3):** Integration of the final model into the client's infrastructure and comprehensive documentation/training.
5. **Post-Deployment Monitoring:** Quarterly performance reviews and model drift detection.

## 4. Availability for Meetings:
Available for 45-minute discovery calls on **Mondays, Wednesdays, and Fridays** between 2:00 PM and 5:00 PM **IST** (Indian Standard Time). Please propose two time slots within this window.
"""
# --------------------------------------------------------------------------------

# Agent 1 Condition: Determines if the email is technical enough for a specialized reply.
AUTOMATION_CONDITION = (
    "Does the incoming email contain a technical question or an explicit project inquiry/pitch related to Data Science, "
    "Machine Learning (ML), Deep Learning, Data Engineering, advanced Statistical Analysis, or any service listed in the core offerings? "
)

# Agent 2 & 4 Persona: Defines reply style and meeting scheduling logic.
AGENTIC_SYSTEM_INSTRUCTIONS = (
    "You are a professional, Agentic AI system acting ONLY as Senior Data Scientist, Akash BV. You MUST NOT impersonate anyone else or reply to third-party content (like certificates or team emails).\n"
    "Your primary goal is to provide a helpful, professional, and courteous response to every email. Your task is to perform all required roles and provide a structured JSON output.\n"
    "1. CONDITION CHECK: Determine if the email is technical or a project pitch (based on the AUTOMATION_CONDITION).\n"
    "2. TRANSLATOR: Generate a reply. If technical, use the Knowledge Base for details. If non-technical, generate a polite general acknowledgement.\n"
    "3. TONE ANALYZER: If the email contains a serious project inquiry, set 'request_meeting' to true.\n\n"
    
    "CRITICAL FORMATTING GUIDANCE:\n"
    " - All generated drafts (simple_reply_draft, non_technical_reply_draft, meeting_suggestion_draft) MUST be in **PLAIN TEXT** format. **DO NOT USE HTML TAGS (like <br> or <b>)**.\n"
    " - All replies MUST be signed off with the exact signature: 'Best regards,\\nAkash BV'."
)

# JSON Schema definition to enforce structured output for Gemini
RESPONSE_SCHEMA_JSON = {
    "type": "object",
    "properties": {
        "is_technical": {"type": "boolean", "description": "True if the email matches the technical/project condition, False otherwise."},
        "simple_reply_draft": {"type": "string", "description": "The primary reply to the client, simplified and non-technical, based on the knowledge base (USED IF is_technical is TRUE)."},
        "non_technical_reply_draft": {"type": "string", "description": "A polite, professional acknowledgement and offer to help, used if is_technical is FALSE."},
        "request_meeting": {"type": "boolean", "description": "True if the tone suggests a serious project inquiry or pitch, False otherwise. (Triggers meeting suggestion)."},
        "meeting_suggestion_draft": {"type": "string", "description": "If request_meeting is true, draft a reply suggesting available dates from the knowledge base (e.g., 'Are you available this week on Monday, Wednesday, or Friday afternoon?')."}
    },
    "required": ["is_technical", "simple_reply_draft", "non_technical_reply_draft", "request_meeting", "meeting_suggestion_draft"]
}

# Convert Python dict to the required types.Schema for the Gemini API
response_schema = types.Schema.from_dict(RESPONSE_SCHEMA_JSON)

# --- Helper Functions (SMTP and IMAP remain unchanged) ---

def _send_smtp_email(to_email, subject, content):
    """Utility to send an email via SMTP_SSL."""
    if not EMAIL_ADDRESS or not EMAIL_PASSWORD:
        print("ERROR: Email credentials not available.")
        return False
    
    try:
        msg = EmailMessage()
        msg["Subject"] = subject
        msg["From"] = EMAIL_ADDRESS
        msg["To"] = to_email
        msg.set_content(content)
        
        context = ssl.create_default_context()
        with smtplib.SMTP_SSL(SMTP_SERVER, SMTP_PORT, context=context) as server:
            print(f"DEBUG: Attempting to log into SMTP server {SMTP_SERVER}...")
            server.login(EMAIL_ADDRESS, EMAIL_PASSWORD)
            server.send_message(msg)
            print("DEBUG: Successfully logged in and sent message.")
        return True
    except smtplib.SMTPAuthenticationError:
        print("CRITICAL SMTP ERROR: Authentication failed. Is your EMAIL_PASSWORD a 16-character App Password?")
        return False
    except Exception as e:
        print(f"ERROR: Failed to send email to {to_email}: {e}")
        return False

def _fetch_latest_unread_email():
    """Fetches the latest unread email details and marks it as read."""
    if not EMAIL_ADDRESS or not EMAIL_PASSWORD:
        print("CRITICAL ERROR: EMAIL_ADDRESS or EMAIL_PASSWORD not set in environment.")
        return None, None, None

    try:
        print(f"DEBUG: Attempting to log into IMAP server {IMAP_SERVER}...")
        mail = imaplib.IMAP4_SSL(IMAP_SERVER)
        mail.login(EMAIL_ADDRESS, EMAIL_PASSWORD)
        print("DEBUG: IMAP login successful.")
        
        mail.select("inbox")
        
        status, data = mail.search(None, 'UNSEEN')
        ids = data[0].split()

        if not ids:
            print("STATUS: IMAP search found no unread emails.")
            return None, None, None

        latest_id = ids[-1]
        mail.store(latest_id, '+FLAGS', '\\Seen')
        
        status, msg_data = mail.fetch(latest_id, "(RFC822)")
        raw_email = msg_data[0][1]
        email_message = email.message_from_bytes(raw_email)

        from_header = email_message.get("From", "")
        subject = email_message.get("Subject", "No Subject")
        
        from_match = re.search(r"<([^>]+)>", from_header)
        from_email = from_match.group(1) if from_match else from_header
        
        body = ""
        if email_message.is_multipart():
            for part in email_message.walk():
                ctype = part.get_content_type()
                cdispo = str(part.get("Content-Disposition"))
                if ctype == "text/plain" and "attachment" not in cdispo:
                    body = part.get_payload(decode=True).decode()
                    break
        else:
            body = email_message.get_payload(decode=True).decode()
        
        print(f"DEBUG: Successfully processed email from {from_email} with subject: {subject[:30]}...")
        return from_email, subject, body

    except imaplib.IMAP4.error as e:
        print(f"CRITICAL IMAP ERROR: Failed to fetch email. Check your App Password and IMAP settings. Error: {e}")
        return None, None, None
    except Exception as e:
        print(f"CRITICAL IMAP ERROR: An unexpected error occurred during email fetching: {e}")
        return None, None, None


def _run_ai_agent(email_data):
    """
    Calls the Gemini API using the official SDK, enforcing structured JSON output.
    """
    global gemini_client

    if not gemini_client:
        print("CRITICAL ERROR: Gemini client not initialized. Cannot run agent.")
        return None
    
    # Construct the consolidated prompt for the model
    user_prompt = (
        f"**KNOWLEDGE BASE (For context and reply)**:\n{DATA_SCIENCE_KNOWLEDGE}\n\n"
        f"**TASK CONFIGURATION**:\n"
        f"CONDITION TO CHECK: {AUTOMATION_CONDITION}\n"
        f"Generate the JSON object based on the incoming email content below. "
        f"Ensure all reply drafts are in PLAIN TEXT and end with the exact signature: 'Best regards,\\nAkash BV'.\n\n"
        f"--- INCOMING EMAIL CONTENT ---\n"
        f"FROM: {email_data['from_email']}\n"
        f"SUBJECT: {email_data['subject']}\n"
        f"BODY:\n{email_data['body']}\n\n"
    )

    # Configuration for the Gemini API call
    config = types.GenerateContentConfig(
        system_instruction=AGENTIC_SYSTEM_INSTRUCTIONS,
        response_mime_type="application/json",
        response_schema=response_schema,
        temperature=0.3
    )
    
    # Retry with exponential backoff
    for i in range(3):
        try:
            print(f"DEBUG: Attempting Gemini API call (Retry {i+1})...")
            response = gemini_client.models.generate_content(
                model=GEMINI_MODEL,
                contents=user_prompt,
                config=config,
            )
            
            # The response text will be a JSON string due to the config
            raw_content = response.text.strip()
            print("DEBUG: Gemini call successful. Attempting JSON parse...")
            
            # Use regex to find the most likely valid JSON in case the model adds extra text
            json_match = re.search(r'\{.*\}', raw_content, re.DOTALL)
            
            if json_match:
                raw_json_string = json_match.group(0)
                return json.loads(raw_json_string)
            else:
                print(f"ERROR: Could not find valid JSON in Gemini response. Raw content start: {raw_content[:200]}")
                raise ValueError("Gemini response did not contain a valid JSON object.")

        except Exception as e:
            print(f"CRITICAL AI AGENT ERROR: Failed to get or parse response from Gemini: {e}")
            if i < 2:
                sleep_time = 2 ** (i + 1)
                print(f"Retrying in {sleep_time} seconds...")
                time.sleep(sleep_time)
            else:
                return None
    return None


def main_agent_workflow():
    """The main entry point for the scheduled job."""
    
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] --- STARTING AGENTIC AI RUN ---")

    from_email, subject, body = _fetch_latest_unread_email()

    if not from_email:
        print("STATUS: No new unread emails found to process. Exiting.")
        return

    print(f"STATUS: Found new email from: {from_email} (Subject: {subject})")
    
    email_data = {
        "from_email": from_email,
        "subject": subject,
        "body": body
    }

    ai_output = _run_ai_agent(email_data)

    if not ai_output:
        print(f"CRITICAL ERROR: Agentic AI failed to produce structured output for {from_email}. Exiting.")
        return

    SAFE_DEFAULT_REPLY = "Thank you for reaching out. I'm currently reviewing your inquiry and will send a proper, detailed response shortly. Best regards,\nAkash BV"
    
    # Extract results and handle potential string/boolean mismatch from LLM
    is_technical = ai_output.get("is_technical", False)
    if isinstance(is_technical, str):
         is_technical = is_technical.lower() == "true"
    
    request_meeting = ai_output.get("request_meeting", False)
    if isinstance(request_meeting, str):
        request_meeting = request_meeting.lower() == "true"

    
    simple_reply_draft = ai_output.get("simple_reply_draft", SAFE_DEFAULT_REPLY)
    non_technical_reply_draft = ai_output.get("non_technical_reply_draft", SAFE_DEFAULT_REPLY)
    meeting_suggestion_draft = ai_output.get("meeting_suggestion_draft", SAFE_DEFAULT_REPLY)
    
    print(f"AGENT RESULT: Is Technical/Project? {is_technical} | Request Meeting? {request_meeting}")

    final_subject = f"Re: {subject}"
    reply_draft = ""
    action_log = ""

    if is_technical:
        if request_meeting:
            reply_draft = meeting_suggestion_draft
            action_log = "Condition met AND tone required meeting. Sending meeting suggestion."
        else:
            reply_draft = simple_reply_draft
            action_log = "Condition met. Sending simple technical explanation."
    else:
        reply_draft = non_technical_reply_draft
        action_log = "Condition NOT met (General Inquiry). Sending polite, non-technical acknowledgement."
    
    # Clean up any residual HTML tags the LLM might have generated despite instructions
    reply_draft = re.sub(r'<[^>]+>', '', reply_draft).strip()

    # Add a polite salutation if one is missing
    if not reply_draft.lower().startswith(("hello", "hi", "dear", "thank you")):
          reply_draft = f"Hello,\n\n{reply_draft}"
        
    print(f"ACTION: {action_log}")
    print("ACTION: Attempting to send automated reply...")
    
    if _send_smtp_email(from_email, final_subject, reply_draft):
        print(f"SUCCESS: Automated reply sent to {from_email}.")
    else:
        print(f"FAILURE: Failed to send email to {from_email}.")
    
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] --- AGENTIC AI RUN COMPLETE ---")

if __name__ == "__main__":
    main_agent_workflow()
