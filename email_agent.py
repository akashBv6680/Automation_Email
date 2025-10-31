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
OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434/api/generate")
LLM_MODEL = os.environ.get("OLLAMA_MODEL", "mistral:7b-instruct-v0.2-q4_0")  # Fallback to corrected tag
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

# --- Condition to determine if email is technical ---
AUTOMATION_CONDITION = (
    "Does the incoming email contain a technical question or an explicit project inquiry/pitch related to Data Science, "
    "Machine Learning (ML), Deep Learning, Data Engineering, advanced Statistical Analysis, or any service listed in the core offerings? "
)

# --- Persona & reply style instructions ---
AGENTIC_SYSTEM_INSTRUCTIONS = (
    "You are a professional, Agentic AI system acting ONLY as Senior Data Scientist, Akash BV. You MUST NOT impersonate anyone else or reply to third-party content (like certificates or team emails).\n"
    "Your primary goal is to provide a helpful, professional, and courteous response to every email. Your task is to perform all required roles and provide a structured JSON output.\n"
    "1. CONDITION CHECK: Determine if the email is technical or a project pitch (based on the AUTOMATION_CONDITION).\n"
    "2. TRANSLATOR: Generate a reply. If technical, use the Knowledge Base for details. If non-technical, generate a polite general acknowledgment.\n"
    "3. TONE ANALYZER: If the email contains a serious project inquiry, set 'request_meeting' to true.\n\n"
    "CRITICAL FORMATTING GUIDANCE:\n"
    " - All generated drafts (simple_reply_draft, non_technical_reply_draft, meeting_suggestion_draft) MUST be in **PLAIN TEXT** format. **DO NOT USE HTML TAGS (like <br> or <b>)**.\n"
    " - All replies MUST be signed off with the exact signature: 'Best regards,\\nAkash BV'."
)

# JSON schema for the response
RESPONSE_SCHEMA_JSON = {
    "is_technical": "True if the email matches the technical/project condition, False otherwise.",
    "simple_reply_draft": "The primary reply to the client, simplified and non-technical, based on the knowledge base (USED IF is_technical is TRUE).",
    "non_technical_reply_draft": "A polite, professional acknowledgement and offer to help, used if is_technical is FALSE.",
    "request_meeting": "True if the tone suggests a serious project inquiry or pitch, False otherwise. (Triggers meeting suggestion).",
    "meeting_suggestion_draft": "If request_meeting is true, draft a reply suggesting available dates from the knowledge base (e.g., 'Are you available this week on Monday, Wednesday, or Friday afternoon?')."
}
RESPONSE_SCHEMA_PROMPT = json.dumps(RESPONSE_SCHEMA_JSON, indent=2)

# --- Helper functions ---
def _send_smtp_email(to_email, subject, content):
    """Send email via SMTP_SSL."""
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
            print(f"DEBUG: Logging into SMTP server {SMTP_SERVER}...")
            server.login(EMAIL_ADDRESS, EMAIL_PASSWORD)
            server.send_message(msg)
            print("DEBUG: Email sent successfully.")
        return True
    except smtplib.SMTPAuthenticationError:
        print("CRITICAL SMTP ERROR: Authentication failed.")
        return False
    except Exception as e:
        print(f"ERROR: Failed to send email: {e}")
        return False

def _fetch_latest_unread_email():
    """Fetch the latest unread email details."""
    if not EMAIL_ADDRESS or not EMAIL_PASSWORD:
        print("CRITICAL ERROR: Email credentials missing.")
        return None, None, None
    
    try:
        print(f"DEBUG: Connecting to IMAP server {IMAP_SERVER}...")
        mail = imaplib.IMAP4_SSL(IMAP_SERVER)
        mail.login(EMAIL_ADDRESS, EMAIL_PASSWORD)
        print("DEBUG: IMAP login successful.")
        mail.select("inbox")
        status, data = mail.search(None, 'UNSEEN')
        ids = data[0].split()

        if not ids:
            print("STATUS: No unread emails found.")
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

        print(f"DEBUG: Email from {from_email} with subject '{subject[:30]}...' fetched successfully.")
        return from_email, subject, body

    except imaplib.IMAP4.error as e:
        print(f"CRITICAL IMAP ERROR: {e}")
        return None, None, None
    except Exception as e:
        print(f"CRITICAL ERROR: {e}")
        return None, None, None

def _run_ai_agent(email_data):
    """Call Ollama local LLM with structured prompt."""
    if not OLLAMA_URL or not LLM_MODEL:
        print("CRITICAL ERROR: Ollama URL or model missing.")
        return None

    # Build prompt with schema
    prompt = (
        f"**SYSTEM INSTRUCTIONS**:\n{AGENTIC_SYSTEM_INSTRUCTIONS}\n\n"
        f"**KNOWLEDGE BASE (For context and reply)**:\n{DATA_SCIENCE_KNOWLEDGE}\n\n"
        f"**TASK CONFIGURATION**:\n"
        f"CONDITION TO CHECK: {AUTOMATION_CONDITION}\n"
        f"Output a JSON matching the schema below. Do not include extra text.\n\n"
        f"**RESPONSE SCHEMA**:\n{RESPONSE_SCHEMA_PROMPT}\n\n"
        f"--- INCOMING EMAIL ---\n"
        f"FROM: {email_data['from_email']}\n"
        f"SUBJECT: {email_data['subject']}\n"
        f"BODY:\n{email_data['body']}\n\n"
        "Generate the JSON object."
    )

    payload = {
        "model": LLM_MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.3,
            "top_p": 0.9,
            "num_predict": 2048
        }
    }

    headers = {'Content-Type': 'application/json'}

    for i in range(3):
        try:
            print(f"DEBUG: Ollama API call attempt {i+1}...")
            response = requests.post(OLLAMA_URL, headers=headers, data=json.dumps(payload), timeout=180)
            response.raise_for_status()
            response_json = response.json()

            raw_content = response_json.get('response', '').strip()

            # Extract JSON block
            json_match = re.search(r'\{.*\}', raw_content, re.DOTALL)
            if json_match:
                raw_json_string = json_match.group(0)
                print("DEBUG: JSON response received, parsing...")
                return json.loads(raw_json_string)
            else:
                print(f"ERROR: JSON not found in response. Raw: {raw_content[:200]}")
                raise ValueError("Invalid JSON")

        except requests.exceptions.RequestException as e:
            print(f"HTTP error: {e}. Retrying in {2 ** (i + 1)} seconds.")
            time.sleep(2 ** (i + 1))
        except Exception as e:
            print(f"CRITICAL AI ERROR: {e}")
            return None
    return None

def main_agent_workflow():
    """Main logic for scheduled email processing and AI reply."""
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] --- STARTING AGENTIC AI RUN ---")
    from_email, subject, body = _fetch_latest_unread_email()
    if not from_email:
        print("STATUS: No new emails. Exiting.")
        return

    print(f"STATUS: Email from {from_email} with subject '{subject}'")
    email_data = {
        "from_email": from_email,
        "subject": subject,
        "body": body
    }

    ai_output = _run_ai_agent(email_data)

    if not ai_output:
        print(f"CRITICAL ERROR: No structured output from AI for {from_email}")
        return

    # Handle AI output fields
    def str_to_bool(val):
        return str(val).lower() == "true"

    is_technical = ai_output.get("is_technical", "False")
    if isinstance(is_technical, str):
        is_technical = str_to_bool(is_technical)
    request_meeting = ai_output.get("request_meeting", "False")
    if isinstance(request_meeting, str):
        request_meeting = str_to_bool(request_meeting)

    simple_reply = ai_output.get("simple_reply_draft", "Thank you for reaching out. I'll review your inquiry.")
    non_technical_reply = ai_output.get("non_technical_reply_draft", "Thank you for your message. I'm here to assist.")
    meeting_suggestion = ai_output.get("meeting_suggestion_draft", "Would you be available this week on Monday, Wednesday, or Friday afternoon?")

    print(f"RESULT: Technical? {is_technical} | Request Meeting? {request_meeting}")
    
    final_subject = f"Re: {subject}"
    reply_draft = ""
    action_log = ""

    if is_technical:
        if request_meeting:
            reply_draft = meeting_suggestion
            action_log = "Technical email, requesting a meeting."
        else:
            reply_draft = simple_reply
            action_log = "Technical email, simple reply."
    else:
        reply_draft = non_technical_reply
        action_log = "General inquiry, polite acknowledgement."

    reply_draft = re.sub(r'<[^>]+>', '', reply_draft).strip()

    # Add greeting if not present
    if not reply_draft.lower().startswith(("hello", "hi", "thank you")):
        reply_draft = f"Hello,\n\n{reply_draft}"

    print(f"ACTION: {action_log}")
    print("ACTION: Sending reply...")

    if _send_smtp_email(from_email, final_subject, reply_draft):
        print(f"SUCCESS: Replied to {from_email}")
    else:
        print(f"FAILURE: Could not send email to {from_email}")
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] --- AI RUN COMPLETE ---")

if __name__ == "__main__":
    main_agent_workflow()
