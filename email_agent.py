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

# --- Configuration & Secrets ---
OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434/api/generate")
LLM_MODEL = os.environ.get("OLLAMA_MODEL", "mistral:7b-instruct-v0.2-q4_0")
EMAIL_ADDRESS = os.environ.get("EMAIL_ADDRESS")
EMAIL_PASSWORD = os.environ.get("EMAIL_PASSWORD")
SMTP_SERVER = "smtp.gmail.com"
SMTP_PORT = 465
IMAP_SERVER = "imap.gmail.com"

# --- LangSmith Configuration ---
langsmith_key = os.environ.get("LANGCHAIN_API_KEY")
if langsmith_key:
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_API_KEY"] = langsmith_key
    os.environ["LANGCHAIN_PROJECT"] = "Email_automation_schedule"
    print("STATUS: LangSmith tracing configured.")
else:
    os.environ["LANGCHAIN_TRACING_V2"] = "false"
    print("STATUS: LANGCHAIN_API_KEY not found. LangSmith tracing is disabled.")

# Knowledge base (shortened here for brevity, use your full text)
DATA_SCIENCE_KNOWLEDGE = """
... your knowledge base here ...
"""

AUTOMATION_CONDITION = (
    "Does the incoming email contain a technical question or an explicit project inquiry/pitch related to Data Science, "
    "Machine Learning (ML), Deep Learning, Data Engineering, advanced Statistical Analysis, or any service listed in the core offerings? "
)

AGENTIC_SYSTEM_INSTRUCTIONS = (
    "You are a professional, Agentic AI system acting ONLY as Senior Data Scientist, Akash BV... (full instructions as before) ..."
)

RESPONSE_SCHEMA_JSON = {
    "is_technical": "True if the email matches the technical/project condition, False otherwise.",
    "simple_reply_draft": "The primary reply to the client, simplified and non-technical, based on the knowledge base (USED IF is_technical is TRUE).",
    "non_technical_reply_draft": "A polite, professional acknowledgement and offer to help, used if is_technical is FALSE.",
    "request_meeting": "True if the tone suggests a serious project inquiry or pitch, False otherwise. (Triggers meeting suggestion).",
    "meeting_suggestion_draft": "If request_meeting is true, draft a reply suggesting available dates from the knowledge base."
}
RESPONSE_SCHEMA_PROMPT = json.dumps(RESPONSE_SCHEMA_JSON, indent=2)


def _send_smtp_email(to_email, subject, content):
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
            server.login(EMAIL_ADDRESS, EMAIL_PASSWORD)
            server.send_message(msg)
        return True
    except smtplib.SMTPAuthenticationError:
        print("CRITICAL SMTP ERROR: Authentication failed. Use 16-char App Password.")
        return False
    except Exception as e:
        print(f"ERROR: Failed to send email: {e}")
        return False


def _fetch_latest_unread_email():
    if not EMAIL_ADDRESS or not EMAIL_PASSWORD:
        print("CRITICAL ERROR: Email credentials missing.")
        return None, None, None
    try:
        mail = imaplib.IMAP4_SSL(IMAP_SERVER)
        mail.login(EMAIL_ADDRESS, EMAIL_PASSWORD)
        mail.select("inbox")
        status, data = mail.search(None, 'UNSEEN')
        ids = data[0].split()
        if not ids:
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

        return from_email, subject, body

    except Exception as e:
        print(f"CRITICAL ERROR fetching email: {e}")
        return None, None, None


def _run_ai_agent(email_data):
    if not OLLAMA_URL or not LLM_MODEL:
        print("CRITICAL ERROR: Ollama URL or model missing.")
        return None

    prompt = (
        f"**SYSTEM INSTRUCTIONS**:\n{AGENTIC_SYSTEM_INSTRUCTIONS}\n\n"
        f"**KNOWLEDGE BASE:**\n{DATA_SCIENCE_KNOWLEDGE}\n\n"
        f"**TASK CONFIGURATION:**\nCONDITION: {AUTOMATION_CONDITION}\n"
        f"Output ONLY a JSON matching schema:\n{RESPONSE_SCHEMA_PROMPT}\n\n"
        f"--- INCOMING EMAIL ---\nFROM: {email_data['from_email']}\nSUBJECT: {email_data['subject']}\nBODY:\n{email_data['body']}\n\n"
    )

    payload = {
        "model": LLM_MODEL,
        "prompt": prompt,
        "stream": False
    }

    headers = {'Content-Type': 'application/json'}

    for attempt in range(3):
        try:
            response = requests.post(OLLAMA_URL, headers=headers, data=json.dumps(payload), timeout=180)
            response.raise_for_status()
            raw_content = response.json().get('response', '').strip()

            json_match = re.search(r'\{.*\}', raw_content, re.DOTALL)
            if json_match:
                return json.loads(json_match.group(0))
            else:
                print(f"ERROR: No valid JSON in response. Raw start: {raw_content[:100]}")
                return None
        except requests.RequestException as e:
            print(f"HTTP error: {e}. Retry in {2 ** (attempt + 1)} seconds.")
            time.sleep(2 ** (attempt + 1))
    return None


def main_agent_workflow():
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] --- STARTING AGENTIC AI RUN ---")
    from_email, subject, body = _fetch_latest_unread_email()
    if not from_email:
        print("STATUS: No new emails. Exiting.")
        return

    email_data = {"from_email": from_email, "subject": subject, "body": body}
    ai_output = _run_ai_agent(email_data)

    if not ai_output:
        print(f"CRITICAL ERROR: No AI response for {from_email}")
        return

    def str_to_bool(val):
        return str(val).lower() == "true"

    is_technical = str_to_bool(ai_output.get("is_technical", "False"))
    request_meeting = str_to_bool(ai_output.get("request_meeting", "False"))
    simple_reply = ai_output.get("simple_reply_draft", "Thank you for your inquiry. Best regards,\nAkash BV")
    non_technical_reply = ai_output.get("non_technical_reply_draft", simple_reply)
    meeting_suggestion = ai_output.get("meeting_suggestion_draft", "Are you available this week on Monday, Wednesday, or Friday afternoon?")

    final_subject = f"Re: {subject}"

    if is_technical:
        reply = meeting_suggestion if request_meeting else simple_reply
    else:
        reply = non_technical_reply

    reply = re.sub(r'<[^>]+>', '', reply).strip()
    if not reply.lower().startswith(("hello", "hi", "thank you")):
        reply = "Hello,\n\n" + reply

    print(f"ACTION: Sending email reply to {from_email}...")
    if _send_smtp_email(from_email, final_subject, reply):
        print(f"SUCCESS: Email sent to {from_email}")
    else:
        print(f"FAILURE: Email send failed to {from_email}")
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] --- AGENTIC AI RUN COMPLETE ---")


if __name__ == "__main__":
    main_agent_workflow()
