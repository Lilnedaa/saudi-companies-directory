"""
Email Agent — reads Gmail inbox via IMAP and sends replies via SMTP.
Uses GPT-4o to classify emails and draft professional replies.

Setup (add to .env):
  GMAIL_ADDRESS=your.email@gmail.com
  GMAIL_APP_PASSWORD=xxxx xxxx xxxx xxxx   ← Gmail App Password (not your normal password)

How to get a Gmail App Password:
  1. Go to myaccount.google.com/security
  2. Enable 2-Step Verification
  3. Search "App passwords" → create one for "Mail"
  4. Copy the 16-character password to .env
"""

import os
import imaplib
import email
import smtplib
import json
from email.header import decode_header
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

GMAIL_ADDRESS = os.getenv("GMAIL_ADDRESS", "")
GMAIL_APP_PASSWORD = os.getenv("GMAIL_APP_PASSWORD", "")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")


def is_gmail_configured() -> bool:
    return bool(GMAIL_ADDRESS and GMAIL_APP_PASSWORD)


def fetch_unread_emails(limit: int = 15) -> list[dict]:
    """
    Connects to Gmail via IMAP and fetches the latest unread emails.
    Returns a list of dicts with id, subject, from, date, body.
    """
    mail = imaplib.IMAP4_SSL("imap.gmail.com")
    mail.login(GMAIL_ADDRESS, GMAIL_APP_PASSWORD)
    mail.select("inbox")

    _, message_numbers = mail.search(None, "UNSEEN")
    nums = message_numbers[0].split()
    if not nums:
        mail.logout()
        return []

    nums = nums[-limit:]
    result = []

    for num in reversed(nums):
        _, msg_data = mail.fetch(num, "(RFC822)")
        raw = msg_data[0][1]
        msg = email.message_from_bytes(raw)

        # Decode subject
        raw_subject, enc = decode_header(msg.get("Subject", "") or "")[0]
        if isinstance(raw_subject, bytes):
            subject = raw_subject.decode(enc or "utf-8", errors="ignore")
        else:
            subject = raw_subject or "(no subject)"

        from_addr = msg.get("From", "")
        date_str = msg.get("Date", "")

        # Extract plain text body
        body = ""
        if msg.is_multipart():
            for part in msg.walk():
                ct = part.get_content_type()
                cd = str(part.get("Content-Disposition", ""))
                if ct == "text/plain" and "attachment" not in cd:
                    payload = part.get_payload(decode=True)
                    if payload:
                        body = payload.decode("utf-8", errors="ignore")
                        break
        else:
            payload = msg.get_payload(decode=True)
            if payload:
                body = payload.decode("utf-8", errors="ignore")

        result.append({
            "id": num.decode(),
            "subject": subject,
            "from": from_addr,
            "date": date_str,
            "body": body[:2000],
        })

    mail.logout()
    return result


def classify_and_draft(email_data: dict) -> dict:
    """
    Uses GPT-4o to classify an email and draft a professional reply.
    Returns a dict with classification, signals, priority, draft_subject, draft_body.
    """
    if not OPENAI_API_KEY:
        return _fallback_draft(email_data)

    client = OpenAI(api_key=OPENAI_API_KEY)

    prompt = f"""You are a senior B2B sales email agent for BeamData, an AI and data solutions company in Saudi Arabia.

Analyze this incoming email and respond accordingly:

From: {email_data['from']}
Subject: {email_data['subject']}
Date: {email_data['date']}
Body:
{email_data['body'][:800]}

Tasks:
1. Classify the intent: Interested | Pricing Request | Meeting Request | Not Interested | Needs More Info | Other
2. Extract key signals (budget hints, timeline, specific AI/data needs mentioned)
3. Assess priority: High | Medium | Low
4. Draft a short, professional reply from BeamData (under 150 words) that directly addresses their message

Return ONLY valid JSON:
{{
  "classification": "<category>",
  "signals": "<key insights from the email>",
  "priority": "<High|Medium|Low>",
  "draft_subject": "Re: {email_data['subject']}",
  "draft_body": "<professional reply body>"
}}"""

    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a B2B sales email agent for BeamData. Return valid JSON only."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.3,
            max_tokens=500,
            response_format={"type": "json_object"},
        )
        return json.loads(response.choices[0].message.content.strip())
    except Exception:
        return _fallback_draft(email_data)


def _fallback_draft(email_data: dict) -> dict:
    return {
        "classification": "Other",
        "signals": "Could not analyze automatically.",
        "priority": "Medium",
        "draft_subject": f"Re: {email_data['subject']}",
        "draft_body": (
            "Dear Sir/Madam,\n\n"
            "Thank you for reaching out to BeamData. "
            "We appreciate your interest and will follow up with you shortly.\n\n"
            "Best regards,\nBeamData Team\nwww.beamdata.ai"
        ),
    }


def send_email(to_address: str, subject: str, body: str) -> None:
    """
    Sends an email via Gmail SMTP (SSL).
    Raises an exception if sending fails.
    """
    if not (GMAIL_ADDRESS and GMAIL_APP_PASSWORD):
        raise EnvironmentError(
            "Gmail not configured. Add GMAIL_ADDRESS and GMAIL_APP_PASSWORD to your .env file."
        )

    msg = MIMEMultipart()
    msg["From"] = GMAIL_ADDRESS
    msg["To"] = to_address
    msg["Subject"] = subject
    msg.attach(MIMEText(body, "plain", "utf-8"))

    with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
        server.login(GMAIL_ADDRESS, GMAIL_APP_PASSWORD)
        server.send_message(msg)
