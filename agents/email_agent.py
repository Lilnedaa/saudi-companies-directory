"""
Email Agent — Gmail OAuth integration.

Uses the Gmail API (not IMAP/SMTP) so the user signs in once with their
Google account via a browser consent flow. Tokens are cached in
`token.json` next to `credentials.json` at the project root.

Setup (one-time):
  1. Google Cloud Console → APIs & Services → Credentials → OAuth client
     ID → Desktop app, then save the file as `credentials.json` in this
     project's root directory.
  2. Launch the app, click "Sign in to Gmail" in the sidebar — a browser
     window opens for consent. `token.json` is written on success.
"""
import os
import re
import json
import base64
from pathlib import Path
from email.mime.text import MIMEText

from openai import OpenAI
from dotenv import load_dotenv

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

SCOPES = [
    "https://www.googleapis.com/auth/gmail.readonly",
    "https://www.googleapis.com/auth/gmail.send",
    "https://www.googleapis.com/auth/gmail.modify",
]

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CREDENTIALS_PATH = PROJECT_ROOT / "credentials.json"
TOKEN_PATH = PROJECT_ROOT / "token.json"

_gmail_service = None


def _get_gmail_service():
    """Return an authenticated Gmail API client. Opens a browser for OAuth
    consent on first use, then reuses the cached token."""
    global _gmail_service
    if _gmail_service is not None:
        return _gmail_service

    creds = None
    if TOKEN_PATH.exists():
        creds = Credentials.from_authorized_user_file(str(TOKEN_PATH), SCOPES)

    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            if not CREDENTIALS_PATH.exists():
                raise FileNotFoundError(
                    f"Missing OAuth client file at {CREDENTIALS_PATH}.\n"
                    "Get it from Google Cloud Console → APIs & Services → "
                    "Credentials → OAuth client ID → Desktop app, then "
                    "save it here as credentials.json."
                )
            flow = InstalledAppFlow.from_client_secrets_file(
                str(CREDENTIALS_PATH), SCOPES
            )
            creds = flow.run_local_server(port=0)
        TOKEN_PATH.write_text(creds.to_json())

    _gmail_service = build("gmail", "v1", credentials=creds)
    return _gmail_service


# ── Auth helpers (called from the Streamlit sidebar) ─────────────────

def is_signed_in() -> bool:
    """True if a usable cached token exists. Never triggers consent."""
    if not TOKEN_PATH.exists():
        return False
    try:
        creds = Credentials.from_authorized_user_file(str(TOKEN_PATH), SCOPES)
    except Exception:
        return False
    if not creds:
        return False
    if creds.valid:
        return True
    return bool(creds.expired and creds.refresh_token)


def get_authenticated_email() -> str | None:
    """If signed in, return the connected Gmail address; else None.
    May refresh silently but never opens a browser."""
    if not is_signed_in():
        return None
    try:
        service = _get_gmail_service()
        profile = service.users().getProfile(userId="me").execute()
        return profile.get("emailAddress")
    except Exception:
        return None


def sign_in_gmail() -> dict:
    """Run the OAuth consent flow (opens a browser). Returns
    {status, email}. Raises FileNotFoundError if credentials.json
    is missing."""
    global _gmail_service
    _gmail_service = None
    service = _get_gmail_service()
    profile = service.users().getProfile(userId="me").execute()
    return {
        "status": "connected",
        "email": profile.get("emailAddress"),
        "messages_total": profile.get("messagesTotal"),
    }


def sign_out_gmail() -> dict:
    """Disconnect by deleting the cached token."""
    global _gmail_service
    _gmail_service = None
    if TOKEN_PATH.exists():
        TOKEN_PATH.unlink()
    return {"status": "disconnected"}


def is_gmail_configured() -> bool:
    """Backwards-compatible alias used across the UI. With OAuth this is
    equivalent to "the user is signed in"."""
    return is_signed_in()


# ── Email body extraction (Gmail API format) ─────────────────────────

def _decode(data: str) -> str:
    return base64.urlsafe_b64decode(data).decode("utf-8", errors="ignore")


def _extract_body(payload: dict) -> str:
    if "parts" in payload:
        for part in payload["parts"]:
            if part.get("mimeType") == "text/plain" and part.get("body", {}).get("data"):
                return _decode(part["body"]["data"])
        for part in payload["parts"]:
            nested = _extract_body(part)
            if nested:
                return nested
    if payload.get("body", {}).get("data"):
        return _decode(payload["body"]["data"])
    return ""


# ── Inbox ────────────────────────────────────────────────────────────

def fetch_unread_emails(limit: int = 15) -> list[dict]:
    """Fetch unread Gmail messages via the Gmail API.
    Returns dicts with id, subject, from, date, body."""
    service = _get_gmail_service()
    listing = service.users().messages().list(
        userId="me", q="is:unread", maxResults=limit
    ).execute()

    refs = listing.get("messages", [])
    emails = []
    for ref in refs:
        msg = service.users().messages().get(
            userId="me", id=ref["id"], format="full"
        ).execute()
        headers = {h["name"].lower(): h["value"] for h in msg["payload"]["headers"]}
        body = _extract_body(msg["payload"])
        emails.append({
            "id": ref["id"],
            "thread_id": msg.get("threadId"),
            "subject": headers.get("subject", "(no subject)"),
            "from": headers.get("from", ""),
            "date": headers.get("date", ""),
            "body": body[:2000],
        })
    return emails


# ── AI classification & draft (unchanged behaviour) ──────────────────

def classify_and_draft(email_data: dict) -> dict:
    """Classify an incoming email and draft a professional reply."""
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


# ── Sending ──────────────────────────────────────────────────────────

_EMAIL_REGEX = re.compile(r"^[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}$")


def _validate_email(address: str) -> None:
    if not address or not address.strip():
        raise ValueError("Recipient email address is empty.")
    if not _EMAIL_REGEX.match(address.strip()):
        raise ValueError(f"Invalid email address format: '{address}'")


def send_email(to_address: str, subject: str, body: str,
               thread_id: str | None = None) -> dict:
    """Send an email via the Gmail API as the signed-in user."""
    if not is_signed_in():
        raise EnvironmentError(
            "Not signed in to Gmail. Click 'Sign in to Gmail' in the sidebar first."
        )

    _validate_email(to_address)
    if not subject or not subject.strip():
        raise ValueError("Email subject cannot be empty.")
    if not body or not body.strip():
        raise ValueError("Email body cannot be empty.")

    service = _get_gmail_service()
    mime = MIMEText(body)
    mime["to"] = to_address.strip()
    mime["subject"] = subject
    raw = base64.urlsafe_b64encode(mime.as_bytes()).decode()

    payload = {"raw": raw}
    if thread_id:
        payload["threadId"] = thread_id

    sent = service.users().messages().send(userId="me", body=payload).execute()

    return {
        "status": "sent",
        "message_id": sent.get("id"),
        "thread_id": sent.get("threadId"),
        "to": to_address.strip(),
        "subject": subject,
    }
