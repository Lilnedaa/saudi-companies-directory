
import os
import json
import urllib.request
from dotenv import load_dotenv

load_dotenv()

OPENAI_API_URL = "https://api.openai.com/v1/chat/completions"
ANTHROPIC_API_URL = "https://api.anthropic.com/v1/messages"
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
BEAMDATA_CRITERIA = """
- Sector priority (High): Information Technology, Financials/Fintech, Telecom, Health Care
- Sector priority (Medium): Consumer Discretionary, Industrials, Energy
- Sector priority (Low): Consumer Staples, Materials
- Company size: prefer 200+ employees (larger = bigger AI budget)
- Location: Riyadh is top priority, other Saudi cities are acceptable
- Digital presence: having a website increases accessibility
- Established companies preferred over early-stage startups for enterprise AI deals
- BeamData sells: AI Hub Platform, Data & AI Strategy, POC development, Deployment, AI Governance
"""


def web_search(query: str) -> str:
    """Search the web using Claude's web search tool."""
    payload = json.dumps({
        "model": "claude-sonnet-4-6",
        "max_tokens": 1024,
        "tools": [{"type": "web_search_20250305", "name": "web_search"}],
        "messages": [{"role": "user", "content": f"Search for: {query}. Return a brief summary of what you find in 2-3 sentences."}]
    }).encode("utf-8")

    req = urllib.request.Request(
        ANTHROPIC_API_URL,
        data=payload,
        headers={
            "Content-Type": "application/json",
            "x-api-key": ANTHROPIC_API_KEY,
            "anthropic-version": "2023-06-01"
        },
        method="POST"
    )

    with urllib.request.urlopen(req, timeout=30) as resp:
        data = json.loads(resp.read().decode("utf-8"))

    # Extract text from response
    for block in data.get("content", []):
        if block.get("type") == "text":
            return block["text"]
    return "No results found."


def research_company(company: dict) -> str:
    """Use web search to get fresh info about a company."""
    name = company.get("name", "")
    query = f"{name} Saudi Arabia AI technology latest news 2024 2025 stock exchange Tadawul"
    try:
        result = web_search(query)
        return result
    except Exception as e:
        return f"Could not research: {str(e)}"


def score_company_with_research(company: dict, criteria: str, use_beamdata_defaults: bool, research: str) -> dict:
    """Score a single company using GPT with fresh research data."""
    final_criteria = BEAMDATA_CRITERIA if use_beamdata_defaults else criteria

    prompt = f"""You are a B2B sales qualification agent for BeamData AI.

SCORING CRITERIA:
{final_criteria}

COMPANY DATA (from our database):
- Name: {company.get('name', 'N/A')}
- Sector: {company.get('sector', 'N/A')}
- Sub-sector: {company.get('sub_sector', 'N/A')}
- City: {company.get('city_clean', 'N/A')}
- Employees: {company.get('employees', 'N/A')}
- Website: {company.get('website', 'N/A')}
- Description: {company.get('description', 'N/A')}
- Is Startup: {company.get('is_startup', False)}

FRESH RESEARCH FROM WEB:
{research}

Based on BOTH the database info AND the fresh research, score this company.
- 70-100 = High
- 40-69 = Medium  
- 0-39 = Low

Respond ONLY with JSON, no markdown:
{{"score": 85, "grade": "High", "reason": "One sentence explaining why based on research"}}"""

    payload = json.dumps({
        "model": "gpt-4o-mini",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.2
    }).encode("utf-8")

    req = urllib.request.Request(
        OPENAI_API_URL,
        data=payload,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {OPENAI_API_KEY}"
        },
        method="POST"
    )

    with urllib.request.urlopen(req, timeout=30) as resp:
        data = json.loads(resp.read().decode("utf-8"))

    raw = data["choices"][0]["message"]["content"].strip()
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    raw = raw.strip()

    result = json.loads(raw)
    company_result = dict(company)
    company_result["score"] = result.get("score", 0)
    company_result["grade"] = result.get("grade", "Low")
    company_result["reason"] = result.get("reason", "")
    company_result["research"] = research
    return company_result


def score_companies_basic(companies: list, criteria: str, use_beamdata_defaults: bool = False) -> list:
    """Basic scoring without web search (fast)."""
    final_criteria = BEAMDATA_CRITERIA if use_beamdata_defaults else criteria

    companies_text = ""
    for i, c in enumerate(companies):
        companies_text += f"""
Company {i+1}:
- Name: {c.get('name', 'N/A')}
- Sector: {c.get('sector', 'N/A')}
- City: {c.get('city_clean', 'N/A')}
- Employees: {c.get('employees', 'N/A')}
- Website: {c.get('website', 'N/A')}
- Description: {c.get('description', 'N/A')}
- Is Startup: {c.get('is_startup', False)}
"""

    prompt = f"""You are a B2B sales qualification agent.

SCORING CRITERIA:
{final_criteria}

COMPANIES TO SCORE:
{companies_text}

Score each company 0-100. Respond ONLY with JSON array:
[{{"index": 1, "score": 85, "grade": "High", "reason": "One sentence"}}]"""

    payload = json.dumps({
        "model": "gpt-4o-mini",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.2
    }).encode("utf-8")

    req = urllib.request.Request(
        OPENAI_API_URL,
        data=payload,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {OPENAI_API_KEY}"
        },
        method="POST"
    )

    with urllib.request.urlopen(req, timeout=30) as resp:
        data = json.loads(resp.read().decode("utf-8"))

    raw = data["choices"][0]["message"]["content"].strip()
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    raw = raw.strip()

    scores = json.loads(raw)
    results = []
    for item in scores:
        idx = item["index"] - 1
        if 0 <= idx < len(companies):
            company = dict(companies[idx])
            company["score"] = item["score"]
            company["grade"] = item["grade"]
            company["reason"] = item["reason"]
            company["research"] = ""
            results.append(company)

    results.sort(key=lambda x: x.get("score", 0), reverse=True)
    return results


def score_in_batches(companies: list, criteria: str, use_beamdata_defaults: bool = False, batch_size: int = 15, use_agent: bool = False, progress_callback=None) -> list:
    """Score companies - basic or with agent web search."""

    if use_agent:
        # Agent mode: research each company individually
        results = []
        for i, company in enumerate(companies):
            if progress_callback:
                progress_callback(i + 1, len(companies), company.get('name', ''))
            research = research_company(company)
            result = score_company_with_research(company, criteria, use_beamdata_defaults, research)
            results.append(result)
        results.sort(key=lambda x: x.get("score", 0), reverse=True)
        return results
    else:
        # Basic mode: batch scoring
        all_results = []
        for i in range(0, len(companies), batch_size):
            batch = companies[i:i + batch_size]
            batch_results = score_companies_basic(batch, criteria, use_beamdata_defaults)
            all_results.extend(batch_results)
        all_results.sort(key=lambda x: x.get("score", 0), reverse=True)
        return all_results