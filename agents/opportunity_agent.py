"""
Opportunity Qualification Agent
================================
Part of the AI Sales CRM Pipeline:
  1. Lead Generation → 2. Filter → 3. Lead Qualification →
  4. Campaign → 5. Opportunity Qualification ← (YOU ARE HERE)
  → 6. Proposal → 7. Conversion

Evaluates campaign responses and qualifies opportunities for the Proposal Agent.
"""

import os
import json
import logging
import argparse
import time
from typing import Optional

import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI, OpenAIError

# ─────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────

load_dotenv()

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

API_KEY = os.getenv("OPENAI_API_KEY")
MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
MAX_RETRIES = int(os.getenv("MAX_RETRIES", "3"))
RETRY_DELAY = float(os.getenv("RETRY_DELAY", "2.0"))       # seconds between retries
QUALIFY_THRESHOLD = int(os.getenv("QUALIFY_THRESHOLD", "70"))  # min score to flag for review

if not API_KEY:
    raise EnvironmentError(
        "Missing OPENAI_API_KEY. Add it to your .env file or environment variables."
    )

client = OpenAI(api_key=API_KEY)

# ─────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────

REQUIRED_COLUMNS: list[str] = [
    "company_name",
    "industry",
    "budget",
    "email_response",
    "need",
    "decision_maker",
    "timeline",
    "agreement_level",
    "potential_value",
]

SYSTEM_PROMPT = """You are an Opportunity Qualification Agent for an AI Sales CRM.
Your role is to objectively score and qualify sales opportunities after the campaign stage.
Always respond with valid JSON only — no explanation, no markdown, no preamble."""

EVALUATION_TEMPLATE = """Evaluate this sales opportunity using the criteria below.

Scoring Criteria (0–100):
- Budget:               High=25pts  Medium=15pts  Low=5pts
- Customer Need:        Clear=20pts  Vague=10pts  None=0pts
- Email Response:       Engaged=20pts  Neutral=10pts  None=0pts
- Decision Maker:       Available=15pts  Partial=8pts  None=0pts
- Timeline:             <3mo=10pts  3-6mo=6pts  Unknown=2pts
- Agreement Level:      Strong=5pts  Medium=3pts  Weak=0pts
- Potential Value:      High=5pts  Medium=3pts  Low=0pts

Company Data:
  Company Name:    {company_name}
  Industry:        {industry}
  Budget:          {budget}
  Email Response:  {email_response}
  Need:            {need}
  Decision Maker:  {decision_maker}
  Timeline:        {timeline}
  Agreement Level: {agreement_level}
  Potential Value: {potential_value}

Return ONLY valid JSON — no extra text:
{{
  "opportunity_score": <integer 0–100>,
  "status": "<Qualified | Not Qualified>",
  "reason": "<concise one-sentence reason>",
  "recommended_next_step": "<concrete next action>"
}}"""

# ─────────────────────────────────────────────
# Validation
# ─────────────────────────────────────────────

def validate_input(df: pd.DataFrame) -> None:
    """Raise ValueError if any required column is missing."""
    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"Input CSV is missing required columns: {missing}")
    logger.info("Input validation passed — %d rows, %d columns.", len(df), len(df.columns))


# ─────────────────────────────────────────────
# AI Evaluation
# ─────────────────────────────────────────────

def evaluate_opportunity(row: pd.Series) -> dict:
    """
    Call the OpenAI API to score and qualify a single opportunity.
    Retries up to MAX_RETRIES times on transient API errors.
    """
    prompt = EVALUATION_TEMPLATE.format(
        company_name=row.get("company_name", "N/A"),
        industry=row.get("industry", "N/A"),
        budget=row.get("budget", "N/A"),
        email_response=row.get("email_response", "N/A"),
        need=row.get("need", "N/A"),
        decision_maker=row.get("decision_maker", "N/A"),
        timeline=row.get("timeline", "N/A"),
        agreement_level=row.get("agreement_level", "N/A"),
        potential_value=row.get("potential_value", "N/A"),
    )

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = client.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.2,        # low temperature → consistent, factual output
                max_tokens=300,
                response_format={"type": "json_object"},   # enforces JSON mode
            )
            result_text = response.choices[0].message.content.strip()
            result = json.loads(result_text)

            # Sanitise score: must be an integer in [0, 100]
            raw_score = result.get("opportunity_score", 0)
            result["opportunity_score"] = max(0, min(100, int(raw_score)))

            # Sanitise status
            if result.get("status") not in ("Qualified", "Not Qualified"):
                result["status"] = (
                    "Qualified" if result["opportunity_score"] >= QUALIFY_THRESHOLD
                    else "Not Qualified"
                )

            logger.debug(
                "Scored '%s': %d — %s",
                row.get("company_name"),
                result["opportunity_score"],
                result["status"],
            )
            return result

        except json.JSONDecodeError as exc:
            logger.warning("Attempt %d/%d — JSON parse error: %s", attempt, MAX_RETRIES, exc)
        except OpenAIError as exc:
            logger.warning("Attempt %d/%d — OpenAI API error: %s", attempt, MAX_RETRIES, exc)
        except (ValueError, TypeError) as exc:
            logger.warning("Attempt %d/%d — Data error: %s", attempt, MAX_RETRIES, exc)

        if attempt < MAX_RETRIES:
            time.sleep(RETRY_DELAY)

    logger.error("All %d attempts failed for '%s'.", MAX_RETRIES, row.get("company_name"))
    return {
        "opportunity_score": 0,
        "status": "Error",
        "reason": f"AI evaluation failed after {MAX_RETRIES} attempts.",
        "recommended_next_step": "Manual review required.",
    }


# ─────────────────────────────────────────────
# Human-in-the-Loop
# ─────────────────────────────────────────────

def human_review(company_name: str, ai_result: dict) -> dict:
    """
    Interactive human review step for qualified opportunities.
    Returns a dict with human_decision, final_status, and human_note.
    """
    divider = "=" * 65
    print(f"\n{divider}")
    print(f"  HUMAN REVIEW REQUIRED — {company_name}")
    print(divider)
    print(f"  AI Score        : {ai_result.get('opportunity_score')}/100")
    print(f"  AI Status       : {ai_result.get('status')}")
    print(f"  Reason          : {ai_result.get('reason')}")
    print(f"  Recommended Step: {ai_result.get('recommended_next_step')}")
    print(divider)
    print("  Options:")
    print("    1 → Approve  (move to Proposal Agent)")
    print("    2 → Reject   (stop this opportunity)")
    print("    3 → Revise   (modify next step and keep in pipeline)")
    print(divider)

    while True:
        choice = input("  Enter choice [1/2/3]: ").strip()

        if choice == "1":
            return {
                "human_decision": "Approved",
                "final_status": "Move to Proposal",
                "human_note": "Approved by human reviewer.",
            }

        if choice == "2":
            note = input("  Rejection reason: ").strip() or "Rejected by human reviewer."
            return {
                "human_decision": "Rejected",
                "final_status": "Stopped",
                "human_note": note,
            }

        if choice == "3":
            new_step = input("  Revised next step: ").strip() or "Follow-up needed."
            return {
                "human_decision": "Revised",
                "final_status": "Needs Follow-up",
                "human_note": new_step,
            }

        print("  ⚠  Invalid input. Please enter 1, 2, or 3.")


def _default_human_result(needs_review: bool) -> dict:
    """Return the appropriate human_result when human-loop is disabled."""
    if needs_review:
        return {
            "human_decision": "Pending Review",
            "final_status": "Needs Human Review",
            "human_note": "Human approval required before moving to Proposal Agent.",
        }
    return {
        "human_decision": "Not Required",
        "final_status": "Not Qualified",
        "human_note": "Scored below threshold or explicitly not qualified by AI.",
    }


def evaluate_single_opportunity(data: dict) -> dict:
    """
    Evaluate one opportunity from a plain dict (e.g. a form submitted in
    the Streamlit app), without needing a CSV file or terminal input.

    Expected keys (same as REQUIRED_COLUMNS, minus company_name which is
    just carried through): industry, budget, email_response, need,
    decision_maker, timeline, agreement_level, potential_value.

    Returns the same dict shape as evaluate_opportunity(), i.e.
    {opportunity_score, status, reason, recommended_next_step}.
    """
    row = pd.Series(data)
    return evaluate_opportunity(row)


# ─────────────────────────────────────────────
# Auto BANT Assessment (no manual form fields)
# ─────────────────────────────────────────────

AUTO_BANT_PROMPT = """You are an expert sales qualification agent for BeamData, an AI and data solutions company in Saudi Arabia.

Analyze this potential client and evaluate the sales opportunity using the BANT framework.
Infer each criterion intelligently from the company profile and their email reply (if any).

Company Profile:
  Name: {company_name}
  Sector: {sector}
  Sub-sector: {sub_sector}
  Employees: {employees}
  City: {city}
  Description: {description}
  Tags: {tags}
  Founded: {founded_year}

Campaign Email Subject Sent: {email_subject}

Their Reply:
{email_reply}

Evaluate using BANT (each scored 0-25):
- Budget (0-25): Infer budget capacity from company size, sector maturity, and any reply signals.
- Authority (0-25): Assess if the respondent or company structure suggests decision-maker involvement.
- Need (0-25): Evaluate how strongly their sector, description, and reply indicate a need for AI/data solutions.
- Timeline (0-25): Infer urgency from reply tone, sector trends, and any time-related signals.

Return ONLY valid JSON:
{{
  "bant": {{
    "budget_score": <0-25>,
    "budget_reason": "<one sentence>",
    "authority_score": <0-25>,
    "authority_reason": "<one sentence>",
    "need_score": <0-25>,
    "need_reason": "<one sentence>",
    "timeline_score": <0-25>,
    "timeline_reason": "<one sentence>"
  }},
  "opportunity_score": <integer 0-100, sum of BANT scores>,
  "status": "<Qualified | Not Qualified>",
  "reason": "<concise one-sentence overall assessment>",
  "recommended_next_step": "<concrete next action>"
}}"""


def evaluate_opportunity_auto(company_data: dict, email_reply: str = "") -> dict:
    """
    Automatically evaluates an opportunity using BANT framework.
    No manual form inputs needed — AI infers everything from company profile + reply.
    """
    prompt = AUTO_BANT_PROMPT.format(
        company_name=company_data.get("company_name", "N/A"),
        sector=company_data.get("sector", "N/A"),
        sub_sector=company_data.get("sub_sector", "N/A"),
        employees=company_data.get("employees", "N/A"),
        city=company_data.get("city", "N/A"),
        description=company_data.get("description", "N/A"),
        tags=company_data.get("tags", "N/A"),
        founded_year=company_data.get("founded_year", "N/A"),
        email_subject=company_data.get("email_subject", "BeamData Introduction"),
        email_reply=email_reply.strip() if email_reply.strip() else "(No reply received yet — evaluate based on company profile only)",
    )

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = client.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.2,
                max_tokens=500,
                response_format={"type": "json_object"},
            )
            result_text = response.choices[0].message.content.strip()
            result = json.loads(result_text)

            # ── Guardrail: cap each BANT sub-score at its max (25) ──
            if isinstance(result.get("bant"), dict):
                bant = result["bant"]
                for key in ("budget_score", "authority_score", "need_score", "timeline_score"):
                    bant[key] = max(0, min(25, int(bant.get(key, 0))))
                result["opportunity_score"] = sum(
                    bant[k] for k in ("budget_score", "authority_score", "need_score", "timeline_score")
                )
            else:
                raw_score = result.get("opportunity_score", 0)
                result["opportunity_score"] = max(0, min(100, int(raw_score)))

            if result.get("status") not in ("Qualified", "Not Qualified"):
                result["status"] = (
                    "Qualified" if result["opportunity_score"] >= QUALIFY_THRESHOLD
                    else "Not Qualified"
                )

            result["assessment_mode"] = "auto_bant"
            return result

        except (json.JSONDecodeError, OpenAIError, ValueError, TypeError) as exc:
            logger.warning("Auto BANT attempt %d/%d failed: %s", attempt, MAX_RETRIES, exc)
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_DELAY)

    return {
        "opportunity_score": 0,
        "status": "Error",
        "reason": f"AI evaluation failed after {MAX_RETRIES} attempts.",
        "recommended_next_step": "Manual review required.",
        "assessment_mode": "auto_bant",
    }


# ─────────────────────────────────────────────
# Custom Criteria Assessment
# ─────────────────────────────────────────────

CUSTOM_CRITERIA_PROMPT = """You are an expert sales qualification agent for BeamData, an AI and data solutions company in Saudi Arabia.

Evaluate this sales opportunity based on the custom scoring criteria defined below.

Company Profile:
  Name: {company_name}
  Sector: {sector}
  Sub-sector: {sub_sector}
  Employees: {employees}
  City: {city}
  Description: {description}
  Tags: {tags}
  Founded: {founded_year}

Their Reply:
{email_reply}

Custom Scoring Criteria:
{custom_criteria}

Instructions:
- Evaluate the company against EACH criterion listed.
- Score each criterion out of 100.
- Calculate a weighted average for the total opportunity_score.
- Be objective and base scores on evidence from the company profile and reply.

Return ONLY valid JSON:
{{
  "criteria_scores": [
    {{"criterion": "<name>", "score": <0-100>, "reason": "<one sentence>"}}
  ],
  "opportunity_score": <integer 0-100, weighted average>,
  "status": "<Qualified | Not Qualified>",
  "reason": "<concise one-sentence overall assessment>",
  "recommended_next_step": "<concrete next action>"
}}"""


def evaluate_opportunity_custom(company_data: dict, custom_criteria: str, email_reply: str = "") -> dict:
    """
    Evaluates an opportunity against user-defined custom scoring criteria.
    """
    prompt = CUSTOM_CRITERIA_PROMPT.format(
        company_name=company_data.get("company_name", "N/A"),
        sector=company_data.get("sector", "N/A"),
        sub_sector=company_data.get("sub_sector", "N/A"),
        employees=company_data.get("employees", "N/A"),
        city=company_data.get("city", "N/A"),
        description=company_data.get("description", "N/A"),
        tags=company_data.get("tags", "N/A"),
        founded_year=company_data.get("founded_year", "N/A"),
        email_reply=email_reply.strip() if email_reply.strip() else "(No reply received yet — evaluate based on company profile only)",
        custom_criteria=custom_criteria.strip(),
    )

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = client.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.2,
                max_tokens=600,
                response_format={"type": "json_object"},
            )
            result_text = response.choices[0].message.content.strip()
            result = json.loads(result_text)

            raw_score = result.get("opportunity_score", 0)
            result["opportunity_score"] = max(0, min(100, int(raw_score)))

            if result.get("status") not in ("Qualified", "Not Qualified"):
                result["status"] = (
                    "Qualified" if result["opportunity_score"] >= QUALIFY_THRESHOLD
                    else "Not Qualified"
                )

            result["assessment_mode"] = "custom_criteria"
            return result

        except (json.JSONDecodeError, OpenAIError, ValueError, TypeError) as exc:
            logger.warning("Custom criteria attempt %d/%d failed: %s", attempt, MAX_RETRIES, exc)
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_DELAY)

    return {
        "opportunity_score": 0,
        "status": "Error",
        "reason": f"AI evaluation failed after {MAX_RETRIES} attempts.",
        "recommended_next_step": "Manual review required.",
        "assessment_mode": "custom_criteria",
    }


# ─────────────────────────────────────────────
# Main Pipeline
# ─────────────────────────────────────────────

def run_opportunity_agent(
    input_file: str,
    output_file: str,
    proposal_output_file: str,
    human_loop: bool = False,
) -> None:
    """
    Main entry point for the Opportunity Qualification stage.

    Parameters
    ----------
    input_file          : CSV from the Campaign Agent.
    output_file         : Full results CSV (all opportunities).
    proposal_output_file: Filtered CSV for the Proposal Agent.
    human_loop          : If True, prompt a human for each qualified opportunity.
    """
    logger.info("Loading input file: %s", input_file)
    df = pd.read_csv(input_file)
    validate_input(df)

    results: list[dict] = []
    total = len(df)

    for idx, (_, row) in enumerate(df.iterrows(), start=1):
        company_name: str = str(row.get("company_name", "Unknown"))
        logger.info("[%d/%d] Evaluating: %s", idx, total, company_name)

        ai_result = evaluate_opportunity(row)
        score: int = ai_result.get("opportunity_score", 0)
        status: str = ai_result.get("status", "")
        needs_review: bool = status == "Qualified" or score >= QUALIFY_THRESHOLD

        if human_loop and needs_review:
            human_result = human_review(company_name, ai_result)
        else:
            human_result = _default_human_result(needs_review)

        results.append(
            {
                # ── Input fields ──────────────────────────────
                "company_name": company_name,
                "industry": row.get("industry", ""),
                "budget": row.get("budget", ""),
                "email_response": row.get("email_response", ""),
                "need": row.get("need", ""),
                "decision_maker": row.get("decision_maker", ""),
                "timeline": row.get("timeline", ""),
                "agreement_level": row.get("agreement_level", ""),
                "potential_value": row.get("potential_value", ""),
                # ── AI output ─────────────────────────────────
                "ai_opportunity_score": score,
                "ai_status": status,
                "ai_reason": ai_result.get("reason", ""),
                "ai_recommended_next_step": ai_result.get("recommended_next_step", ""),
                # ── Human decision ────────────────────────────
                "human_decision": human_result.get("human_decision", ""),
                "final_status": human_result.get("final_status", ""),
                "human_note": human_result.get("human_note", ""),
            }
        )

    # Build output DataFrame
    output_df = pd.DataFrame(results).sort_values(
        by="ai_opportunity_score", ascending=False
    )
    output_df.to_csv(output_file, index=False)
    logger.info("Full results saved → %s", output_file)

    # Proposal-ready slice (for the next pipeline stage)
    proposal_df = output_df[output_df["final_status"] == "Move to Proposal"].copy()
    proposal_df.to_csv(proposal_output_file, index=False)
    logger.info("Proposal-ready opportunities saved → %s", proposal_output_file)

    # ── Summary ──────────────────────────────────────────────
    qualified = (output_df["ai_status"] == "Qualified").sum()
    proposal_ready = len(proposal_df)
    pending = (output_df["final_status"] == "Needs Human Review").sum()

    print("\n" + "=" * 65)
    print("  OPPORTUNITY QUALIFICATION — SUMMARY")
    print("=" * 65)
    print(f"  Total Evaluated   : {total}")
    print(f"  AI Qualified      : {qualified}")
    print(f"  Proposal Ready    : {proposal_ready}")
    print(f"  Pending Review    : {pending}")
    print("=" * 65)

    summary_cols = [
        "company_name",
        "ai_opportunity_score",
        "ai_status",
        "human_decision",
        "final_status",
    ]
    print(output_df[summary_cols].to_string(index=False))
    print()


# ─────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Opportunity Qualification Agent — AI Sales CRM Pipeline (Stage 5)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input",
        default="campaign_responses.csv",
        help="Input CSV produced by the Campaign Agent.",
    )
    parser.add_argument(
        "--output",
        default="qualified_opportunities.csv",
        help="Full output CSV with all evaluation results.",
    )
    parser.add_argument(
        "--proposal-output",
        default="proposal_ready_opportunities.csv",
        help="Filtered CSV with proposal-ready opportunities for the Proposal Agent.",
    )
    parser.add_argument(
        "--human-loop",
        action="store_true",
        help="Enable interactive human review for qualified opportunities.",
    )
    parser.add_argument(
        "--threshold",
        type=int,
        default=QUALIFY_THRESHOLD,
        help="Minimum AI score (0–100) to flag an opportunity for review.",
    )
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    # Allow CLI override of the qualify threshold
    global QUALIFY_THRESHOLD
    QUALIFY_THRESHOLD = args.threshold

    logger.info("Starting Opportunity Qualification Agent.")
    logger.info("Model: %s | Human-loop: %s | Threshold: %d", MODEL, args.human_loop, QUALIFY_THRESHOLD)

    run_opportunity_agent(
        input_file=args.input,
        output_file=args.output,
        proposal_output_file=args.proposal_output,
        human_loop=args.human_loop,
    )


if __name__ == "__main__":
    main()