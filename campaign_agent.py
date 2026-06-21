import os
import csv
import json
from dotenv import load_dotenv
from openai import OpenAI


load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# ── BeamData Knowledge Base ────────────────────────────────────────────────────

BEAMDATA_PAST_PROJECTS = [
    {
        "title": "RAG Chatbot & Knowledge Base",
        "description": "AI-powered chatbot using Retrieval-Augmented Generation (RAG) that answers questions from company documents, policies, and knowledge bases. Reduced support tickets by 60%.",
        "sectors": ["Technology", "E-Commerce", "Customer Service", "Telecom", "Banking", "Insurance"],
    },
    {
        "title": "Document Verification & Extraction System",
        "description": "Automated document verification using AI/ML to check authenticity, extract structured data from unstructured documents, and reduce manual review time by 80%.",
        "sectors": ["Banking", "FinTech", "Government", "Healthcare", "Insurance", "Real Estate"],
    },
    {
        "title": "Healthcare AI Clinical Assistant",
        "description": "Clinical decision support system using AI to assist doctors with diagnosis, drug interaction checks, patient record analysis, and medical coding automation.",
        "sectors": ["Healthcare", "Pharmaceuticals", "Biotech", "Hospitals"],
    },
    {
        "title": "Dynamic Pricing Engine",
        "description": "Real-time pricing optimization using machine learning to maximize revenue based on demand signals, competitor pricing, and inventory levels. Increased revenue by 18%.",
        "sectors": ["E-Commerce", "Retail", "Travel", "Hospitality", "FMCG"],
    },
    {
        "title": "Marketing Attribution & ROI Model",
        "description": "Multi-touch attribution model that identifies which marketing channels drive the most conversions, enabling smarter ad spend decisions and reducing CAC by 25%.",
        "sectors": ["E-Commerce", "Marketing", "Media", "Retail", "Telecom"],
    },
    {
        "title": "Personalized Recommender System",
        "description": "Product and content recommendation engine using collaborative filtering and deep learning. Increased cross-sell revenue by 30% for a major e-commerce client.",
        "sectors": ["E-Commerce", "Media", "Retail", "Streaming", "Publishing"],
    },
    {
        "title": "AI-Powered Loan Approval & Credit Scoring",
        "description": "Automated loan approval using AI risk scoring, fraud detection, and alternative credit assessment. Reduced approval time from 5 days to 3 minutes.",
        "sectors": ["FinTech", "Banking", "Financial Services", "Insurance"],
    },
    {
        "title": "Predictive Maintenance System",
        "description": "IoT + AI system that predicts equipment failures before they happen using sensor data and anomaly detection, reducing downtime by 40% and maintenance costs by 35%.",
        "sectors": ["Manufacturing", "Energy", "Oil & Gas", "Industrial", "Utilities", "Mining"],
    },
    {
        "title": "Customer Segmentation & Churn Prediction",
        "description": "AI-powered customer segmentation grouping customers by behavior, lifetime value, and churn risk. Enabled targeted retention campaigns that reduced churn by 22%.",
        "sectors": ["Telecom", "E-Commerce", "Banking", "Retail", "Insurance"],
    },
    {
        "title": "Sales & Demand Forecasting",
        "description": "Machine learning model forecasting sales demand by product, region, and season with 92% accuracy. Helped clients optimize inventory and reduce stockouts by 45%.",
        "sectors": ["Retail", "FMCG", "Manufacturing", "Supply Chain", "Wholesale"],
    },
    {
        "title": "Real Estate AI Valuation Platform",
        "description": "AI-powered property valuation, market trend analysis, and investment opportunity scoring. Automated AVM (Automated Valuation Model) for thousands of properties in real time.",
        "sectors": ["Real Estate", "PropTech", "Construction", "Banking"],
    },
    {
        "title": "Computer Vision Quality Control",
        "description": "Automated visual inspection system using computer vision to detect product defects on production lines, achieving 99.2% defect detection accuracy and replacing manual inspection.",
        "sectors": ["Manufacturing", "Food & Beverage", "Industrial", "Pharmaceuticals", "Electronics"],
    },
    {
        "title": "Fleet & Logistics Route Optimization",
        "description": "AI-based route optimization and fleet distribution management system that reduced delivery times by 28% and fuel costs by 20% for a large logistics operator.",
        "sectors": ["Logistics", "Transportation", "Supply Chain", "E-Commerce", "Retail"],
    },
    {
        "title": "Data Warehouse & ETL Automation",
        "description": "Centralized data warehouse with automated ETL pipelines that consolidate data from 15+ sources, enabling a single source of truth and cutting data engineering effort by 70%.",
        "sectors": ["All Industries", "Technology", "Banking", "Retail", "Telecom"],
    },
    {
        "title": "Business Intelligence Dashboards",
        "description": "Real-time interactive BI dashboards giving executives and operations teams live visibility into KPIs, sales, logistics, and customer data.",
        "sectors": ["All Industries", "Retail", "Banking", "Manufacturing", "Telecom"],
    },
    {
        "title": "Web Scraping & Competitive Intelligence",
        "description": "Automated data collection pipelines for market intelligence, competitor price tracking, and industry research, delivering daily structured datasets.",
        "sectors": ["E-Commerce", "Research", "Market Intelligence", "Retail", "Finance"],
    },
    {
        "title": "Digital Process Automation (DPA/RPA)",
        "description": "End-to-end digital automation of paper-based and manual workflows using AI + RPA, cutting operational costs by 50% and processing time by 80%.",
        "sectors": ["Government", "Banking", "Insurance", "HR", "Healthcare", "Telecom"],
    },
    {
        "title": "AI Sales CRM Platform",
        "description": "AI-powered CRM with lead scoring, opportunity qualification, campaign automation, and proposal generation. Increased sales team productivity by 3x.",
        "sectors": ["Sales", "B2B", "Technology", "Real Estate", "Financial Services"],
    },
    {
        "title": "Smart Agriculture & Crop Intelligence",
        "description": "AI/IoT system for crop health monitoring, yield prediction, and water/fertilizer optimization. Increased crop yield by 35% while reducing resource waste.",
        "sectors": ["Agriculture", "AgriTech", "Food & Beverage"],
    },
    {
        "title": "Clinical LLM for Medical Documentation",
        "description": "Large language model fine-tuned on clinical data for medical documentation automation, clinical coding, and research summarization.",
        "sectors": ["Healthcare", "Pharmaceuticals", "Biotech", "Hospitals"],
    },
    {
        "title": "AI Hub Enterprise Platform",
        "description": "Comprehensive enterprise AI platform with AI Assistant, Knowledge Hub (RAG-based), and Agentic Workflow automation for end-to-end business process intelligence.",
        "sectors": ["Enterprise", "Technology", "Banking", "Telecom", "Government"],
    },
    {
        "title": "AI-Powered Learning Management System (LMS)",
        "description": "AI-enhanced LMS with personalized learning paths, content recommendations, performance analytics, and automated skills gap detection.",
        "sectors": ["Education", "HR", "Training", "Government", "Corporate"],
    },
]

BEAMDATA_SERVICES = """
BeamData Core Services:
1. Data & AI Strategy — Define your AI roadmap, identify high-ROI use cases, and plan your data transformation.
2. Proof of Concepts (PoC) — Rapidly validate AI/data solutions with working prototypes before committing to full build.
3. Deployment & Integration — Production-ready model deployment, API integration, and system connectivity.
4. Operations & Monitoring — Ongoing model performance monitoring, retraining, and continuous optimization.
5. AI Governance & Compliance — Ensure AI systems are transparent, auditable, and compliant with regulations.

BeamData AI Hub Products:
- AI Assistant: Company-specific intelligent chatbot powered by your documents and knowledge.
- Knowledge Hub: RAG-based document intelligence — search and retrieve answers from any internal content.
- Agentic Workflow: Autonomous AI agents that handle complex multi-step business processes end-to-end.
"""


def _get_relevant_projects(sector: str, sub_sector: str, description: str, tags: str, top_k: int = 3) -> list:
    """Returns the most relevant past projects for a company based on sector matching."""
    combined_text = f"{sector} {sub_sector} {description} {tags}".lower()

    scored = []
    for project in BEAMDATA_PAST_PROJECTS:
        score = 0
        for s in project["sectors"]:
            if s.lower() in combined_text or any(word in combined_text for word in s.lower().split()):
                score += 2
            if "All Industries" in project["sectors"]:
                score += 1
        scored.append((score, project))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [p for _, p in scored[:top_k]]


class CampaignAgent:
    """
    Task 2: Campaign Agent

    Input:
    - A list of company dicts (e.g. selected rows from Lead Scoring results)

    Process:
    - Generates personalized campaign emails using OpenAI API.
    - References relevant BeamData past projects and services matching the company's sector.

    Output:
    - campaign_emails.csv (optional) or just a list of dicts returned to the app
    """

    def __init__(self):
        if not OPENAI_API_KEY:
            raise ValueError("Missing OPENAI_API_KEY. Add it inside .env file.")

        self.client = OpenAI(api_key=OPENAI_API_KEY)

    def normalize_company(self, company):
        """
        Maps the Saudi Companies Directory field names (name, city_clean, ...)
        to the field names this agent expects (company_name, city, ...).
        Works whether the input already uses the old names or the new ones.
        """
        return {
            "company_name": company.get("company_name") or company.get("name", ""),
            "arabic_name": company.get("arabic_name", ""),
            "sector": company.get("sector", ""),
            "sub_sector": company.get("sub_sector", ""),
            "city": company.get("city") or company.get("city_clean", ""),
            "country": company.get("country", "Saudi Arabia"),
            "website": company.get("website", ""),
            "email": company.get("email", ""),
            "phone": company.get("phone", ""),
            "linkedin_url": company.get("linkedin_url", ""),
            "employees": company.get("employees", ""),
            "founded_year": company.get("founded_year", ""),
            "description": company.get("description", ""),
            "is_startup": company.get("is_startup", ""),
            "is_listed": company.get("is_listed", ""),
            "tags": company.get("tags", ""),
        }

    def read_companies(self, input_file="saudi_companies_500.csv", limit=20):
        companies = []

        with open(input_file, mode="r", encoding="utf-8-sig") as file:
            reader = csv.DictReader(file)

            for row in reader:
                company_name = row.get("company_name", "").strip()

                if company_name:
                    companies.append(row)

                if limit and len(companies) >= limit:
                    break

        return companies

    def ask_openai_json(self, prompt):
        try:
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                temperature=0.4,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a professional B2B sales campaign assistant for BeamData, "
                            "an AI and data solutions company. "
                            "Return valid JSON only. No markdown. No explanation."
                        )
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ]
            )

            text = response.choices[0].message.content.strip()
            text = text.replace("```json", "").replace("```", "").strip()

            return json.loads(text)

        except Exception:
            return {}

    def generate_email_for_company(self, company):
        company_name = company.get("company_name", "")
        arabic_name = company.get("arabic_name", "")
        sector = company.get("sector", "")
        sub_sector = company.get("sub_sector", "")
        city = company.get("city", "")
        country = company.get("country", "")
        website = company.get("website", "")
        email = company.get("email", "")
        phone = company.get("phone", "")
        linkedin_url = company.get("linkedin_url", "")
        employees = company.get("employees", "")
        founded_year = company.get("founded_year", "")
        description = company.get("description", "")
        is_startup = company.get("is_startup", "")
        is_listed = company.get("is_listed", "")
        tags = company.get("tags", "")

        relevant_projects = _get_relevant_projects(sector, sub_sector, description, tags)
        projects_text = "\n".join(
            f"  - {p['title']}: {p['description']}"
            for p in relevant_projects
        )

        prompt = f"""
Generate a highly personalized B2B cold email for this company from BeamData.

Company data:
  Company name: {company_name}
  Arabic name: {arabic_name}
  Sector: {sector}
  Sub-sector: {sub_sector}
  City: {city}, {country}
  Employees: {employees}
  Founded: {founded_year}
  Website: {website}
  Description: {description}
  Is startup: {is_startup}
  Is listed: {is_listed}
  Tags: {tags}

BeamData's most relevant PAST PROJECTS for this company:
{projects_text}

BeamData Services:
{BEAMDATA_SERVICES}

Instructions:
- Write in English. Be professional, concise, and warm.
- The email MUST reference 1-2 of the past projects listed above that are most relevant to this specific company's sector and needs. Mention them naturally (e.g., "We recently helped a similar company in the {sector} space with...").
- Identify what this company likely NEEDS based on their sector, description, and tags. Address that specific pain point.
- Suggest 1-2 BeamData services that would best solve their problem.
- Keep the email under 200 words. Goal: schedule a short meeting.
- Do NOT invent fake facts. Do NOT use generic filler.
- Sender: BeamData Team | beamdata.ai

Return JSON only in this exact format:
{{
  "email_subject": "subject here",
  "email_body": "email body here",
  "campaign_goal": "goal here",
  "suggested_service": "service here",
  "matched_projects": ["project title 1", "project title 2"]
}}
"""

        data = self.ask_openai_json(prompt)

        if not data:
            return self.generate_fallback_email(company)

        return {
            "company_name": company_name,
            "arabic_name": arabic_name,
            "sector": sector,
            "sub_sector": sub_sector,
            "city": city,
            "country": country,
            "website": website,
            "email": email,
            "phone": phone,
            "linkedin_url": linkedin_url,
            "employees": employees,
            "founded_year": founded_year,
            "description": description,
            "tags": tags,
            "email_subject": data.get("email_subject", f"AI Solutions for {company_name}"),
            "email_body": data.get("email_body", ""),
            "campaign_goal": data.get("campaign_goal", "Schedule a short meeting"),
            "suggested_service": data.get("suggested_service", "AI automation and data analytics"),
            "matched_projects": ", ".join(data.get("matched_projects", [])),
        }

    def generate_fallback_email(self, company):
        company_name = company.get("company_name", "")
        arabic_name = company.get("arabic_name", "")
        sector = company.get("sector", "")
        sub_sector = company.get("sub_sector", "")
        city = company.get("city", "")
        country = company.get("country", "")
        website = company.get("website", "")
        email = company.get("email", "")
        phone = company.get("phone", "")
        linkedin_url = company.get("linkedin_url", "")
        employees = company.get("employees", "")
        founded_year = company.get("founded_year", "")
        description = company.get("description", "")
        tags = company.get("tags", "")

        relevant_projects = _get_relevant_projects(sector, sub_sector, description, tags, top_k=1)
        project_mention = ""
        if relevant_projects:
            p = relevant_projects[0]
            project_mention = f"\nWe recently delivered a '{p['title']}' for a client in a similar space — {p['description'][:100]}...\n"

        subject = f"AI Solutions for {company_name} — BeamData"

        body = f"""Dear {company_name} Team,

We've been following the growth of organizations in the {sector} sector in Saudi Arabia, and we believe BeamData can add real value to your operations.
{project_mention}
At BeamData, we specialize in AI automation, data analytics, and intelligent workflow solutions — purpose-built for companies like yours.

We'd love to schedule a quick 20-minute call to explore how we can support your goals.

Best regards,
BeamData Team
www.beamdata.ai
"""

        return {
            "company_name": company_name,
            "arabic_name": arabic_name,
            "sector": sector,
            "sub_sector": sub_sector,
            "city": city,
            "country": country,
            "website": website,
            "email": email,
            "phone": phone,
            "linkedin_url": linkedin_url,
            "employees": employees,
            "founded_year": founded_year,
            "description": description,
            "tags": tags,
            "email_subject": subject,
            "email_body": body.strip(),
            "campaign_goal": "Schedule a short meeting",
            "suggested_service": "AI automation and data analytics",
            "matched_projects": "",
        }

    def save_campaign_emails(self, emails, output_file="campaign_emails.csv"):
        with open(output_file, mode="w", newline="", encoding="utf-8-sig") as file:
            writer = csv.writer(file)

            writer.writerow([
                "company_name",
                "arabic_name",
                "sector",
                "sub_sector",
                "city",
                "country",
                "website",
                "email",
                "phone",
                "linkedin_url",
                "employees",
                "founded_year",
                "description",
                "tags",
                "email_subject",
                "email_body",
                "campaign_goal",
                "suggested_service",
                "matched_projects",
            ])

            for email in emails:
                writer.writerow([
                    email.get("company_name", ""),
                    email.get("arabic_name", ""),
                    email.get("sector", ""),
                    email.get("sub_sector", ""),
                    email.get("city", ""),
                    email.get("country", ""),
                    email.get("website", ""),
                    email.get("email", ""),
                    email.get("phone", ""),
                    email.get("linkedin_url", ""),
                    email.get("employees", ""),
                    email.get("founded_year", ""),
                    email.get("description", ""),
                    email.get("tags", ""),
                    email.get("email_subject", ""),
                    email.get("email_body", ""),
                    email.get("campaign_goal", ""),
                    email.get("suggested_service", ""),
                    email.get("matched_projects", ""),
                ])

        return output_file

    def run(self, input_file="saudi_companies_500.csv", limit=20):
        companies = self.read_companies(input_file=input_file, limit=limit)

        campaign_emails = []

        for company in companies:
            email = self.generate_email_for_company(company)
            campaign_emails.append(email)

        output_file = self.save_campaign_emails(campaign_emails)

        return {
            "total_companies": len(campaign_emails),
            "output_file": output_file
        }

    def run_on_companies(self, companies, progress_callback=None):
        """
        Generates campaign emails for an in-memory list of company dicts
        (e.g. rows the user picked manually from Lead Scoring results
        in the Streamlit app). Does NOT write to disk — returns the list
        so the app can display it and let the user export it.
        """
        campaign_emails = []

        for i, raw_company in enumerate(companies):
            company = self.normalize_company(raw_company)

            if progress_callback:
                progress_callback(i + 1, len(companies), company.get("company_name", ""))

            email = self.generate_email_for_company(company)
            campaign_emails.append(email)

        return campaign_emails
