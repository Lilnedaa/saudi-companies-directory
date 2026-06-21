import os
import csv
import json
from dotenv import load_dotenv
from openai import OpenAI


load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


class CampaignAgent:
    """
    Task 2: Campaign Agent

    Input:
    - A list of company dicts (e.g. selected rows from Lead Scoring results)

    Process:
    - Generates personalized campaign emails using OpenAI API.

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
                            "You are a professional B2B sales campaign assistant. "
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

        prompt = f"""
Generate a personalized B2B cold email for this company.

Company data:
Company name: {company_name}
Arabic name: {arabic_name}
Sector: {sector}
Sub-sector: {sub_sector}
City: {city}
Country: {country}
Employees: {employees}
Founded year: {founded_year}
Website: {website}
Email: {email}
Phone: {phone}
LinkedIn: {linkedin_url}
Description: {description}
Is startup: {is_startup}
Is listed: {is_listed}
Tags: {tags}

BeamData offers:
- AI automation
- Data analytics
- Workflow automation
- Lead qualification automation
- Proposal automation
- CRM automation

Rules:
- Make the email professional and short.
- Do not invent fake facts.
- Personalize using sector, sub-sector, description, and tags.
- Sender is BeamData Team.
- Goal is to schedule a short meeting.
- Return JSON only.

Return JSON in this exact format:
{{
  "email_subject": "subject here",
  "email_body": "email body here",
  "campaign_goal": "goal here",
  "suggested_service": "service here"
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
            "email_subject": data.get("email_subject", f"AI Automation Opportunity for {company_name}"),
            "email_body": data.get("email_body", ""),
            "campaign_goal": data.get("campaign_goal", "Schedule a short meeting"),
            "suggested_service": data.get("suggested_service", "AI automation and data analytics")
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

        subject = f"AI Automation Opportunity for {company_name}"

        body = f"""
Dear {company_name} Team,

We noticed that your organization operates in the {sector} sector, specifically in {sub_sector}.

At BeamData, we provide AI automation and data-driven solutions that help organizations improve workflows, lead qualification, CRM automation, campaign automation, and proposal generation.

Based on your company profile and focus areas such as {tags}, we believe there may be an opportunity to support your digital and business goals.

We would be happy to schedule a short meeting to explore how our solution can help.

Best regards,
BeamData Team
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
            "suggested_service": "AI automation and data analytics"
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
                "suggested_service"
            ])

            for email in emails:
                writer.writerow([
                    email["company_name"],
                    email["arabic_name"],
                    email["sector"],
                    email["sub_sector"],
                    email["city"],
                    email["country"],
                    email["website"],
                    email["email"],
                    email["phone"],
                    email["linkedin_url"],
                    email["employees"],
                    email["founded_year"],
                    email["description"],
                    email["tags"],
                    email["email_subject"],
                    email["email_body"],
                    email["campaign_goal"],
                    email["suggested_service"]
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