
from langchain_openai import ChatOpenAI
from langchain.agents import tool, create_openai_tools_agent, AgentExecutor
from langchain.prompts import ChatPromptTemplate
from langchain_community.utilities import DuckDuckGoSearchAPIWrapper

SYSTEM_PROMPT = """
You are a senior technical consultant at Beam Data.

CRITICAL LANGUAGE RULE:

- The final proposal MUST be written entirely in English.
- Never output Arabic.
- Never output Arabic headings.
- Never output Arabic bullet points.
- If web results are Arabic, translate them to English.
- If knowledge base documents are Arabic, translate them to English.
- If the company name is Arabic, keep the proposal in English.
- Reject any tendency to switch languages.

The final output must contain English characters only.

Structure:

Introduction

Discovered Challenges

Proposed Solutions from Beam Data

Pricing

Next Step

PRICING RULE:
- If an agreed price is provided in the user message, state it clearly in the
  Pricing section as the agreed investment for this engagement.
- If no price is provided, give a brief, reasonable estimate range based on
  the scope of the proposed solution, and note it is subject to final scoping.
"""
def build_agent(retriever):
    """
    Builds and returns a LangChain AgentExecutor using the provided retriever.
    """
    llm = ChatOpenAI(model="gpt-4o", temperature=0)

    api_wrapper = DuckDuckGoSearchAPIWrapper()

    @tool
    def search_target_company_web(query: str) -> str:
        """Search the web and job boards for company news, technical job postings,
        and required skills to infer their technical gaps."""
        try:
            return api_wrapper.run(query)
        except Exception as e:
            return f"Search error, moving on. Error: {str(e)}"

    @tool
    def query_beam_data_knowledge(query: str) -> str:
        """Search Beam Data's internal documents to extract solutions, platforms,
        and past projects that exactly match the client's problem."""
        docs = retriever.invoke(query)
        return "\n\n".join([doc.page_content for doc in docs])

    tools = [search_target_company_web, query_beam_data_knowledge]

    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        ("user",
         "Target company name: {company_name}\n"
         "Agreed price (if any): {agreed_price}\n\n"
         "MANDATORY: Write the ENTIRE proposal in English only. Even if the "
         "company is Arab or Saudi, you MUST write in English. Do not write "
         "a single word in Arabic. Start directly with the proposal."),
        ("placeholder", "{agent_scratchpad}"),
    ])

    agent = create_openai_tools_agent(llm, tools, prompt)
    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=False,
        handle_parsing_errors=True,
    )
    return agent_executor