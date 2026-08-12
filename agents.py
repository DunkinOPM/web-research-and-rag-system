from langchain.agents import create_agent
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from tools import web_search
from dotenv import load_dotenv

load_dotenv()

llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0
)


def build_search_agent():
    return create_agent(
        model=llm,
        tools=[web_search]
    )


writer_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """You are an expert research writer.

Write clear, structured, factual and insightful research reports.

Use only the research evidence provided to you.
Do not invent facts or sources.
When making claims, use the provided source URLs."""
    ),
    (
        "human",
        """Write a detailed research report on the topic below.

Topic:
{topic}

Retrieved Research:
{research}

Structure the report as:

# Introduction

# Key Findings

Provide at least 3 well-explained findings.

# Conclusion

# Sources

List the URLs of the sources used.

Be detailed, factual and professional.
Ground the report in the retrieved research."""
    ),
])

writer_chain = writer_prompt | llm | StrOutputParser()


critic_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """You are a sharp and constructive research critic.

Evaluate the report for factual grounding, completeness,
source usage, clarity and relevance."""
    ),
    (
        "human",
        """Review the research report below.

Report:
{report}

Respond in this exact format:

Score: X/10

Strengths:

- ...
- ...

Areas to Improve:

- ...
- ...

One line verdict:
..."""
    ),
])

critic_chain = critic_prompt | llm | StrOutputParser()