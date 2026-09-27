"""LangChain agent definition."""

from langchain.agents import create_agent
from langchain_openai import ChatOpenAI
from .config import OPENAI_API_KEY, OPENAI_MODEL, validate_config
from .guardrail import InputGuardrailMiddleware, OutputGuardrailMiddleware
from .prompts import SYSTEM_PROMPT
from .tools import TOOLS


def build_agent():
    """Build the MediCare agent using LangChain's standard tool-calling agent."""
    validate_config()

    model = ChatOpenAI(
        model=OPENAI_MODEL,
        api_key=OPENAI_API_KEY,
        temperature=0,
    )

    return create_agent(
        model=model,
        tools=TOOLS,
        system_prompt=SYSTEM_PROMPT,
        middleware=[
            InputGuardrailMiddleware(),
            OutputGuardrailMiddleware(),
        ],
    )