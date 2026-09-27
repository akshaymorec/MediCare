"""FastAPI REST API for the MediCare Follow-Up Agent.

Wraps the existing LangChain agent (with its input/output guardrails alreadyattached in agent.py) behind HTTP endpoints. The CLI in main.py is unaffected and still works independently.

Run with:
    uvicorn agent.api:app --reload
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from .agent import build_agent

app = FastAPI(
    title="MediCare Follow-Up Agent API",
    description="REST API for AI-powered patient follow-up and care-coordination.",
    version="1.0.0",
)

# Build once at import time and reuse across requests, rather than rebuilding
# the LLM client + tool bindings on every call.
_agent = build_agent()

class QueryRequest(BaseModel):
    query: str = Field(
        ...,
        min_length=1,
        max_length=2000,
        description="Natural-language follow-up request, e.g. 'Analyze patient P0001'.",
    )


class QueryResponse(BaseModel):
    response: str


def _run_agent(query: str) -> str:
    try:
        result = _agent.invoke({"messages": [{"role": "user", "content": query}]})
    except Exception as exc:  # agent-level failure (LLM/API error, etc.)
        raise HTTPException(status_code=502, detail=f"Agent error: {exc}") from exc

    messages = result.get("messages", [])
    if not messages:
        raise HTTPException(status_code=502, detail="No response generated.")
    return messages[-1].content


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/agent/query", response_model=QueryResponse)
def query_agent(payload: QueryRequest) -> QueryResponse:
    """General-purpose endpoint. Routes any natural-language request through the
    full agent (input guardrail -> tool-calling -> output guardrail), so it
    covers cohort review, search, or free-form questions as well as the two
    endpoints below."""
    return QueryResponse(response=_run_agent(payload.query))

@app.get("/patients/{patient_id}/follow-up", response_model=QueryResponse)
def analyze_patient(patient_id: str) -> QueryResponse:
    """Single-patient analysis: builds the equivalent natural-language request
    and routes it through the same guardrailed agent as /agent/query."""
    query = f"Analyze patient {patient_id} and give me a prioritized follow-up action."
    return QueryResponse(response=_run_agent(query))

@app.get("/missed-appointments/follow-up", response_model=QueryResponse)
def missed_appointments_follow_up() -> QueryResponse:
    """Missed-appointment follow-up plan, via the same guardrailed agent."""
    query = (
        "Review patients who missed their last appointment and create a "
        "prioritized follow-up plan."
    )
    return QueryResponse(response=_run_agent(query))
