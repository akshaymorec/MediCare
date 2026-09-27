# MediCare Patient Follow-Up Agent

An AI-powered patient follow-up assistant built with **Python, LangChain, OpenAI, and FastAPI**.

The agent reviews anonymized patient records, identifies patients who may require follow-up, prioritizes them, and generates care-coordination actions for the clinical team. It is accessible both as an interactive CLI and as a REST API.

This project was developed as a prototype for the **MediCare Patient Follow-Up Agent**.

---

## 1. Problem Statement

MediCare Clinic manages chronic-disease patients across conditions such as:

- Type 2 Diabetes
- Hypertension
- Heart Failure
- COPD
- Anemia
- Chronic Kidney Disease

Patients who miss appointments or show concerning clinical indicators may not be identified early enough for proactive follow-up.

The goal of this prototype is to build an AI agent that can:

1. Review patient records.
2. Identify patients requiring attention.
3. Prioritize patients based on the available clinical information.
4. Explain the evidence behind the priority.
5. Generate appropriate care-coordination actions.
6. Handle individual patient analysis and cohort-level analysis.

The agent uses an LLM to make prioritization and action decisions rather than relying on hard-coded clinical thresholds.

---

## 2. Architecture

The application uses a straightforward LangChain tool-calling architecture, reachable through two front doors — a CLI and a REST API — that both invoke the same guardrailed agent.

```text
                    CLI (main.py)      REST API (api.py, FastAPI)
                          \                 /
                           \               /
                              build_agent()
                                    |
                                    v
                             Input Guardrail
                            /               \
                         Block              Allow
                           |                  |
                           v                  v
                        Response        LangChain Agent
                                             |
                                             v
                                        Patient Tools
                                             |
                                             v
                                      Patient Data CSV
                                             |
                                             v
                                        LLM Analysis
                                             |
                                             v
                                      Output Guardrail
                                     /        |        \
                                  SAFE      REWRITE    BLOCK
                                   |           |          |
                                   v           v          v
                                Return     Safe output  Fallback
```

### Why LangChain?

The workflow is primarily a tool-calling workflow:

```text
User Request
     ↓
LLM
     ↓
Tool Call
     ↓
Patient Data
     ↓
LLM Decision
     ↓
Final Response
```

`create_agent` (from `langchain.agents`) compiles to a LangGraph `StateGraph` internally, so a custom hand-built graph isn't needed — the middleware hooks it exposes (`before_agent`, `after_agent`) are exactly what the guardrails need.

### Why FastAPI?

The CLI is convenient for interactive testing, but a REST API is what makes the agent consumable by other services or a frontend — a required deliverable for this round. `agent.py`'s `build_agent()` is reused unchanged by both `main.py` and `api.py`, so there's a single agent implementation with two access layers, not two separate agents to keep in sync.

---

## 3. Project Structure

```text
MediCare/
│
├── agent/
│   ├── __init__.py
│   ├── agent.py
│   ├── api.py
│   ├── config.py
│   ├── data.py
│   ├── guardrail.py
│   ├── main.py
│   ├── prompts.py
│   └── tools.py
│
├── patient_data.csv
├── .env
├── .env.example
├── .gitignore
├── requirements.txt
└── README.md
```

### File Responsibilities

#### `agent/main.py`

Interactive CLI entry point.

It:

- starts the interactive CLI;
- accepts natural-language requests;
- invokes the MediCare agent;
- displays the response.

Run using:

```bash
python -m agent.main
```

#### `agent/api.py`

FastAPI REST API entry point. Wraps the same agent from `agent.py` behind HTTP endpoints, so it shares the identical guardrail and tool-calling behavior as the CLI. See [Section 7](#7-running-the-application) for endpoints and run instructions.

#### `agent/agent.py`

Creates and configures the LangChain agent.

Responsibilities include:

- initializing the OpenAI chat model;
- registering the patient-data tools;
- connecting the guardrails;
- returning the agent for the CLI and API to invoke.

#### `agent/config.py`

Centralizes environment/config loading: API keys, model names, and the resolved path to `patient_data.csv`. Validates that required config is present before the agent runs.

#### `agent/tools.py`

Contains the tools available to the LLM.

Current tools include:

- Get a specific patient.
- Find patients who missed their last appointment.
- Retrieve a patient cohort.
- Search patients.

The tools retrieve information from the patient dataset rather than making clinical decisions.

#### `agent/data.py`

Handles loading and accessing `patient_data.csv`. Keeps data-access logic separate from the agent and tools, and normalizes records (date parsing, NaN-safe field extraction) before they reach the LLM.

#### `agent/prompts.py`

Contains the system instructions used by the MediCare agent.

The prompt defines:

- the agent's role;
- how patient information should be evaluated;
- how priorities should be determined;
- how evidence should be presented;
- allowed care-coordination actions;
- prohibited medical treatment recommendations.

#### `agent/guardrail.py`

Contains the input and output guardrails.

### Input Guardrail

Two layers, both fail **closed** (block) if the check itself errors, rather than silently letting a request through:

#### Layer 1: Regex

Deterministic checks such as:

- prompt injection attempts;
- attempts to expose system instructions;
- obvious API-key or secret patterns;
- invalid or excessively large requests.

#### Layer 2: NLP / LLM Validation

The request is semantically checked to determine whether it is relevant to the MediCare patient follow-up use case.

For example:

```text
Analyze patient P0001.
```

is allowed.

An unrelated request such as:

```text
Write me a Python game.
```

should be rejected.

### Output Guardrail

The output guardrail validates the final agent response against the patient data actually retrieved by the tools, and follows one of three paths:

```text
APPROVE
  ↓
Return response unchanged

REWRITE
  ↓
Convert unsafe wording into
safe care-coordination language
  ↓
Return corrected response

BLOCK
  ↓
Return safe fallback message
```

It checks that the response:

- is grounded in retrieved patient information (no invented facts);
- does not expose system instructions or secrets;
- stays within the care-coordination scope;
- does not prescribe treatment or change medication dosage;
- does not provide definitive diagnoses.

Example — REWRITE:

```text
Consider medication adjustment.
```
becomes:
```text
Coordinate medication review with the treating clinician.
```

Example — BLOCK (not fixable by rephrasing):

```text
Increase the patient's medication dosage.
```

Like the input guardrail, this fails **closed**: if the reviewer LLM call itself errors, the response is blocked rather than returned unreviewed.

---

## 4. Dataset

The project uses:

```text
patient_data.csv
```

The dataset contains anonymized patient records.

Each record includes information such as:

- `patient_id`
- `patient_name`
- `age`
- `gender`
- `diagnosis`
- `current_medication`
- `lab_test`
- `lab_value`
- `lab_unit`
- `last_visit_date`
- `next_scheduled_visit`
- `days_since_last_visit`
- `missed_last_appointment`
- `vitals_bp_systolic`
- `vitals_bp_diastolic`
- `vitals_heart_rate`
- `vitals_spo2`
- `notes`

The agent uses these fields as evidence when analyzing patients.

---

## 5. Requirements

Python 3.10+ is recommended.

Install the dependencies:

```bash
pip install -r requirements.txt
```

Key dependencies: `langchain`, `langchain-openai`, `pandas`, `python-dotenv`, `fastapi`, `uvicorn[standard]`, `pydantic`.

---

## 6. Environment Configuration

Create a `.env` file in the project root.

Example:

```env
OPENAI_API_KEY=your_openai_api_key
OPENAI_MODEL=gpt-4o-mini
```

Do not commit `.env` to source control.

The `.gitignore` file should contain:

```text
.env
.venv/
__pycache__/
*.pyc
```

---

## 7. Running the Application

The agent can be reached two ways — both call the same `build_agent()` and share the same guardrails.

### Option A — CLI

```bash
python -m agent.main
```

```text
MediCare Follow-Up Agent
Type 'exit' or 'quit' or 'q' to stop.

You:
```

### Option B — REST API (FastAPI)

```bash
uvicorn agent.api:app --reload
```

Defaults to `http://127.0.0.1:8000`. Interactive docs (Swagger UI) at `http://127.0.0.1:8000/docs`.

| Method | Endpoint | Purpose |
|---|---|---|
| GET | `/health` | Liveness check |
| POST | `/agent/query` | General-purpose: any natural-language request (body: `{"query": "..."}`) |
| GET | `/patients/{patient_id}/follow-up` | Single-patient analysis |
| GET | `/missed-appointments/follow-up` | Missed-appointment follow-up plan |

Example:

```bash
curl -X POST http://127.0.0.1:8000/agent/query \
  -H "Content-Type: application/json" \
  -d '{"query": "Analyze patient P0001 and give me a prioritized follow-up action."}'
```

---

## 8. Example Requests

### Single Patient Analysis

```text
Analyze patient P0001 and give me a prioritized follow-up action.
```

Example response:

```text
Priority: High

Patient ID: P0001

Key evidence:
Type 2 Diabetes with HbA1c of 9.9%, elevated blood pressure,
and a prolonged gap since the last visit.

Suggested follow-up action:
Coordinate clinician review to assess diabetes management and blood
pressure control, and review medication adherence with the care team.
```

### Missed Appointment Follow-Up

```text
Review patients who missed their last appointment and
create a prioritized follow-up plan.
```

### Cohort Analysis

```text
Review the patient cohort and identify the highest-priority
patients for care coordination.
```

The agent returns prioritized patients along with:

1. Priority
2. Patient ID
3. Key evidence
4. Suggested follow-up action

### Patient Search

```text
Find patients with Heart Failure.
```

or:

```text
Find patients who missed their last appointment.
```

All of the above work identically through the CLI or through `POST /agent/query`.

---

## 9. Agent Decision Making

The application intentionally does not hard-code clinical risk thresholds inside the Python code.

Instead, the available patient information is provided to the LLM through tools, and the LLM determines:

- which findings are important;
- which patients require greater attention;
- the priority level;
- the appropriate care-coordination action.

The Python application is responsible for:

- retrieving data;
- providing tools;
- orchestrating the agent;
- enforcing guardrails;
- exposing the agent via CLI and REST API.

This keeps clinical prioritization decisions within the LLM-driven agent rather than embedding arbitrary thresholds in the application.

---

## 10. Safety Boundary

This is a prototype for **care coordination**, not an autonomous clinical decision-making system.

The agent must not:

- prescribe medication;
- start or stop medication;
- increase or decrease dosage;
- recommend a specific treatment;
- provide definitive diagnosis;
- provide specific medical procedures;
- replace clinician judgment.

When medication-related follow-up is relevant, the agent should use language such as:

```text
Review medication adherence with the care team.
```

or:

```text
Coordinate medication reconciliation with the treating clinician.
```

rather than recommending a medication change. This is enforced at two independent layers: the system prompt (`prompts.py`) instructs the model directly, and the output guardrail (`guardrail.py`) independently re-checks the final response against the same rule before it's returned.

All recommendations should be treated as suggestions for review by qualified healthcare professionals.

---

## 11. Example Agent Flow

For a request such as:

```text
Analyze patient P0001.
```

the process is:

```text
Request (via CLI or POST /agent/query)
     |
     v
Regex Input Guardrail
     |
     v
Semantic Input Guardrail
     |
     v
LangChain Agent
     |
     v
get_patient(P0001)
     |
     v
Patient Record
     |
     v
LLM Analysis
     |
     v
Prioritization + Follow-up Action
     |
     v
Output Guardrail (approve / rewrite / block)
     |
     v
Final Response
```

---

## 12. Design Decisions

### Two Entry Points, One Agent

```bash
python -m agent.main        # interactive CLI
uvicorn agent.api:app        # REST API
uvicorn agent.api:app --reload --host 0.0.0.0 --port 8000
```

Both call the same `build_agent()` from `agent.py`, so there is one agent implementation, not two implementations to keep behaviorally in sync. The API was added specifically to satisfy this round's requirement for a REST interface (FastAPI) alongside modular code.

### Separation of Responsibilities

```text
Config
 ↓
Data
 ↓
Tools
 ↓
Agent
 ↓
Guardrails
 ↓
CLI / API entry points
```

This makes individual components easier to test and modify independently.

### Guardrails

The application uses both deterministic and LLM-based validation.

Deterministic checks are useful for high-confidence patterns such as prompt injection attempts.

LLM-based validation is useful for semantic checks that are difficult to express reliably using regular expressions alone.

---

## 15. Running Example

### CLI

```bash
python -m agent.main
```
```text
MediCare Follow-Up Agent
Type 'exit' or 'quit' or 'q' to stop.

You: Analyze patient P0001 and give me a prioritized follow-up action.

Agent:
...

You: exit
```

### API

```bash
uvicorn agent.api:app --reload
```
```bash
curl -X POST http://127.0.0.1:8000/agent/query \
  -H "Content-Type: application/json" \
  -d '{"query": "Analyze patient P0001 and give me a prioritized follow-up action."}'
```

---

## 16. Project Goal

The main goal of this project is to demonstrate how an LLM-powered agent can combine:

```text
Natural Language
       +
Patient Data
       +
Tool Calling
       +
LLM Reasoning
       +
Input Guardrails
       +
Output Guardrails
       +
Modular Code
       +
REST API (FastAPI)
       =
Care Coordination Assistant
```

The prototype demonstrates agentic AI concepts while maintaining a clear boundary between AI-assisted care coordination and clinical decision-making.
