#Tools Creation 
from langchain.tools import tool
from .data import compact_record, load_patients, patient_record

@tool
def get_patient(patient_id: str) -> dict:
    """Get the complete clinical snapshot for one patient by patient ID."""
    df = load_patients()
    matches = df[df["patient_id"].str.upper() == patient_id.strip().upper()]
    if matches.empty:
        return {"error": f"Patient {patient_id} was not found."}
    return patient_record(matches.iloc[0])

@tool
def get_missed_appointments() -> list[dict]:
    """Return patients who missed their most recent appointment."""
    df = load_patients()
    missed = df[df["missed_last_appointment"].str.lower() == "yes"]
    return [compact_record(row) for _, row in missed.iterrows()]

@tool
def get_patient_cohort(limit: int = 100) -> list[dict]:
    """Return a compact view of the patient cohort for proactive review."""
    limit = max(1, min(int(limit), 100))
    df = load_patients().head(limit)
    return [compact_record(row) for _, row in df.iterrows()]

@tool
def search_patients(query: str, limit: int = 10) -> list[dict]:
    """Search records by patient ID, diagnosis, medication, lab, or notes."""
    df = load_patients()
    query = query.strip().lower()
    if not query:
        return []

    searchable = df.astype(str).apply(lambda col: col.str.lower())
    mask = searchable.apply(
        lambda row: row.str.contains(query, regex=False).any(), axis=1
    )
    results = df[mask].head(max(1, min(int(limit), 25)))
    return [compact_record(row) for _, row in results.iterrows()]

# tool registry
TOOLS = [
    get_patient,
    get_missed_appointments,
    get_patient_cohort,
    search_patients,
]