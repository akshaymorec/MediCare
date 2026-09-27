from functools import lru_cache
import pandas as pd
from .config import DATA_PATH

@lru_cache(maxsize=1)
def load_patients() -> pd.DataFrame:
    """Load and normalize the patient snapshot dataset."""
    df = pd.read_csv(DATA_PATH)
    for column in ("last_visit_date", "next_scheduled_visit"):
        df[column] = pd.to_datetime(df[column], errors="coerce")
    return df


def patient_record(row: pd.Series) -> dict:
    record = row.to_dict()
    for key in ("last_visit_date", "next_scheduled_visit"):
        value = record.get(key)
        record[key] = value.strftime("%Y-%m-%d") if pd.notna(value) else None
    return record


def _safe_int(row: pd.Series, key: str):
    value = row[key]
    return int(value) if pd.notna(value) else None


def compact_record(row: pd.Series) -> dict:
    """Return the fields most useful for cohort-level LLM review.

    Uses bracket indexing (not attribute access) so a column name can never
    collide with a pandas Series method/attribute (e.g. 'count', 'size', 'T'),
    and guards each numeric field against missing (NaN) values.
    """
    return {
        "patient_id": row["patient_id"],
        "age": _safe_int(row, "age"),
        "diagnosis": row["diagnosis"],
        "lab": f"{row['lab_test']}: {row['lab_value']} {row['lab_unit']}",
        "bp": f"{row['vitals_bp_systolic']}/{row['vitals_bp_diastolic']} mmHg",
        "heart_rate": _safe_int(row, "vitals_heart_rate"),
        "spo2": _safe_int(row, "vitals_spo2"),
        "days_since_last_visit": _safe_int(row, "days_since_last_visit"),
        "missed_last_appointment": row["missed_last_appointment"],
        "notes": row["notes"],
    }
