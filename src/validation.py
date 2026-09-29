"""Input validation functions."""


def validate_student_input(study_hours, attendance, sleep_hours, previous_score):
    """Validate and normalize prediction inputs."""
    values = {
        "study_hours": float(study_hours),
        "attendance": float(attendance),
        "sleep_hours": float(sleep_hours),
        "previous_score": float(previous_score),
    }
    if not 0 <= values["study_hours"] <= 24:
        raise ValueError("Study hours must be between 0 and 24.")
    if not 0 <= values["attendance"] <= 100:
        raise ValueError("Attendance must be between 0 and 100.")
    if not 0 <= values["sleep_hours"] <= 24:
        raise ValueError("Sleep hours must be between 0 and 24.")
    if not 0 <= values["previous_score"] <= 100:
        raise ValueError("Previous score must be between 0 and 100.")
    return values
