"""HTTP interface for the churn project, built one endpoint at a time."""

from fastapi import FastAPI

app = FastAPI(title="Telco Churn API")


@app.get("/health")
def health() -> dict[str, str]:
    """Confirm that the web process is running and can answer requests."""
    return {"status": "ok"}
