"""Start the API locally on port 8000."""

from pathlib import Path
import sys

project_dir = Path(__file__).resolve().parent
sys.path.insert(0, str(project_dir))
sys.path.insert(0, str(project_dir / ".vendor"))

import uvicorn

from api import app


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
