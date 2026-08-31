import os
import sys
import uvicorn

# Ensure src module is on Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.config import BASE_DIR
from src.api import app

if __name__ == "__main__":
    print(f"Starting MSc Dissertation FastAPI server on http://0.0.0.0:8000")
    uvicorn.run("src.api:app", host="0.0.0.0", port=8000, reload=True)
