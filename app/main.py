# ==========================================================\n# Handwritten Mathematical Formula Recognition System
# Module: app.py
# Description: FastAPI Application Server providing REST APIs and Web Interface.
# ==========================================================\n
# ==========================================================

import uuid
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from app.api.routes import router as api_router
from app.models.predict import init_predictor, get_predictor
from fastapi.middleware.cors import CORSMiddleware

# Directory configurations
BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent
STATIC_DIR = BASE_DIR / "static"
TEMPLATES_DIR = BASE_DIR / "templates"


@asynccontextmanager
async def lifespan(app: FastAPI):
    """FastAPI Lifespan Context Manager: Load PyTorch model ONCE during startup."""
    print("\n[STARTUP] [FastAPI Server] Starting Handwritten Formula Analyzer...")
    try:
        init_predictor(
            model_path=str(PROJECT_ROOT / "weights" / "best_model.pth"),
            vocab_path=str(PROJECT_ROOT / "weights" / "vocab.pkl")
        )
        print("[SUCCESS] [FastAPI Server] Model & Vocabulary successfully preloaded into memory!\n")
    except Exception as e:
        print(f"[ERROR] [FastAPI Server] Error preloading model: {e}")
    yield
    print("[SHUTDOWN] [FastAPI Server] Shutting down application server.")


app = FastAPI(
    title="Handwritten Mathematical Formula Recognition System",
    description="Modern AI-powered LaTeX formula recognition system built with PyTorch & FastAPI.",
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
app.include_router(api_router)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="127.0.0.1", port=8000, reload=True)
