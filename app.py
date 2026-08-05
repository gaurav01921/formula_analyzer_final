# ==========================================================\n# Handwritten Mathematical Formula Recognition System
# Module: app.py
# Description: FastAPI Application Server providing REST APIs and Web Interface.
# ==========================================================\n
import os
import uuid
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, UploadFile, Request, HTTPException, Form
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from predict import init_predictor, predict, get_predictor, predict_multiline

# Directory configurations
BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"
UPLOADS_DIR = STATIC_DIR / "uploads"
TEMPLATES_DIR = BASE_DIR / "templates"

# Create directories if they do not exist
UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
(STATIC_DIR / "css").mkdir(parents=True, exist_ok=True)
(STATIC_DIR / "js").mkdir(parents=True, exist_ok=True)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """FastAPI Lifespan Context Manager: Load PyTorch model ONCE during startup."""
    print("\n[STARTUP] [FastAPI Server] Starting Handwritten Formula Analyzer...")
    try:
        init_predictor(
            model_path=str(BASE_DIR / "weights" / "best_model.pth"),
            vocab_path=str(BASE_DIR / "weights" / "vocab.pkl")
        )
        print("[SUCCESS] [FastAPI Server] Model & Vocabulary successfully preloaded into memory!\n")
    except Exception as e:
        print(f"[ERROR] [FastAPI Server] Error preloading model: {e}")
    yield
    print("[SHUTDOWN] [FastAPI Server] Shutting down application server.")


from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(
    title="Handwritten Mathematical Formula Recognition System",
    description="Modern AI-powered LaTeX formula recognition system built with PyTorch & FastAPI.",
    version="1.0.0",
    lifespan=lifespan
)

# Enable CORS for Vercel frontend & cross-origin API calls
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount Static Files & Templates
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))

# Allowed image extensions & maximum size (10 MB)
ALLOWED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
MAX_FILE_SIZE = 10 * 1024 * 1024


@app.get("/", response_class=HTMLResponse)
async def render_homepage(request: Request):
    """Renders the main Web Application Interface."""
    return templates.TemplateResponse(
        "index.html",
        {"request": request, "title": "Handwritten Formula Analyzer"}
    )


@app.get("/result", response_class=HTMLResponse)
async def render_result_page(request: Request, formula: str = "", image_url: str = ""):
    """Renders the formula result view page."""
    return templates.TemplateResponse(
        "result.html",
        {
            "request": request,
            "title": "Formula Prediction Result",
            "prediction": formula,
            "image_url": image_url
        }
    )


@app.post("/api/predict")
@app.post("/predict")
async def handle_prediction(
    file: UploadFile = File(...),
    decode_method: str = Form("beam"),
    beam_size: int = Form(5)
):
    """
    POST /predict API Endpoint:
    1. Accepts an uploaded handwritten formula image.
    2. Validates image extension and file payload.
    3. Saves file inside static/uploads/ directory.
    4. Executes predict(image_path) -> returns prediction JSON.
    """
    if not file or not file.filename:
        return JSONResponse(
            status_code=400,
            content={"success": False, "error": "No image file uploaded."}
        )

    # Validate Extension
    ext = Path(file.filename).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        return JSONResponse(
            status_code=400,
            content={
                "success": False,
                "error": f"Invalid file format '{ext}'. Allowed formats: PNG, JPG, JPEG, WEBP, BMP."
            }
        )

    try:
        # Read payload bytes & validate size
        contents = await file.read()
        if len(contents) == 0:
            return JSONResponse(
                status_code=400,
                content={"success": False, "error": "Uploaded image file is empty."}
            )

        if len(contents) > MAX_FILE_SIZE:
            return JSONResponse(
                status_code=400,
                content={"success": False, "error": "File size exceeds maximum limit of 10MB."}
            )

        # Generate unique filename & save to uploads
        timestamp = int(time.time())
        unique_filename = f"formula_{timestamp}_{uuid.uuid4().hex[:8]}{ext}"
        saved_file_path = UPLOADS_DIR / unique_filename

        with open(saved_file_path, "wb") as f:
            f.write(contents)

        # Run Multi-line segmentation & Inference
        result = predict_multiline(
            str(saved_file_path),
            decode_method=decode_method,
            beam_size=beam_size
        )

        relative_image_url = f"/static/uploads/{unique_filename}"
        headers = {"ngrok-skip-browser-warning": "true"}

        return JSONResponse(
            status_code=200,
            headers=headers,
            content={
                "success": True,
                "prediction": result["prediction"],
                "lines": result["lines"],
                "is_multiline": result["is_multiline"],
                "line_count": result["line_count"],
                "preprocessed_image_base64": result.get("preprocessed_image_base64"),
                "otsu_image_base64": result.get("otsu_image_base64"),
                "line_crops_base64": result.get("line_crops_base64"),
                "tensor_shape": result.get("tensor_shape", "[1, 3, 128, 512]"),
                "filename": unique_filename,
                "image_url": relative_image_url,
                "decode_method": decode_method
            }
        )

    except Exception as e:
        print(f"[ERROR] [Prediction Error]: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "error": f"Prediction failed due to an internal server error: {str(e)}"
            }
        )


@app.get("/api/health")
@app.get("/health")
async def health_check():
    """Health check endpoint confirming API server & predictor status."""
    try:
        predictor = get_predictor()
        headers = {"ngrok-skip-browser-warning": "true"}
        return JSONResponse(
            status_code=200,
            headers=headers,
            content={
                "status": "healthy",
                "device": str(predictor.device),
                "vocab_size": len(predictor.vocab)
            }
        )
    except Exception as e:
        return JSONResponse(status_code=500, content={"status": "unhealthy", "error": str(e)})


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)
