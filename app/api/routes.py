import time
import uuid
from pathlib import Path

from fastapi import APIRouter, File, Form, Request, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates

from app.models.predict import get_predictor, predict_multiline
from app.models.solver import solve_and_explain

BASE_DIR = Path(__file__).resolve().parents[1]
STATIC_DIR = BASE_DIR / "static"
UPLOADS_DIR = STATIC_DIR / "uploads"
TEMPLATES_DIR = BASE_DIR / "templates"

UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
(STATIC_DIR / "css").mkdir(parents=True, exist_ok=True)
(STATIC_DIR / "js").mkdir(parents=True, exist_ok=True)

templates = Jinja2Templates(directory=str(TEMPLATES_DIR))

router = APIRouter()

ALLOWED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
MAX_FILE_SIZE = 10 * 1024 * 1024


@router.get("/", response_class=HTMLResponse)
async def render_homepage(request: Request):
    return templates.TemplateResponse(
        "index.html",
        {"request": request, "title": "Handwritten Formula Analyzer"}
    )


@router.get("/result", response_class=HTMLResponse)
async def render_result_page(request: Request, formula: str = "", image_url: str = ""):
    return templates.TemplateResponse(
        "result.html",
        {
            "request": request,
            "title": "Formula Prediction Result",
            "prediction": formula,
            "image_url": image_url
        }
    )


@router.post("/api/predict")
@router.post("/predict")
async def handle_prediction(
    file: UploadFile = File(...),
    decode_method: str = Form("beam"),
    beam_size: int = Form(5)
):
    if not file or not file.filename:
        return JSONResponse(
            status_code=400,
            content={"success": False, "error": "No image file uploaded."}
        )

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

        timestamp = int(time.time())
        unique_filename = f"formula_{timestamp}_{uuid.uuid4().hex[:8]}{ext}"
        saved_file_path = UPLOADS_DIR / unique_filename

        with open(saved_file_path, "wb") as f:
            f.write(contents)

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


@router.options("/api/solve")
@router.options("/solve")
async def solve_options():
    headers = {
        "Access-Control-Allow-Origin": "*",
        "Access-Control-Allow-Methods": "POST, OPTIONS",
        "Access-Control-Allow-Headers": "*",
        "ngrok-skip-browser-warning": "true"
    }
    return JSONResponse(status_code=200, headers=headers, content={"status": "ok"})


@router.post("/api/solve")
async def solve_formula_endpoint(request: Request):
    headers = {
        "Access-Control-Allow-Origin": "*",
        "ngrok-skip-browser-warning": "true"
    }
    try:
        xmin, xmax, yoffset, resolution = -10.0, 10.0, 0.0, 400
        try:
            body = await request.json()
            formula = body.get("formula", "")
            api_key = body.get("api_key", None)
            xmin = float(body.get("xmin", -10.0))
            xmax = float(body.get("xmax", 10.0))
            yoffset = float(body.get("yoffset", 0.0))
            resolution = int(body.get("resolution", 400))
        except Exception:
            form = await request.form()
            formula = form.get("formula", "")
            api_key = form.get("api_key", None)
            if form.get("xmin"): xmin = float(form.get("xmin"))
            if form.get("xmax"): xmax = float(form.get("xmax"))
            if form.get("yoffset"): yoffset = float(form.get("yoffset"))
            if form.get("resolution"): resolution = int(form.get("resolution"))

        if not formula:
            return JSONResponse(status_code=400, headers=headers, content={"success": False, "error": "Formula is required."})

        result = solve_and_explain(formula, api_key=api_key, xmin=xmin, xmax=xmax, yoffset=yoffset, resolution=resolution)
        return JSONResponse(status_code=200, headers=headers, content=result)
    except Exception as e:
        print(f"[ERROR] [/api/solve Error]: {str(e)}")
        return JSONResponse(status_code=500, headers=headers, content={"success": False, "error": str(e)})


@router.post("/solve")
async def solve_formula_alias(request: Request):
    return await solve_formula_endpoint(request)


@router.get("/api/health")
@router.get("/health")
async def health_check():
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
