# Handwritten Mathematical Formula Recognition System

A production-ready, modern AI web application for converting handwritten mathematical formula images into clean LaTeX markup and rendered mathematical equations using **FastAPI**, **Uvicorn**, **PyTorch**, **MathJax**, and **Bootstrap 5**.

---

## 🌟 Key Features

- **Deep Learning Architecture**: EfficientNet-B0 Encoder + Positional Encoding + 4-Layer Transformer Decoder.
- **FastAPI & Uvicorn Backend**: Asynchronous RESTful server providing high-throughput prediction endpoints.
- **Model Lifetime Preloading**: Model weights and vocabulary are loaded into GPU/CPU memory **only once** on server startup.
- **Dual Decoding Strategies**: Supports both **Beam Search** (with length penalty parameter) and **Greedy Search**.
- **Interactive UI**:
  - Glassmorphic UI with vibrant AI gradient themes.
  - Drag-and-Drop image dropzone & file picker.
  - Live upload preview with file size validation.
  - Progress bar and spinner animations.
- **Mathematical Rendering**: Live MathJax 3 SVG rendering of LaTeX formulas.
- **Action Tools**: One-click **Copy LaTeX to Clipboard** and **Download `.tex` File**.
- **Dark / Light Theme**: Dynamic theme toggle with local storage persistence.

---

## 📁 Project Structure

```text
handwrittenformulaanalyzer/
│
├── app.py              # FastAPI server routes & lifespan model loader
├── predict.py          # ModelPredictor singleton & inference handler
├── model.py            # Vocabulary & PyTorch architecture (Encoder/Decoder)
├── utils.py            # Image preprocessing, transforms, & greedy/beam search
├── requirements.txt    # Python dependency specifications
├── README.md           # Documentation
│
├── weights/
│   ├── best_model.pth  # Pretrained PyTorch model checkpoint
│   └── vocab.pkl       # Serialized Vocabulary mapping object
│
├── static/
│   ├── css/
│   │   └── style.css   # Custom CSS theme & glassmorphic styling
│   ├── js/
│   │   └── script.js   # Client-side file handling, fetch API & MathJax integration
│   └── uploads/        # Saved user-uploaded formula images
│
└── templates/
    ├── index.html      # Main application homepage template
    └── result.html     # Formula result template
```

---

## 🛠️ Installation & Setup

### 1. Prerequisites
- Python 3.9+ installed
- CUDA-enabled GPU (optional, auto-fallback to CPU)

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

---

## 🚀 Running the Web Application

Start the FastAPI server using Uvicorn:

```bash
python app.py
```
*Or directly via uvicorn:*
```bash
uvicorn app:app --host 127.0.0.1 --port 8000 --reload
```

Open your browser and navigate to:
👉 **`http://127.0.0.1:8000`**

---

## 🔌 API Documentation

### `POST /predict`

Uploads an image file and returns the recognized LaTeX formula string.

**Request:**
- **Content-Type**: `multipart/form-data`
- **Body**:
  - `file`: Image file (PNG, JPG, JPEG, WEBP, BMP)
  - `decode_method` *(optional)*: `"beam"` or `"greedy"` (default: `"beam"`)
  - `beam_size` *(optional)*: `1` to `10` (default: `5`)

**Response:** `200 OK`
```json
{
  "success": true,
  "prediction": "\\frac{5}{14}=\\frac{1}{x}",
  "filename": "formula_1720000000_a1b2c3d4.png",
  "image_url": "/static/uploads/formula_1720000000_a1b2c3d4.png",
  "decode_method": "beam"
}
```

---

## 🔒 Error Handling

- Returns `400 Bad Request` if no file is uploaded, format is unsupported, or file size exceeds 10MB limit.
- Returns `500 Internal Server Error` with JSON error payload if inference fails.

---

## 📜 License
Developed as part of the Handwritten Formula Recognition System project.
