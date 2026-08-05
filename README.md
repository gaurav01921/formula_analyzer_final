# Handwritten Mathematical Formula Recognition System

This project turns a photo of a handwritten math formula into clean LaTeX text and renders the result in the browser. It combines a neural network model with a lightweight FastAPI backend and a simple web frontend so you can upload an image, get a prediction, and immediately see the rendered formula.

---

## 🌟 What This Project Does

- Accepts handwritten formula images via a web frontend or API.
- Preprocesses the image and normalizes it to the model input shape.
- Uses an EfficientNet-based encoder and a Transformer decoder to predict LaTeX tokens.
- Supports both greedy decoding and beam search.
- Handles multi-line formulas by splitting the image into line crops and predicting each line independently.
- Renders the predicted LaTeX with MathJax so users can see the equation instantly.

---

## 🧠 System Architecture

This app is built around three main layers:

1. **User Interface**
   - Browser-based frontend served from `templates/index.html`.
   - Uses JavaScript in `static/js/script.js` to upload images, display progress, and show rendered formulas.
   - Saves uploaded files under `static/uploads/` for convenient preview.

2. **Backend API**
   - `app.py` runs FastAPI and exposes routes like `/` and `/predict`.
   - It mounts static files, renders templates, validates image uploads, and returns JSON results.
   - The backend initializes the model once on startup so inference stays fast.

3. **Model + Prediction Engine**
   - `model.py` defines the neural network: EfficientNet-B0 encoder + Transformer decoder.
   - `predict.py` loads the model checkpoint and vocabulary, then wraps prediction logic in a singleton.
   - `utils.py` handles image resizing, preprocessing, line segmentation, and decoder algorithms.

---

## 🔁 Model Flow

Here is how a single image flows through the system:

1. User uploads an image from the browser.
2. FastAPI receives the upload and validates its file type and size.
3. The image is saved into `static/uploads/` with a unique filename.
4. `predict_multiline()` in `predict.py` begins inference:
   - `segment_formula_lines()` detects one or more handwritten formula lines.
   - Each line crop is preprocessed with `preprocess_image()`.
   - The model encodes the image using EfficientNet-B0 and decodes tokens with the Transformer.
   - `beam_search_decode()` or `greedy_decode()` generates the final LaTeX.
5. The API returns the combined prediction string plus metadata and visual debug images.
6. The frontend displays the predicted formula and renders it with MathJax.

---

## 🧩 Task Flow

The app is designed around these tasks:

```text
+-------------------------------+
|          Startup              |
|-------------------------------|
| Load FastAPI and model once   |
| with `init_predictor()`       |
+---------------+---------------+
                |
                v
+-------------------------------+
|     Upload & Validation       |
|-------------------------------|
| Validate file type and size   |
| Save image to uploads folder  |
+---------------+---------------+
                |
                v
+-------------------------------+
|         Segmentation          |
|-------------------------------|
| Detect handwritten formula    |
| lines and crop each line      |
+---------------+---------------+
                |
                v
+-------------------------------+
|          Inference            |
|-------------------------------|
| Preprocess each crop and      |
| decode with greedy or beam    |
+---------------+---------------+
                |
                v
+-------------------------------+
|           Output              |
|-------------------------------|
| Combine results, return JSON, |
| render LaTeX on frontend      |
+-------------------------------+
```

- **Startup**
  - `app.py` initializes FastAPI and loads the model once using `init_predictor()`.
  - The vocabulary and weights are loaded into CPU/GPU memory.

- **Upload & Validation**
  - The `/predict` endpoint verifies the image extension and maximum size.
  - The uploaded image is stored safely with a timestamped filename.

- **Segmentation**
  - If the image contains multiple formula lines, the app splits it into separate crops.
  - This improves accuracy for stacked expressions.

- **Inference**
  - Each cropped line is converted to a tensor and run through the model.
  - The decoder builds the LaTeX sequence token-by-token.

- **Output**
  - Predictions are joined into a final string.
  - The API returns JSON with `prediction`, `lines`, `image_url`, and optional debug images.
  - The frontend renders both LaTeX text and the formatted equation.

---

## 📁 Project Structure

```text
handwrittenformulaanalyzer/
│
├── app.py              # FastAPI application and API routes
├── predict.py          # Singleton predictor and inference wrappers
├── model.py            # Encoder/Decoder architecture and vocabulary loader
├── utils.py            # Preprocessing, line segmentation, and decoding helpers
├── requirements.txt    # Python dependency list
├── README.md           # Project documentation
│
├── weights/
│   ├── best_model.pth  # Pretrained PyTorch model checkpoint
│   └── vocab.pkl       # Serialized vocabulary object
│
├── static/
│   ├── css/
│   │   └── style.css   # Styling for the app UI
│   ├── js/
│   │   └── script.js   # Client upload logic, API calls, and MathJax rendering
│   └── uploads/        # Saved prediction input images
│
└── templates/
    ├── index.html      # Main interface template
    └── result.html     # Result display template
```

---

## 🛠️ Installation & Setup

### Prerequisites
- Python 3.9 or higher
- Optional: CUDA-enabled GPU for faster inference

### Install dependencies

```bash
pip install -r requirements.txt
```

---

## 🚀 Run Locally

Start the app:

```bash
python app.py
```

Or with Uvicorn:

```bash
uvicorn app:app --host 127.0.0.1 --port 8000 --reload
```

Then open:

**`http://127.0.0.1:8000`**

---

## 🔌 API Endpoint

### `POST /predict`

Accepts an uploaded image and returns the recognized LaTeX prediction.

**Request:** `multipart/form-data`
- `file`: handwritten formula image file
- `decode_method` (optional): `beam` or `greedy`
- `beam_size` (optional): integer beam width

**Response:**

```json
{
  "success": true,
  "prediction": "\\frac{5}{14}=\\frac{1}{x}",
  "lines": ["\\frac{5}{14}=\\frac{1}{x}"],
  "is_multiline": false,
  "line_count": 1,
  "filename": "formula_...png",
  "image_url": "/static/uploads/formula_...png",
  "decode_method": "beam"
}
```

---

## 🧪 Notes

- Model weights are loaded once at startup for fast inference.
- Multi-line prediction is supported by segmenting the input into line crops.
- The app returns optional debug visualizations for preprocessed images and binarized line crops.

---

## ✨ Improvements You Can Make

- Add support for more upload formats or a drag-and-drop UI.
- Improve the line segmentation algorithm for dense handwritten math.
- Add a training script to fine-tune the model on new formula datasets.

---

## 📜 License
This repository is for the Handwritten Mathematical Formula Recognition System.
