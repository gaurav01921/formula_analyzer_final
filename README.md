# Handwritten Formula Analyzer

An end-to-end artificial intelligence and computer vision platform that converts handwritten mathematical equations from notebook paper photos into clean, copy-ready LaTeX code, evaluates exact symbolic solutions with SymPy, plots function curves using Matplotlib, and provides step-by-step pedagogical explanations powered by Gemini AI.

---

## 🤖 About our AI

Recognizing handwritten mathematical expressions is one of the most intriguing challenges in computer vision and artificial intelligence. Unlike standard text documents—where words flow linearly from left to right—mathematical formulas are inherently two-dimensional. A single handwritten expression can combine nested fractions, exponents, subscripts, square roots, Greek letters, and summation limits, where the spatial arrangement of symbols carries as much meaning as the symbols themselves.

Traditional Optical Character Recognition (OCR) engines struggle with mathematical notation because they fail to capture these non-linear spatial dependencies. 

To solve this, our system employs a hybrid **CRNN + Transformer Decoder** architecture:
- **Vision Encoder (EfficientNet-B0)**: Acts as the visual eyes of the system. Fine-tuned via transfer learning, it extracts multi-scale spatial feature maps from raw handwritten strokes while suppressing notebook line noise.
- **Sinusoidal Positional Embeddings**: Encodes 2D spatial positions so the model understands symbol placement (such as recognizing whether a character is an exponent $x^2$ or a subscript $x_2$).
- **Sequence Decoder (Transformer)**: Uses multi-head self-attention over a 241-token mathematical vocabulary to decode LaTeX expressions token-by-token. Using **Beam Search decoding ($k=5$)**, it evaluates multiple candidate sequences simultaneously to pick the most grammatically coherent mathematical formula.

---

## 📌 Project Overview

**Handwritten Formula Analyzer** provides a complete pipeline for recognizing, editing, solving, and understanding handwritten mathematical content.

1. **Image Upload & Line Segmentation**: Users can upload any photo or notebook snapshot. OpenCV uses Adaptive Gaussian Thresholding and Horizontal Projection Profiling (HPP) to isolate dark ink strokes and segment multi-line stacked equations.
2. **Neural LaTeX Generation**: The preprocessed image tensor ($128 \times 512$) is passed through the PyTorch model to generate formatted LaTeX code.
3. **Interactive KaTeX Editor**: Rendered instantly in the browser using KaTeX, users can edit or refine the recognized LaTeX in real-time.
4. **Symbolic Math Solver (SymPy)**: Computes exact symbolic solutions for linear equations, quadratic roots, indefinite integrals, and summations.
5. **Function Curve Plotter (Matplotlib)**: Automatically generates dark-mode function curve graphs ($x \in [-10, 10]$) exported as high-resolution base64 data URLs.
6. **AI Concept Breakdown (Gemini AI)**: Generates step-by-step educational explanations and real-world mathematical intuition powered by `gemini-2.5-flash`.

---

## 🎯 Objectives

- **High Accuracy Recognition**: Achieve high token-level accuracy on complex handwritten mathematical expressions from the HME100K dataset.
- **Robust Preprocessing**: Automatically handle real-world noise, such as lined notebook paper, uneven lighting, and variable handwriting styles.
- **End-to-End Learning Tool**: Go beyond simple OCR by integrating symbolic math solving, graphing, and AI explanations into a single web application.
- **Low Latency & High Performance**: Provide fast neural inference using lightweight EfficientNet-B0 feature extraction and FastAPI backend deployment.
- **Seamless Accessibility**: Deliver a zero-dependency, responsive web interface accessible across desktop, tablet, and mobile browsers.

---

## ✨ Key Features

- 📷 **Smart Image Preprocessing**: Automatic background binarization via Otsu thresholding and line cropping with Horizontal Projection Profiling.
- ⚡ **CRNN + Transformer Model**: Vision encoder paired with a 4-layer, 8-head Transformer Decoder trained on 241 LaTeX tokens.
- 🎯 **Flexible Decoding Modes**: Choose between **Greedy Search** for instant results or **Beam Search ($k=5$)** for maximum accuracy.
- ✏️ **Live Interactive LaTeX Editor**: Edit recognized LaTeX code with real-time KaTeX rendering.
- 🧮 **SymPy Symbolic Engine**: Evaluates exact roots (e.g. $x+5=0 \implies x=-5$), integrals ($\int x^2 dx = \frac{x^3}{3} + C$), and summations.
- 📈 **Matplotlib Dark-Mode Grapher**: Visualizes mathematical function curves on a dark background.
- 🧠 **Gemini AI Pedagogical Explanations**: Generates step-by-step mathematical intuition and derivations.
- 🛡️ **Fail-Safe Client-Side Solver**: Integrated client-side mathematical solver fallback ensuring the **Solve** button always succeeds even if offline.
- 📜 **Local Recognition History**: Automatically saves up to 30 past predictions to `localStorage` with thumbnails, time stamps, and copy/delete actions.
- 🎨 **Modern Glassmorphism UI**: Beautiful dark-mode interface built with CSS design tokens, animated progress steppers, and responsive tab navigation.

---

## 🏗️ End-to-End System Architecture

```text
  ┌─────────────────┐
  │ User Uploads    │ (Notebook paper / Scanned Image)
  │ Formula Image   │
  └────────┬────────┘
           │
           v
  ┌─────────────────┐
  │ OpenCV          │  - Adaptive Gaussian Thresholding
  │ Preprocessing   │  - Horizontal Projection Profiling (HPP) Line Crop
  │ & Line Crop     │  - Binarization & Tensor Normalization [1, 3, 128, 512]
  └────────┬────────┘
           │
           v
  ┌─────────────────┐
  │ PyTorch Model   │  - EfficientNet-B0 Spatial Feature Extractor
  │ Encoder-Decoder │  - 1x1 Conv Projection (d_model=256)
  │ Inference       │  - 4-Layer Transformer Decoder (8 Heads)
  │                 │  - Beam Search Decoding (k=5)
  └────────┬────────┘
           │
           v
  ┌─────────────────┐
  │ Recognized      │  - Raw LaTeX Output: \int x^2 dx
  │ LaTeX Code      │  - Real-Time KaTeX Math Rendering
  └────────┬────────┘
           │
           v
  ┌─────────────────┬───────────────────┬──────────────────┐
  │                 │                   │                  │
  v                 v                   v                  v
┌───────────┐ ┌──────────────┐ ┌─────────────────┐ ┌────────────────┐
│ KaTeX     │ │ SymPy Engine │ │ Matplotlib      │ │ Gemini AI      │
│ Live      │ │ Symbolic     │ │ Dark-Mode       │ │ Step-by-Step   │
│ Editor    │ │ Solver       │ │ Curve Plotter   │ │ Explanation    │
└───────────┘ └──────────────┘ └─────────────────┘ └────────────────┘
```

---

## 🛠️ Tech Stack

| Category | Technologies & Tools |
| :--- | :--- |
| **Core Language** | Python 3.10+ |
| **Deep Learning** | PyTorch, torchvision, EfficientNet-B0 (Transfer Learning), Transformer Decoder |
| **Computer Vision** | OpenCV (`cv2`), NumPy, Pillow (`PIL`) |
| **Symbolic Math & Graphing** | SymPy, Matplotlib |
| **AI Explanation Engine** | Google Gemini API (`google-genai` / `gemini-2.5-flash`) |
| **Backend Framework** | FastAPI, Uvicorn, Pydantic |
| **Frontend UI** | HTML5, Vanilla CSS3 (Design Tokens), Vanilla JS (ES6+), FontAwesome 6 |
| **Math Rendering** | KaTeX 0.16.8 |
| **Deployment & Cloud** | Vercel Edge CDN, Ngrok Secure HTTPS Tunnel |

---

## 📁 Repo Structure

```text
handwrittenformulaanalyzer/
├── app.py                     # Main FastAPI server with /api/predict & /api/solve routes
├── predict.py                 # FormulaPredictor singleton & Beam/Greedy decoding logic
├── solver.py                  # SymPy symbolic solver, Matplotlib grapher & Gemini AI engine
├── utils.py                   # OpenCV image preprocessing & line segmentation (HPP)
├── model.py                   # PyTorch FormulaRecognizer model architecture
├── dataset.py                 # PyTorch Dataset loader & vocab mapping utilities
├── weights/
│   ├── best_model.pth         # Pretrained PyTorch model checkpoint (~110 MB)
│   └── vocab.pkl              # Token vocabulary pickle file (241 tokens)
├── frontend/                  # Vercel Production Frontend
│   ├── index.html             # Single Page Application HTML (Analyzer, History, About)
│   ├── css/
│   │   └── style.css          # Glassmorphism dark-mode design system & animations
│   └── js/
│       └── app.js             # Client application logic, KaTeX editor & LocalStorage history
├── static/                    # FastAPI Static Assets
│   ├── css/
│   │   └── style.css
│   └── js/
│       └── app.js
├── templates/
│   └── index.html             # FastAPI Jinja2 template
├── vercel.json                # Vercel production deployment configuration
├── requirements.txt           # Python dependencies manifest
└── README.md                  # Project documentation
```

---

## ⚙️ Installation and Local Setup

### Prerequisites

- **Python**: Version 3.10 or higher
- **Git**: Installed on your system
- **Pip**: Latest Python package installer

### Step 1: Clone the Repository

```bash
git clone https://github.com/gaurav01921/formula_analyzer.git
cd formula_analyzer
```

### Step 2: Create a Virtual Environment

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS / Linux
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Verify Weights & Vocabulary

Ensure model weights exist in the `weights/` directory:
- `weights/best_model.pth`
- `weights/vocab.pkl`

---

## 🚀 Quick Start Guide

### 1. Run the FastAPI Server Locally

Start the backend server on `http://127.0.0.1:8000`:

```bash
python -m uvicorn app:app --host 127.0.0.1 --port 8000 --reload
```

Open your browser and navigate to `http://127.0.0.1:8000`.

### 2. Test Formula Recognition

1. Drag and drop a handwritten formula image (or click one of the **Sample Formula Presets** like `Integral \int x^2 dx`).
2. Select your preferred decoding method (**Beam Search** recommended for best accuracy).
3. Click **Recognize Formula**.
4. View the step-by-step progress, Otsu binarized canvas, and rendered KaTeX equation.

### 3. Solve & Plot Equations

1. Modify or edit the recognized formula in the **Edit Formula LaTeX Code** box.
2. Click **⚡ Solve**.
3. View the 3 dedicated result sections:
   - ⚡ **SymPy Mathematical Solution**
   - 📈 **Matplotlib Function Graph**
   - 🧠 **Gemini AI Concept Explanation**

---

Developed with ❤️ using **PyTorch**, **FastAPI**, **SymPy**, **Matplotlib**, and **KaTeX**.
