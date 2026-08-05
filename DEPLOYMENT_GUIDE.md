# Deployment & Setup Guide: FastAPI (Backend + Ngrok) & Vercel (Frontend)

This guide walks you through setting up and running your **Handwritten Mathematical Formula Analyzer** system with:
1. **Backend**: FastAPI + Uvicorn server running locally, exposed publicly using **Ngrok**.
2. **Frontend**: Modern Vercel-ready Single-Page Web Application (`frontend/`).

---

## 🛠️ Step 1: Start the FastAPI Backend Server

1. Open your terminal in the project directory:
   ```bash
   cd c:\Users\dhaka\OneDrive\Desktop\handwrittenformulaanalyzer
   ```

2. Run the FastAPI backend server using `uvicorn`:
   ```bash
   uvicorn app:app --host 127.0.0.1 --port 8000 --reload
   ```

3. Confirm that the server starts up cleanly and prints:
   ```text
   [STARTUP] [FastAPI Server] Starting Handwritten Formula Analyzer...
   [ModelPredictor] Initializing Inference Engine on Device: cpu (or cuda)
   [SUCCESS] [FastAPI Server] Model & Vocabulary successfully preloaded into memory!
   INFO:     Uvicorn running on http://127.0.0.1:8000
   ```

4. Test local health check by visiting [http://127.0.0.1:8000/api/health](http://127.0.0.1:8000/api/health).

---

## 🌐 Step 2: Expose Backend via Ngrok HTTPS Tunnel

To allow your Vercel-hosted frontend to securely call your local FastAPI backend, expose your port 8000 using Ngrok:

1. Download and install **[ngrok](https://ngrok.com/)** if not already installed.
2. In a new terminal window, start the tunnel:
   ```bash
   ngrok http 8000
   ```
3. Copy the generated **Forwarding HTTPS URL** (e.g., `https://a1b2-34-56-78-90.ngrok-free.app`).

---

## 🚀 Step 3: Deploy Frontend to Vercel

The `frontend/` folder contains a clean static web app equipped with `vercel.json`.

### Option A: Using Vercel CLI
1. Install Vercel CLI (if not installed):
   ```bash
   npm i -g vercel
   ```
2. Navigate to the `frontend` directory:
   ```bash
   cd frontend
   ```
3. Deploy to Vercel:
   ```bash
   vercel
   ```
4. Follow the prompts (Select default options) to get your live Vercel URL (e.g. `https://handwritten-formula-analyzer.vercel.app`).

### Option B: Using GitHub & Vercel Dashboard
1. Push your repository to GitHub.
2. Go to [Vercel Dashboard](https://vercel.com/new).
3. Import your GitHub repository.
4. Set **Root Directory** to `frontend`.
5. Click **Deploy**.

---

## 🔗 Step 4: Connect Vercel Frontend to Ngrok Backend

1. Open your live Vercel web app (or local frontend at `http://127.0.0.1:8000`).
2. At the top navigation bar, locate the **Backend URL** input.
3. Paste your **Ngrok HTTPS URL** (e.g., `https://xxxx.ngrok-free.app`).
4. Click **Connect** (or wait for auto-connect).
5. The status badge will turn green (**Connected**).
6. Upload any handwritten formula image or click a sample preset button, then click **Recognize Formula**!

---

## 📂 Project Architecture Overview

```text
handwrittenformulaanalyzer/
├── app.py                  # FastAPI server with CORS & prediction REST APIs
├── predict.py              # ModelPredictor singleton inference engine
├── model.py                # Encoder-Decoder PyTorch neural network model
├── utils.py                # Preprocessing & decoding algorithms
├── weights/
│   └── best_model.pth      # PyTorch model weights (~110 MB)
├── frontend/               # Vercel Frontend Project
│   ├── index.html          # Modern HTML5 SPA interface
│   ├── css/style.css       # Glassmorphism dark theme CSS
│   ├── js/app.js           # KaTeX rendering & API connection logic
│   └── vercel.json         # Vercel deployment configuration
├── templates/              # Local Jinja2 templates (mirrors frontend)
└── static/                 # Local static files & uploaded images
```
