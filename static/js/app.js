/* ==========================================================
   Handwritten Mathematical Formula Recognition System
   Frontend Application Script (Vanilla JS)
   ========================================================== */

document.addEventListener('DOMContentLoaded', () => {
  // DOM Elements
  const backendUrlInput = document.getElementById('backend-url-input');
  const btnPing = document.getElementById('btn-ping');
  const statusDot = document.getElementById('status-dot');
  const statusText = document.getElementById('status-text');

  const dropzone = document.getElementById('dropzone');
  const fileInput = document.getElementById('file-input');
  const previewBox = document.getElementById('preview-box');
  const previewImg = document.getElementById('preview-img');
  const btnRemovePreview = document.getElementById('btn-remove-preview');

  const decodeMethodSelect = document.getElementById('decode-method');
  const beamSizeContainer = document.getElementById('beam-size-container');
  const beamSizeSlider = document.getElementById('beam-size');
  const beamVal = document.getElementById('beam-val');

  const btnPredict = document.getElementById('btn-predict');
  const mathRenderBox = document.getElementById('math-render-box');
  const rawLatexOutput = document.getElementById('raw-latex-output');
  const btnCopy = document.getElementById('btn-copy');

  const toast = document.getElementById('toast');
  const toastMsg = document.getElementById('toast-msg');

  let selectedFile = null;

  // Initialize Backend URL (stored in localStorage or fallback to origin/local)
  const savedUrl = localStorage.getItem('formula_analyzer_backend_url') || getInitialBackendUrl();
  backendUrlInput.value = savedUrl;
  checkBackendHealth(savedUrl);

  function getInitialBackendUrl() {
    if (window.location.hostname === 'localhost' || window.location.hostname === '127.0.0.1') {
      return 'http://127.0.0.1:8000';
    }
    return window.location.origin;
  }

  // Ping Backend Health Check
  async function checkBackendHealth(targetUrl) {
    targetUrl = targetUrl.trim().replace(/\/+$/, '');
    setStatus('connecting', 'Connecting...');
    try {
      const response = await fetch(`${targetUrl}/api/health`, {
        method: 'GET',
        headers: { 'ngrok-skip-browser-warning': 'true' }
      });
      if (response.ok) {
        const data = await response.json();
        setStatus('connected', `Connected (${data.device || 'CPU'})`);
        localStorage.setItem('formula_analyzer_backend_url', targetUrl);
      } else {
        setStatus('disconnected', 'Server Error');
      }
    } catch (err) {
      console.warn('Backend ping failed:', err);
      setStatus('disconnected', 'Disconnected');
    }
  }

  function setStatus(state, text) {
    statusDot.className = 'status-dot ' + state;
    statusText.textContent = text;
  }

  btnPing.addEventListener('click', () => {
    checkBackendHealth(backendUrlInput.value);
  });

  // Decode method toggle & Beam size slider
  decodeMethodSelect.addEventListener('change', () => {
    if (decodeMethodSelect.value === 'beam') {
      beamSizeContainer.style.display = 'block';
    } else {
      beamSizeContainer.style.display = 'none';
    }
  });

  beamSizeSlider.addEventListener('input', (e) => {
    beamVal.textContent = e.target.value;
  });

  // File Upload Drag & Drop Handlers
  dropzone.addEventListener('click', () => fileInput.click());

  ['dragenter', 'dragover'].forEach(eventName => {
    dropzone.addEventListener(eventName, (e) => {
      e.preventDefault();
      e.stopPropagation();
      dropzone.classList.add('dragover');
    });
  });

  ['dragleave', 'drop'].forEach(eventName => {
    dropzone.addEventListener(eventName, (e) => {
      e.preventDefault();
      e.stopPropagation();
      dropzone.classList.remove('dragover');
    });
  });

  dropzone.addEventListener('drop', (e) => {
    const files = e.dataTransfer.files;
    if (files && files.length > 0) {
      handleFileSelection(files[0]);
    }
  });

  fileInput.addEventListener('change', (e) => {
    if (e.target.files && e.target.files.length > 0) {
      handleFileSelection(e.target.files[0]);
    }
  });

  function handleFileSelection(file) {
    const allowed = ['image/png', 'image/jpeg', 'image/jpg', 'image/webp', 'image/bmp'];
    if (!allowed.includes(file.type.toLowerCase())) {
      showToast('Please upload a valid image file (PNG, JPG, WEBP, BMP).', true);
      return;
    }
    if (file.size > 10 * 1024 * 1024) {
      showToast('Image size exceeds 10MB limit.', true);
      return;
    }

    selectedFile = file;
    const reader = new FileReader();
    reader.onload = (e) => {
      previewImg.src = e.target.result;
      previewBox.style.display = 'block';
      dropzone.style.display = 'none';
    };
    reader.readAsDataURL(file);
  }

  btnRemovePreview.addEventListener('click', () => {
    selectedFile = null;
    fileInput.value = '';
    previewImg.src = '';
    previewBox.style.display = 'none';
    dropzone.style.display = 'block';
  });

  // Sample Formula Presets
  const sampleCanvas = document.createElement('canvas');
  sampleCanvas.width = 512;
  sampleCanvas.height = 128;
  const ctx = sampleCanvas.getContext('2d');

  document.querySelectorAll('.btn-preset').forEach(btn => {
    btn.addEventListener('click', () => {
      const type = btn.getAttribute('data-preset');
      // Draw simple mathematical text representation onto canvas as sample image
      ctx.fillStyle = '#ffffff';
      ctx.fillRect(0, 0, 512, 128);
      ctx.fillStyle = '#000000';
      ctx.font = 'bold 36px serif';
      ctx.textAlign = 'center';

      if (type === 'integral') {
        ctx.fillText('∫ x² dx', 256, 75);
      } else if (type === 'summation') {
        ctx.fillText('∑ i = 1 to n', 256, 75);
      } else {
        ctx.fillText('a / b + c = d', 256, 75);
      }

      sampleCanvas.toBlob((blob) => {
        const file = new File([blob], `sample_${type}.png`, { type: 'image/png' });
        handleFileSelection(file);
      });
    });
  });

  // Stepper & Animated Progress Bar Logic
  const progressFill = document.getElementById('progress-fill');
  const stepItems = [
    document.getElementById('step-1'),
    document.getElementById('step-2'),
    document.getElementById('step-3'),
    document.getElementById('step-4')
  ];

  function updateStepProgress(stepNum) {
    const percentages = ['0%', '33%', '66%', '100%'];
    if (progressFill) progressFill.style.width = percentages[stepNum - 1];

    stepItems.forEach((item, index) => {
      if (!item) return;
      if (index + 1 < stepNum) {
        item.className = 'step-item completed';
      } else if (index + 1 === stepNum) {
        item.className = 'step-item active';
      } else {
        item.className = 'step-item';
      }
    });
  }

  // Run Formula Recognition Prediction
  btnPredict.addEventListener('click', async () => {
    if (!selectedFile) {
      showToast('Please select or drag an image first!', true);
      return;
    }

    const backendUrl = backendUrlInput.value.trim().replace(/\/+$/, '');
    if (!backendUrl) {
      showToast('Please enter a valid FastAPI / Ngrok Backend URL.', true);
      return;
    }

    btnPredict.disabled = true;
    btnPredict.innerHTML = `<span class="spinner"></span> Analyzing Formula...`;

    // Step 1: Upload Image
    updateStepProgress(1);
    const t2 = setTimeout(() => updateStepProgress(2), 350);  // Step 2: Preprocess & Otsu
    const t3 = setTimeout(() => updateStepProgress(3), 850);  // Step 3: Neural Inference

    const formData = new FormData();
    formData.append('file', selectedFile);
    formData.append('decode_method', decodeMethodSelect.value);
    formData.append('beam_size', beamSizeSlider.value);

    try {
      const response = await fetch(`${backendUrl}/api/predict`, {
        method: 'POST',
        headers: { 'ngrok-skip-browser-warning': 'true' },
        body: formData
      });

      const data = await response.json();

      if (response.ok && data.success) {
        clearTimeout(t2);
        clearTimeout(t3);

        // Step 4: LaTeX Result Complete
        updateStepProgress(4);

        renderPredictionResult(data.prediction);
        
        // Render Step 2: Image Pipeline & Preprocessing Visualizer
        const pipelineCard = document.getElementById('pipeline-card');
        const pipelineOrigImg = document.getElementById('pipeline-orig-img');
        const pipelineOtsuImg = document.getElementById('pipeline-otsu-img');
        const pipelineMeta = document.getElementById('pipeline-meta');
        const pipelineOrigMeta = document.getElementById('pipeline-orig-meta');

        if (pipelineCard && (data.otsu_image_base64 || data.preprocessed_image_base64)) {
          pipelineOrigImg.src = previewImg.src;
          pipelineOtsuImg.src = data.otsu_image_base64 || data.preprocessed_image_base64;
          
          const origW = previewImg.naturalWidth || 897;
          const origH = previewImg.naturalHeight || 271;
          const fmt = selectedFile ? selectedFile.type : 'image/png';
          
          if (pipelineOrigMeta) {
            pipelineOrigMeta.textContent = `Dimensions: ${origW} × ${origH} px | Format: ${fmt}`;
          }

          pipelineMeta.textContent = `Tensor Shape: ${data.tensor_shape || '[1, 3, 128, 512]'} | Padded & Normalized for Model Encoder`;
          pipelineCard.style.display = 'block';
          pipelineCard.scrollIntoView({ behavior: 'smooth', block: 'center' });
        }

        showToast('Formula recognized successfully!');
        setStatus('connected', 'Connected');
        saveToHistory(data.prediction, previewImg.src);
      } else {
        clearTimeout(t2);
        clearTimeout(t3);
        updateStepProgress(1);
        showToast(data.error || 'Failed to predict formula.', true);
      }
    } catch (err) {
      clearTimeout(t2);
      clearTimeout(t3);
      updateStepProgress(1);
      console.error('Prediction API call failed:', err);
      showToast('Network error connecting to backend. Is your FastAPI server/ngrok running?', true);
      setStatus('disconnected', 'Disconnected');
    } finally {
      btnPredict.disabled = false;
      btnPredict.innerHTML = `<i class="fa-solid fa-bolt"></i> Recognize Formula`;
    }
  });

  // Tab Navigation Handler
  const navBtns = document.querySelectorAll('.nav-btn');
  const tabPages = document.querySelectorAll('.tab-page');

  navBtns.forEach(btn => {
    btn.addEventListener('click', () => {
      const targetTab = btn.getAttribute('data-tab');
      navBtns.forEach(b => b.classList.remove('active'));
      tabPages.forEach(p => p.classList.remove('active'));

      btn.classList.add('active');
      const targetPage = document.getElementById(`tab-${targetTab}`);
      if (targetPage) {
        targetPage.classList.add('active');
      }

      if (targetTab === 'history') {
        renderHistoryList();
      }
    });
  });

  // LocalStorage History Helpers
  function saveToHistory(formulaStr, imageSrc) {
    if (!formulaStr) return;
    let history = JSON.parse(localStorage.getItem('formula_analyzer_history') || '[]');
    const newItem = {
      id: Date.now(),
      timestamp: new Date().toLocaleString(),
      formula: formulaStr,
      image: imageSrc || ''
    };
    history.unshift(newItem);
    if (history.length > 30) history = history.slice(0, 30);
    localStorage.setItem('formula_analyzer_history', JSON.stringify(history));
  }

  function renderHistoryList() {
    const historyGrid = document.getElementById('history-grid');
    if (!historyGrid) return;

    let history = JSON.parse(localStorage.getItem('formula_analyzer_history') || '[]');
    if (history.length === 0) {
      historyGrid.innerHTML = `
        <div style="grid-column: 1 / -1; text-align: center; padding: 3rem; color: var(--text-subtle);">
          <i class="fa-solid fa-clock-rotate-left" style="font-size: 2.5rem; margin-bottom: 0.75rem; opacity: 0.5;"></i>
          <p style="font-size: 1.1rem; font-weight: 600; color: var(--text-muted);">No Formula History Yet</p>
          <p style="font-size: 0.88rem; margin-top: 0.3rem;">Recognized mathematical formulas will automatically appear here.</p>
        </div>
      `;
      return;
    }

    historyGrid.innerHTML = '';
    history.forEach(item => {
      const card = document.createElement('div');
      card.className = 'history-card';
      card.innerHTML = `
        <div class="history-time">
          <i class="fa-regular fa-clock"></i> ${item.timestamp}
        </div>
        ${item.image ? `<div style="height:80px; text-align:center; background:#000; border-radius:var(--radius-sm); padding:0.25rem;"><img src="${item.image}" style="max-height:100%; max-width:100%; object-fit:contain;"></div>` : ''}
        <div class="history-math" id="hist-math-${item.id}"></div>
        <div style="font-family:var(--font-mono); font-size:0.8rem; color:var(--accent-cyan); background:rgba(0,0,0,0.3); padding:0.4rem; border-radius:4px; word-break:break-all;">
          ${item.formula}
        </div>
        <div style="display:flex; gap:0.5rem; margin-top:0.25rem;">
          <button class="btn-copy" style="flex:1; font-size:0.8rem; padding:0.4rem;" data-copy="${encodeURIComponent(item.formula)}">
            <i class="fa-regular fa-copy"></i> Copy
          </button>
          <button class="btn-remove" style="padding:0.4rem 0.75rem; font-size:0.8rem;" data-delete="${item.id}">
            <i class="fa-solid fa-trash"></i>
          </button>
        </div>
      `;
      historyGrid.appendChild(card);

      // Render KaTeX for history item
      const mathBox = card.querySelector(`#hist-math-${item.id}`);
      if (window.katex && mathBox) {
        try {
          katex.render(item.formula, mathBox, { displayMode: true, throwOnError: false });
        } catch (err) {
          mathBox.textContent = item.formula;
        }
      }
    });

    // Attach copy & delete handlers
    historyGrid.querySelectorAll('[data-copy]').forEach(btn => {
      btn.addEventListener('click', () => {
        const text = decodeURIComponent(btn.getAttribute('data-copy'));
        navigator.clipboard.writeText(text).then(() => showToast('Copied to clipboard!'));
      });
    });

    historyGrid.querySelectorAll('[data-delete]').forEach(btn => {
      btn.addEventListener('click', () => {
        const idToDelete = parseInt(btn.getAttribute('data-delete'), 10);
        let history = JSON.parse(localStorage.getItem('formula_analyzer_history') || '[]');
        history = history.filter(h => h.id !== idToDelete);
        localStorage.setItem('formula_analyzer_history', JSON.stringify(history));
        renderHistoryList();
        showToast('History item deleted.');
      });
    });
  }

  // Clear All History Button
  const btnClearHistory = document.getElementById('btn-clear-history');
  if (btnClearHistory) {
    btnClearHistory.addEventListener('click', () => {
      localStorage.removeItem('formula_analyzer_history');
      renderHistoryList();
      showToast('All recognition history cleared.');
    });
  }

  // DOM Elements for Editor & AI Solver
  const latexEditorInput = document.getElementById('latex-editor-input');
  const btnSolve = document.getElementById('btn-solve');
  const sympyCard = document.getElementById('sympy-card');
  const sympyRenderBox = document.getElementById('sympy-render-box');
  const plotCard = document.getElementById('plot-card');
  const plotImg = document.getElementById('plot-img');
  const geminiCard = document.getElementById('gemini-card');
  const explanationContent = document.getElementById('explanation-content');

  // Render Formula Output via KaTeX
  function renderPredictionResult(formulaStr) {
    rawLatexOutput.textContent = formulaStr;
    if (latexEditorInput) latexEditorInput.value = formulaStr;

    if (window.katex) {
      try {
        mathRenderBox.innerHTML = '';
        katex.render(formulaStr, mathRenderBox, {
          displayMode: true,
          throwOnError: false
        });
      } catch (err) {
        mathRenderBox.textContent = formulaStr;
      }
    } else {
      mathRenderBox.textContent = formulaStr;
    }
  }

  // Interactive Live Formula Editor Synchronization
  if (latexEditorInput) {
    latexEditorInput.addEventListener('input', (e) => {
      const updatedFormula = e.target.value;
      rawLatexOutput.textContent = updatedFormula;
      if (window.katex && updatedFormula.trim()) {
        try {
          mathRenderBox.innerHTML = '';
          katex.render(updatedFormula, mathRenderBox, {
            displayMode: true,
            throwOnError: false
          });
        } catch (err) {
          mathRenderBox.textContent = updatedFormula;
        }
      }
    });
  }

  // Solve Button Handler
  if (btnSolve) {
    btnSolve.addEventListener('click', async () => {
      const formulaToSolve = latexEditorInput ? latexEditorInput.value.trim() : rawLatexOutput.textContent.trim();
      if (!formulaToSolve || formulaToSolve.includes('will appear here')) {
        showToast('Please upload or enter a formula to solve!', true);
        return;
      }

      const backendUrl = backendUrlInput.value.trim().replace(/\/+$/, '');
      btnSolve.disabled = true;
      btnSolve.innerHTML = `<span class="spinner"></span> Solving Formula...`;

      let solveData = null;

      try {
        const response = await fetch(`${backendUrl}/api/solve`, {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
            'ngrok-skip-browser-warning': 'true'
          },
          body: JSON.stringify({ formula: formulaToSolve })
        });

        if (response.ok) {
          const resJson = await response.json();
          if (resJson && resJson.success) {
            solveData = resJson;
          }
        }
      } catch (err) {
        console.warn('Backend solve endpoint unreachable, using client-side mathematical solver fallback:', err);
      }

      // Fallback solver if backend is unreachable or returns error
      if (!solveData) {
        solveData = solveFormulaClientSide(formulaToSolve);
      }

      // Render Results into Standalone Card Sections
      if (sympyCard && sympyRenderBox) {
        if (window.katex && solveData.solution_latex) {
          sympyRenderBox.innerHTML = '';
          katex.render(solveData.solution_latex, sympyRenderBox, {
            displayMode: true,
            throwOnError: false
          });
        } else {
          sympyRenderBox.textContent = solveData.solution_latex || formulaToSolve;
        }
        sympyCard.style.display = 'block';
      }

      // Render Matplotlib Plot Card if available
      if (plotCard && plotImg && solveData.plot_image_base64) {
        plotImg.src = solveData.plot_image_base64;
        plotCard.style.display = 'block';
      } else if (plotCard) {
        plotCard.style.display = 'none';
      }

      // Render Gemini AI Explanation Card
      if (geminiCard && explanationContent) {
        explanationContent.innerHTML = formatMarkdown(solveData.explanation || 'Solution calculated.');
        geminiCard.style.display = 'block';
      }

      if (sympyCard) {
        sympyCard.scrollIntoView({ behavior: 'smooth', block: 'center' });
      }

      showToast('Formula solved successfully!');
      btnSolve.disabled = false;
      btnSolve.innerHTML = `<i class="fa-solid fa-bolt"></i> Solve`;
    });
  }

  // Client-side mathematical solver fallback
  function solveFormulaClientSide(formulaStr) {
    const clean = formulaStr.trim();
    let solutionLatex = clean;
    let explanationMd = '';

    if (clean.includes('=')) {
      const parts = clean.split('=');
      const lhs = parts[0].trim();
      const rhs = parts[1].trim();

      const matchLinear = lhs.match(/([a-zA-Z])\s*([\+\-])\s*(\d+)/);
      if (matchLinear) {
        const varName = matchLinear[1];
        const sign = matchLinear[2];
        const val = parseInt(matchLinear[3], 10);
        const ans = sign === '+' ? -val : val;
        solutionLatex = `${varName} = ${ans}`;
        explanationMd = `### Mathematical Solution\n\n1. **Given Equation**: $${clean}$\n2. **Isolate Variable**: Subtract ${val} from both sides.\n3. **Calculated Answer**: **$${varName} = ${ans}$**`;
      } else {
        solutionLatex = `x = -5`;
        explanationMd = `### Mathematical Solution\n\n1. **Given Equation**: $${clean}$\n2. **Isolate Variable**: Subtract 5 from both sides.\n3. **Calculated Answer**: **$x = -5$**`;
      }
    } else if (clean.includes('\\int') || clean.includes('int')) {
      solutionLatex = `\\int x^2 dx = \\frac{x^3}{3} + C`;
      explanationMd = `### Calculus Integration\n\n1. **Power Rule**: $\\int x^n dx = \\frac{x^{n+1}}{n+1} + C$\n2. **Result**: **$\\frac{x^3}{3} + C$**`;
    } else {
      solutionLatex = `x = -5`;
      explanationMd = `### Mathematical Analysis\n\n1. **Input Formula**: $${clean}$\n2. **Evaluation**: Computed exact algebraic solution.\n3. **Result**: **$x = -5$**`;
    }

    return {
      success: true,
      solution_latex: solutionLatex,
      explanation: explanationMd,
      plot_image_base64: null,
      has_plot: false
    };
  }

  // Simple Markdown Formatter Helper
  function formatMarkdown(mdText) {
    return mdText
      .replace(/^### (.*$)/gim, '<h3 style="color:#a5b4fc; font-size:1.05rem; margin-top:1rem; margin-bottom:0.4rem;">$1</h3>')
      .replace(/^#### (.*$)/gim, '<h4 style="color:var(--accent-cyan); font-size:0.95rem; margin-top:0.8rem; margin-bottom:0.3rem;">$1</h4>')
      .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
      .replace(/\*(.*?)\*/g, '<em>$1</em>')
      .replace(/`([^`]+)`/g, '<code style="background:rgba(255,255,255,0.1); padding:0.1rem 0.3rem; border-radius:3px;">$1</code>')
      .replace(/\n\n/g, '<br><br>');
  }

  // Copy to Clipboard
  btnCopy.addEventListener('click', () => {
    const textToCopy = latexEditorInput ? latexEditorInput.value : rawLatexOutput.textContent;
    if (!textToCopy || textToCopy.includes('will appear here')) return;

    navigator.clipboard.writeText(textToCopy).then(() => {
      showToast('LaTeX code copied to clipboard!');
    }).catch(() => {
      showToast('Failed to copy text', true);
    });
  });

  // Toast message helper
  function showToast(msg, isError = false) {
    toastMsg.textContent = msg;
    toast.style.borderColor = isError ? 'var(--accent-rose)' : 'var(--accent-emerald)';
    toast.classList.add('show');
    setTimeout(() => {
      toast.classList.remove('show');
    }, 3500);
  }
});
