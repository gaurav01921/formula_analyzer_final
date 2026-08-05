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

  // DOM Elements for Editor & AI Solver
  const latexEditorInput = document.getElementById('latex-editor-input');
  const btnSolve = document.getElementById('btn-solve');
  const solutionCard = document.getElementById('solution-card');
  const solutionRenderBox = document.getElementById('solution-render-box');
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

  // Solve & Explain with AI Button Handler
  if (btnSolve) {
    btnSolve.addEventListener('click', async () => {
      const formulaToSolve = latexEditorInput ? latexEditorInput.value.trim() : rawLatexOutput.textContent.trim();
      if (!formulaToSolve || formulaToSolve.includes('will appear here')) {
        showToast('Please upload or enter a formula to solve!', true);
        return;
      }

      const backendUrl = backendUrlInput.value.trim().replace(/\/+$/, '');
      btnSolve.disabled = true;
      btnSolve.innerHTML = `<span class="spinner"></span> Solving with AI...`;

      try {
        const response = await fetch(`${backendUrl}/api/solve`, {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
            'ngrok-skip-browser-warning': 'true'
          },
          body: JSON.stringify({ formula: formulaToSolve })
        });

        const data = await response.json();

        if (response.ok && data.success) {
          // Render Solved KaTeX
          if (window.katex && data.solution_latex) {
            solutionRenderBox.innerHTML = '';
            katex.render(data.solution_latex, solutionRenderBox, {
              displayMode: true,
              throwOnError: false
            });
          } else {
            solutionRenderBox.textContent = data.solution_latex || formulaToSolve;
          }

          // Format Markdown Explanation
          explanationContent.innerHTML = formatMarkdown(data.explanation || 'Solution calculated.');
          solutionCard.style.display = 'block';
          solutionCard.scrollIntoView({ behavior: 'smooth', block: 'center' });
          showToast('Formula solved and explained successfully!');
        } else {
          showToast(data.error || 'Failed to solve formula.', true);
        }
      } catch (err) {
        console.error('Solve API Error:', err);
        showToast('Error connecting to solver API.', true);
      } finally {
        btnSolve.disabled = false;
        btnSolve.innerHTML = `<i class="fa-solid fa-brain"></i> Solve & Explain with AI`;
      }
    });
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
