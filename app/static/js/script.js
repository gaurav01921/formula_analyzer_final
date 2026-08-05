document.addEventListener("DOMContentLoaded", function () {
  // UI Elements
  const dropzone = document.getElementById("dropzone");
  const fileInput = document.getElementById("fileInput");
  const browseBtn = document.getElementById("browseBtn");
  const previewContainer = document.getElementById("previewContainer");
  const previewImg = document.getElementById("previewImg");
  const fileNameDisplay = document.getElementById("fileNameDisplay");
  const fileSizeDisplay = document.getElementById("fileSizeDisplay");
  const clearFileBtn = document.getElementById("clearFileBtn");
  
  const predictBtn = document.getElementById("predictBtn");
  const decodeMethodSelect = document.getElementById("decodeMethodSelect");
  const beamSizeRange = document.getElementById("beamSizeRange");
  const beamSizeVal = document.getElementById("beamSizeVal");
  
  const loadingContainer = document.getElementById("loadingContainer");
  const progressBar = document.getElementById("progressBar");
  const statusText = document.getElementById("statusText");
  
  const resultCard = document.getElementById("resultCard");
  const resultImage = document.getElementById("resultImage");
  const rawLatexCode = document.getElementById("rawLatexCode");
  const mathRenderArea = document.getElementById("mathRenderArea");
  const copyBtn = document.getElementById("copyBtn");
  const downloadBtn = document.getElementById("downloadBtn");
  
  const themeToggleBtn = document.getElementById("themeToggleBtn");
  const themeIcon = document.getElementById("themeIcon");

  let selectedFile = null;
  const currentTheme = localStorage.getItem("app_theme") || "dark";
  document.documentElement.setAttribute("data-bs-theme", currentTheme);
  updateThemeIcon(currentTheme);

  if (themeToggleBtn) {
    themeToggleBtn.addEventListener("click", () => {
      const activeTheme = document.documentElement.getAttribute("data-bs-theme");
      const newTheme = activeTheme === "dark" ? "light" : "dark";
      document.documentElement.setAttribute("data-bs-theme", newTheme);
      localStorage.setItem("app_theme", newTheme);
      updateThemeIcon(newTheme);
    });
  }

  function updateThemeIcon(theme) {
    if (!themeIcon) return;
    if (theme === "dark") {
      themeIcon.className = "fas fa-sun text-warning";
    } else {
      themeIcon.className = "fas fa-moon text-primary";
    }
  }

  if (beamSizeRange && beamSizeVal) {
    beamSizeRange.addEventListener("input", function () {
      beamSizeVal.textContent = this.value;
    });
  }

  if (dropzone) {
    dropzone.addEventListener("click", () => fileInput.click());

    ["dragenter", "dragover"].forEach((eventName) => {
      dropzone.addEventListener(eventName, (e) => {
        e.preventDefault();
        e.stopPropagation();
        dropzone.classList.add("dragover");
      });
    });

    ["dragleave", "drop"].forEach((eventName) => {
      dropzone.addEventListener(eventName, (e) => {
        e.preventDefault();
        e.stopPropagation();
        dropzone.classList.remove("dragover");
      });
    });

    dropzone.addEventListener("drop", (e) => {
      const dt = e.dataTransfer;
      const files = dt.files;
      if (files && files.length > 0) {
        handleFileSelect(files[0]);
      }
    });
  }

  if (browseBtn) {
    browseBtn.addEventListener("click", (e) => {
      e.stopPropagation();
      fileInput.click();
    });
  }

  if (fileInput) {
    fileInput.addEventListener("change", function () {
      if (this.files && this.files.length > 0) {
        handleFileSelect(this.files[0]);
      }
    });
  }

  function handleFileSelect(file) {
    const validTypes = ["image/png", "image/jpeg", "image/jpg", "image/webp", "image/bmp"];
    const ext = file.name.split('.').pop().toLowerCase();
    const validExts = ["png", "jpg", "jpeg", "webp", "bmp"];

    if (!validTypes.includes(file.type) && !validExts.includes(ext)) {
      showToast("Invalid file format. Please upload PNG, JPG, JPEG, WEBP, or BMP images.", "danger");
      return;
    }

    if (file.size > 10 * 1024 * 1024) {
      showToast("File size exceeds 10 MB limit.", "warning");
      return;
    }

    selectedFile = file;
    fileNameDisplay.textContent = file.name;
    fileSizeDisplay.textContent = formatBytes(file.size);

    const reader = new FileReader();
    reader.onload = function (e) {
      previewImg.src = e.target.result;
      previewContainer.classList.remove("d-none");
      predictBtn.disabled = false;
    };
    reader.readAsDataURL(file);
  }

  if (clearFileBtn) {
    clearFileBtn.addEventListener("click", function (e) {
      e.stopPropagation();
      resetUploadState();
    });
  }

  function resetUploadState() {
    selectedFile = null;
    fileInput.value = "";
    previewImg.src = "";
    previewContainer.classList.add("d-none");
    predictBtn.disabled = true;
    resultCard.classList.add("d-none");
  }

  function formatBytes(bytes) {
    if (bytes === 0) return "0 Bytes";
    const k = 1024;
    const sizes = ["Bytes", "KB", "MB", "GB"];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + " " + sizes[i];
  }

  if (predictBtn) {
    predictBtn.addEventListener("click", async function () {
      if (!selectedFile) {
        showToast("Please upload a handwritten formula image first.", "warning");
        return;
      }

      // Prepare UI for prediction
      predictBtn.disabled = true;
      loadingContainer.classList.remove("d-none");
      resultCard.classList.add("d-none");
      animateProgressBar(0, 80, 2500);

      const formData = new FormData();
      formData.append("file", selectedFile);
      formData.append("decode_method", decodeMethodSelect ? decodeMethodSelect.value : "beam");
      formData.append("beam_size", beamSizeRange ? beamSizeRange.value : 5);

      try {
        statusText.textContent = "Processing image & running PyTorch Transformer model...";
        const response = await fetch("/predict", {
          method: "POST",
          body: formData
        });

        const data = await response.json();

        if (response.ok && data.success) {
          animateProgressBar(80, 100, 300, () => {
            setTimeout(() => {
              loadingContainer.classList.add("d-none");
              displayResults(data);
              predictBtn.disabled = false;
            }, 400);
          });
        } else {
          loadingContainer.classList.add("d-none");
          predictBtn.disabled = false;
          showToast(data.error || "Prediction failed.", "danger");
        }
      } catch (err) {
        loadingContainer.classList.add("d-none");
        predictBtn.disabled = false;
        showToast("Network error or server unavailable: " + err.message, "danger");
      }
    });
  }

  function animateProgressBar(start, end, duration, callback) {
    let current = start;
    const stepTime = 50;
    const steps = duration / stepTime;
    const increment = (end - start) / steps;

    const interval = setInterval(() => {
      current += increment;
      if (current >= end) {
        current = end;
        clearInterval(interval);
        if (callback) callback();
      }
      progressBar.style.width = `${current}%`;
    }, stepTime);
  }

  function displayResults(data) {
    resultImage.src = data.image_url;
    rawLatexCode.textContent = data.prediction;

    // Render MathJax equation
    mathRenderArea.innerHTML = `\\[ ${data.prediction} \\]`;

    if (window.MathJax) {
      MathJax.typesetPromise([mathRenderArea])
        .then(() => {
          console.log("MathJax rendering complete.");
        })
        .catch((err) => console.error("MathJax Error:", err));
    }

    resultCard.classList.remove("d-none");
    resultCard.scrollIntoView({ behavior: "smooth", block: "start" });
    showToast("Formula recognized successfully!", "success");
  }

  if (copyBtn) {
    copyBtn.addEventListener("click", function () {
      const textToCopy = rawLatexCode.textContent;
      if (!textToCopy) return;

      navigator.clipboard.writeText(textToCopy).then(() => {
        const origHtml = copyBtn.innerHTML;
        copyBtn.innerHTML = `<i class="fas fa-check text-success"></i> Copied!`;
        showToast("LaTeX formula copied to clipboard!", "success");
        setTimeout(() => {
          copyBtn.innerHTML = origHtml;
        }, 2000);
      }).catch(err => {
        showToast("Copy failed: " + err, "danger");
      });
    });
  }

  if (downloadBtn) {
    downloadBtn.addEventListener("click", function () {
      const formulaText = rawLatexCode.textContent;
      if (!formulaText) return;

      const blob = new Blob([formulaText], { type: "text/plain;charset=utf-8" });
      const url = URL.createObjectURL(blob);
      const link = document.createElement("a");
      link.href = url;
      link.download = `formula_prediction.tex`;
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
      URL.revokeObjectURL(url);

      showToast("LaTeX file downloaded (.tex)", "info");
    });
  }

  function showToast(message, type = "info") {
    const toastContainer = document.getElementById("toastContainer");
    if (!toastContainer) return;

    const bgClass = type === "danger" ? "bg-danger text-white" :
                    type === "success" ? "bg-success text-white" :
                    type === "warning" ? "bg-warning text-dark" : "bg-primary text-white";

    const toastEl = document.createElement("div");
    toastEl.className = `toast align-items-center ${bgClass} border-0 show shadow-lg mb-2`;
    toastEl.setAttribute("role", "alert");
    toastEl.innerHTML = `
      <div class="d-flex">
        <div class="toast-body font-weight-medium">
          <i class="fas ${type === 'success' ? 'fa-check-circle' : type === 'danger' ? 'fa-exclamation-circle' : 'fa-info-circle'} me-2"></i>
          ${message}
        </div>
        <button type="button" class="btn-close btn-close-white me-2 m-auto" data-bs-dismiss="toast"></button>
      </div>
    `;

    toastContainer.appendChild(toastEl);
    setTimeout(() => {
      toastEl.classList.remove("show");
      setTimeout(() => toastEl.remove(), 400);
    }, 4000);
  }
});
