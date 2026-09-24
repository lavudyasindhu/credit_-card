/*
 * script.js
 * ---------
 * All frontend logic for talking to the FastAPI backend.
 * No page reloads: every action uses fetch() and updates the DOM directly.
 */

// Change this if your backend runs on a different host/port.
const API_BASE_URL = "http://127.0.0.1:8000";

// ---- Element references ----
const apiStatusDot = document.getElementById("apiStatusDot");
const apiStatusText = document.getElementById("apiStatusText");

const featureInput = document.getElementById("featureInput");
const predictBtn = document.getElementById("predictBtn");
const loadingIndicator = document.getElementById("loadingIndicator");
const errorBox = document.getElementById("errorBox");
const resultCard = document.getElementById("resultCard");
const resultValue = document.getElementById("resultValue");

const trainBtn = document.getElementById("trainBtn");
const metricsLoading = document.getElementById("metricsLoading");
const metricR2 = document.getElementById("metricR2");
const metricMAE = document.getElementById("metricMAE");
const metricMSE = document.getElementById("metricMSE");
const metricRMSE = document.getElementById("metricRMSE");

const dataTotalRecords = document.getElementById("dataTotalRecords");
const dataFeatureColumn = document.getElementById("dataFeatureColumn");
const dataTargetColumn = document.getElementById("dataTargetColumn");
const dataFeatureRange = document.getElementById("dataFeatureRange");
const dataTargetRange = document.getElementById("dataTargetRange");

// Formats a number as US currency, e.g. 237022.25 -> "$237,022.25"
function formatCurrency(value) {
  return new Intl.NumberFormat("en-US", {
    style: "currency",
    currency: "USD",
    maximumFractionDigits: 2,
  }).format(value);
}

function showError(message) {
  errorBox.textContent = message;
  errorBox.classList.remove("hidden");
}

function hideError() {
  errorBox.classList.add("hidden");
  errorBox.textContent = "";
}

// ---------------------------------------------------------------------------
// 1. API HEALTH CHECK  (GET /)
// ---------------------------------------------------------------------------
async function checkApiStatus() {
  try {
    const response = await fetch(`${API_BASE_URL}/`);
    if (!response.ok) throw new Error("API responded with an error");
    apiStatusDot.classList.add("online");
    apiStatusDot.classList.remove("offline");
    apiStatusText.textContent = "API Online";
  } catch (err) {
    apiStatusDot.classList.add("offline");
    apiStatusDot.classList.remove("online");
    apiStatusText.textContent = "API Offline";
  }
}

// ---------------------------------------------------------------------------
// 2. LOAD DATASET INFO  (GET /data)
// ---------------------------------------------------------------------------
async function loadDatasetInfo() {
  try {
    const response = await fetch(`${API_BASE_URL}/data`);
    if (!response.ok) throw new Error("Failed to load dataset info");
    const data = await response.json();

    dataTotalRecords.textContent = data.total_records;
    dataFeatureColumn.textContent = data.feature_column;
    dataTargetColumn.textContent = data.target_column;
    dataFeatureRange.textContent =
      `${data.feature_stats.min} / ${data.feature_stats.mean} / ${data.feature_stats.max}`;
    dataTargetRange.textContent =
      `${formatCurrency(data.target_stats.min)} / ${formatCurrency(data.target_stats.mean)} / ${formatCurrency(data.target_stats.max)}`;
  } catch (err) {
    console.error(err);
    dataTotalRecords.textContent = "Unavailable";
  }
}

// ---------------------------------------------------------------------------
// 3. TRAIN MODEL  (POST /train)
// ---------------------------------------------------------------------------
async function trainModel() {
  metricsLoading.classList.remove("hidden");
  trainBtn.disabled = true;

  try {
    const response = await fetch(`${API_BASE_URL}/train`, { method: "POST" });
    const data = await response.json();

    if (!response.ok) {
      throw new Error(data.detail || "Training failed.");
    }

    metricR2.textContent = data.metrics.r2_score;
    metricMAE.textContent = formatCurrency(data.metrics.mae);
    metricMSE.textContent = data.metrics.mse.toLocaleString("en-US");
    metricRMSE.textContent = formatCurrency(data.metrics.rmse);

    // Dataset info may have changed too (e.g. if data.csv was updated)
    await loadDatasetInfo();
  } catch (err) {
    showError(`Training error: ${err.message}`);
  } finally {
    metricsLoading.classList.add("hidden");
    trainBtn.disabled = false;
  }
}

// ---------------------------------------------------------------------------
// 4. PREDICT  (POST /predict)
// ---------------------------------------------------------------------------
async function predict() {
  hideError();
  resultCard.classList.add("hidden");

  const rawValue = featureInput.value.trim();

  // --- Basic client-side validation before calling the API ---
  if (rawValue === "") {
    showError("Please enter a house size in square feet.");
    return;
  }
  const featureValue = Number(rawValue);
  if (Number.isNaN(featureValue) || featureValue <= 0) {
    showError("Please enter a valid positive number.");
    return;
  }

  loadingIndicator.classList.remove("hidden");
  predictBtn.disabled = true;

  try {
    const response = await fetch(`${API_BASE_URL}/predict`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ feature: featureValue }),
    });

    const data = await response.json();

    if (!response.ok) {
      // FastAPI validation errors come back as an array in `detail`
      const detailMessage = Array.isArray(data.detail)
        ? data.detail.map((d) => d.msg).join(", ")
        : data.detail;
      throw new Error(detailMessage || "Prediction failed.");
    }

    resultValue.textContent = formatCurrency(data.prediction);
    resultCard.classList.remove("hidden");
  } catch (err) {
    showError(err.message || "Something went wrong. Is the backend running?");
  } finally {
    loadingIndicator.classList.add("hidden");
    predictBtn.disabled = false;
  }
}

// ---------------------------------------------------------------------------
// EVENT LISTENERS
// ---------------------------------------------------------------------------
predictBtn.addEventListener("click", predict);
trainBtn.addEventListener("click", trainModel);

// Allow pressing Enter in the input field to trigger a prediction
featureInput.addEventListener("keydown", (e) => {
  if (e.key === "Enter") {
    predict();
  }
});

// ---------------------------------------------------------------------------
// INITIAL LOAD
// ---------------------------------------------------------------------------
checkApiStatus();
loadDatasetInfo();
