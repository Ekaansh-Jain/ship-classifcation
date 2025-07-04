// script.js
let selectedFile = null;

// File input handling
document.getElementById('fileInput').addEventListener('change', handleFileSelect);
document.getElementById('classifyBtn').addEventListener('click', classifyImage);

// Drag and drop functionality
const uploadArea = document.querySelector('.upload-area');
uploadArea.addEventListener('dragover', handleDragOver);
uploadArea.addEventListener('dragleave', handleDragLeave);
uploadArea.addEventListener('drop', handleDrop);

function handleFileSelect(event) {
    const file = event.target.files[0];
    if (file) {
        processFile(file);
    }
}

function handleDragOver(event) {
    event.preventDefault();
    uploadArea.classList.add('dragover');
}

function handleDragLeave(event) {
    event.preventDefault();
    uploadArea.classList.remove('dragover');
}

function handleDrop(event) {
    event.preventDefault();
    uploadArea.classList.remove('dragover');
    const file = event.dataTransfer.files[0];
    if (file) {
        processFile(file);
    }
}

function processFile(file) {
    if (!file.type.startsWith('image/')) {
        showError('Please select a valid image file');
        return;
    }

    if (file.size > 10 * 1024 * 1024) {
        showError('File size must be less than 10MB');
        return;
    }

    selectedFile = file;
    const reader = new FileReader();
    reader.onload = function(e) {
        const preview = document.getElementById('imagePreview');
        preview.src = e.target.result;
        preview.style.display = 'block';
    };
    reader.readAsDataURL(file);

    document.getElementById('classifyBtn').disabled = false;
    hideError();
    hideResult();
}

async function classifyImage() {
    if (!selectedFile) return;

    showLoading();
    hideError();
    hideResult();

    const formData = new FormData();
    formData.append('image', selectedFile);

    try {
        const response = await fetch('/predict', {
            method: 'POST',
            body: formData
        });

        if (response.ok) {
            const result = await response.json();
            displayResult(result);
        } else {
            const error = await response.json();
            showError(error.error || 'Classification failed');
        }
    } catch (error) {
        showError('Unable to connect to server. Make sure the Flask app is running.');
    } finally {
        hideLoading();
    }
}

function displayResult(result) {
    if (!result.success) {
        showError(result.error || 'Classification failed');
        return;
    }

    document.getElementById('resultPlaceholder').style.display = 'none';
    document.getElementById('resultContent').style.display = 'block';

    document.getElementById('predictedClass').textContent = result.predicted_class + ' Ship';
    document.getElementById('confidenceScore').textContent = `Confidence: ${(result.confidence * 100).toFixed(1)}%`;

    if (result.description) {
        document.getElementById('shipDescription').textContent = result.description.description;

        const featuresList = document.getElementById('featuresList');
        featuresList.innerHTML = '';
        result.description.features.forEach(feature => {
            const li = document.createElement('li');
            li.textContent = feature;
            featuresList.appendChild(li);
        });
    }

    const allPredictionsDiv = document.getElementById('allPredictions');
    const existingBars = allPredictionsDiv.querySelectorAll('.prediction-bar');
    existingBars.forEach(bar => bar.remove());

    if (result.all_predictions) {
        const sortedPredictions = Object.entries(result.all_predictions)
            .sort(([, a], [, b]) => b - a);

        sortedPredictions.forEach(([shipType, confidence]) => {
            const predictionBar = document.createElement('div');
            predictionBar.className = 'prediction-bar';
            predictionBar.innerHTML = `
                <div class="prediction-label">${shipType}</div>
                <div class="progress-bar">
                    <div class="progress-fill" style="width: ${confidence * 100}%"></div>
                </div>
                <div class="prediction-value">${(confidence * 100).toFixed(1)}%</div>
            `;
            allPredictionsDiv.appendChild(predictionBar);
        });
    }
}

function showLoading() {
    document.getElementById('loading').style.display = 'block';
    document.getElementById('classifyBtn').disabled = true;
}

function hideLoading() {
    document.getElementById('loading').style.display = 'none';
    document.getElementById('classifyBtn').disabled = false;
}

function showError(message) {
    const errorDiv = document.getElementById('errorMessage');
    errorDiv.textContent = message;
    errorDiv.style.display = 'block';
}

function hideError() {
    document.getElementById('errorMessage').style.display = 'none';
}

function hideResult() {
    document.getElementById('resultPlaceholder').style.display = 'block';
    document.getElementById('resultContent').style.display = 'none';
}
