from flask import Flask, request, jsonify, render_template_string
from flask_cors import CORS
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image
import numpy as np
import io
import os

app = Flask(__name__)
CORS(app)

# Load your trained model
model = load_model('best_vgg16_model.h5')

# Your class labels
class_names = {
    0: 'Cargo',
    1: 'Military', 
    2: 'Carrier',
    3: 'Cruise',
    4: 'Tanker'
}

# Ship descriptions
ship_info = {
    'Cargo': {
        'description': 'Large vessels designed to carry goods and materials across oceans. These ships are the backbone of global trade.',
        'features': ['Container storage', 'Heavy lifting equipment', 'Large cargo holds']
    },
    'Military': {
        'description': 'Naval vessels designed for defense and combat operations. Equipped with advanced weaponry and radar systems.',
        'features': ['Weapon systems', 'Radar equipment', 'Armored hull']
    },
    'Carrier': {
        'description': 'Aircraft carriers that serve as floating airbases, capable of launching and recovering aircraft at sea.',
        'features': ['Flight deck', 'Aircraft hangar', 'Catapult systems']
    },
    'Cruise': {
        'description': 'Passenger ships designed for leisure travel and entertainment, featuring luxury amenities and accommodations.',
        'features': ['Passenger cabins', 'Entertainment facilities', 'Dining areas']
    },
    'Tanker': {
        'description': 'Specialized vessels designed to transport liquid cargo such as oil, chemicals, or liquefied natural gas.',
        'features': ['Liquid storage tanks', 'Pumping systems', 'Safety equipment']
    }
}

# HTML template embedded in Python
HTML_TEMPLATE = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Ship Classification AI</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            line-height: 1.6;
            color: #333;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
        }

        .container {
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
        }

        .header {
            text-align: center;
            color: white;
            margin-bottom: 40px;
        }

        .header h1 {
            font-size: 3rem;
            margin-bottom: 10px;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
        }

        .header p {
            font-size: 1.2rem;
            opacity: 0.9;
        }

        .main-content {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 40px;
            margin-bottom: 40px;
        }

        .upload-section {
            background: white;
            padding: 30px;
            border-radius: 15px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.2);
        }

        .upload-section h2 {
            margin-bottom: 20px;
            color: #333;
            text-align: center;
        }

        .upload-area {
            border: 3px dashed #ddd;
            border-radius: 10px;
            padding: 40px;
            text-align: center;
            cursor: pointer;
            transition: all 0.3s ease;
            margin-bottom: 20px;
        }

        .upload-area:hover {
            border-color: #667eea;
            background-color: #f8f9ff;
        }

        .upload-area.dragover {
            border-color: #667eea;
            background-color: #f0f2ff;
        }

        #fileInput {
            display: none;
        }

        .upload-icon {
            font-size: 3rem;
            color: #ddd;
            margin-bottom: 10px;
        }

        .preview-container {
            text-align: center;
            margin: 20px 0;
        }

        #imagePreview {
            max-width: 100%;
            max-height: 300px;
            border-radius: 10px;
            box-shadow: 0 5px 15px rgba(0,0,0,0.2);
            display: none;
        }

        .btn {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border: none;
            padding: 12px 30px;
            border-radius: 25px;
            cursor: pointer;
            font-size: 1rem;
            transition: all 0.3s ease;
            width: 100%;
            margin-top: 10px;
        }

        .btn:hover {
            transform: translateY(-2px);
            box-shadow: 0 5px 15px rgba(0,0,0,0.2);
        }

        .btn:disabled {
            opacity: 0.6;
            cursor: not-allowed;
            transform: none;
        }

        .result-section {
            background: white;
            padding: 30px;
            border-radius: 15px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.2);
        }

        .result-section h2 {
            margin-bottom: 20px;
            color: #333;
            text-align: center;
        }

        .result-content {
            display: none;
        }

        .prediction-result {
            background: linear-gradient(135deg, #4CAF50 0%, #45a049 100%);
            color: white;
            padding: 20px;
            border-radius: 10px;
            margin-bottom: 20px;
            text-align: center;
        }

        .prediction-result h3 {
            font-size: 2rem;
            margin-bottom: 10px;
        }

        .confidence-score {
            font-size: 1.2rem;
            opacity: 0.9;
        }

        .ship-details {
            background: #f8f9fa;
            padding: 20px;
            border-radius: 10px;
            margin-bottom: 20px;
        }

        .ship-details h4 {
            color: #333;
            margin-bottom: 10px;
        }

        .ship-details p {
            margin-bottom: 15px;
            line-height: 1.6;
        }

        .features-list {
            list-style: none;
        }

        .features-list li {
            background: #e9ecef;
            padding: 8px 15px;
            margin: 5px 0;
            border-radius: 5px;
            border-left: 4px solid #667eea;
        }

        .all-predictions {
            margin-top: 20px;
        }

        .prediction-bar {
            display: flex;
            align-items: center;
            margin: 10px 0;
        }

        .prediction-label {
            width: 80px;
            font-weight: bold;
            font-size: 0.9rem;
        }

        .progress-bar {
            flex: 1;
            height: 20px;
            background: #e9ecef;
            border-radius: 10px;
            margin: 0 10px;
            overflow: hidden;
        }

        .progress-fill {
            height: 100%;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            border-radius: 10px;
            transition: width 0.5s ease;
        }

        .prediction-value {
            width: 50px;
            text-align: right;
            font-weight: bold;
            font-size: 0.9rem;
        }

        .ship-types {
            background: white;
            padding: 40px;
            border-radius: 15px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.2);
            margin-bottom: 40px;
        }

        .ship-types h2 {
            text-align: center;
            margin-bottom: 30px;
            color: #333;
            font-size: 2.5rem;
        }

        .ship-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 20px;
        }

        .ship-card {
            background: #f8f9fa;
            padding: 20px;
            border-radius: 10px;
            border-left: 5px solid #667eea;
            transition: transform 0.3s ease;
        }

        .ship-card:hover {
            transform: translateY(-5px);
        }

        .ship-card h3 {
            color: #333;
            margin-bottom: 10px;
        }

        .ship-card p {
            margin-bottom: 15px;
            line-height: 1.6;
        }

        .loading {
            display: none;
            text-align: center;
            padding: 20px;
        }

        .spinner {
            border: 4px solid #f3f3f3;
            border-top: 4px solid #667eea;
            border-radius: 50%;
            width: 40px;
            height: 40px;
            animation: spin 1s linear infinite;
            margin: 0 auto 10px;
        }

        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }

        .error-message {
            background: #f8d7da;
            color: #721c24;
            padding: 15px;
            border-radius: 5px;
            margin: 10px 0;
            border: 1px solid #f5c6cb;
            display: none;
        }

        @media (max-width: 768px) {
            .main-content {
                grid-template-columns: 1fr;
            }
            
            .header h1 {
                font-size: 2rem;
            }
            
            .ship-grid {
                grid-template-columns: 1fr;
            }
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🚢 Ship Classification AI</h1>
            <p>Advanced VGG16-powered ship identification system</p>
        </div>

        <div class="main-content">
            <div class="upload-section">
                <h2>Upload Ship Image</h2>
                <div class="upload-area" onclick="document.getElementById('fileInput').click()">
                    <div class="upload-icon">📁</div>
                    <p>Click here or drag and drop your ship image</p>
                    <small>Supports only JPG(max 10MB)</small>
                </div>
                <input type="file" id="fileInput" accept="image/*">
                <div class="preview-container">
                    <img id="imagePreview" alt="Preview">
                </div>
                <button class="btn" id="classifyBtn" disabled>Classify Ship</button>
                <div class="error-message" id="errorMessage"></div>
                <div class="loading" id="loading">
                    <div class="spinner"></div>
                    <p>Analyzing image...</p>
                </div>
            </div>

            <div class="result-section">
                <h2>Classification Result</h2>
                <div id="resultPlaceholder">
                    <p style="text-align: center; color: #666; padding: 40px;">
                        Upload an image to see classification results
                    </p>
                </div>
                <div class="result-content" id="resultContent">
                    <div class="prediction-result" id="predictionResult">
                        <h3 id="predictedClass"></h3>
                        <div class="confidence-score" id="confidenceScore"></div>
                    </div>
                    <div class="ship-details" id="shipDetails">
                        <h4>About this ship type:</h4>
                        <p id="shipDescription"></p>
                        <h4>Key Features:</h4>
                        <ul class="features-list" id="featuresList"></ul>
                    </div>
                    <div class="all-predictions" id="allPredictions">
                        <h4>All Predictions:</h4>
                    </div>
                </div>
            </div>
        </div>

        <div class="ship-types">
            <h2>Ship Types We Can Identify</h2>
            <div class="ship-grid">
                <div class="ship-card">
                    <h3>🚢 Cargo Ships</h3>
                    <p>Large vessels designed to carry goods and materials across oceans. These ships are the backbone of global trade.</p>
                    <ul class="features-list">
                        <li>Container storage</li>
                        <li>Heavy lifting equipment</li>
                        <li>Large cargo holds</li>
                    </ul>
                </div>
                <div class="ship-card">
                    <h3>⚓ Military Ships</h3>
                    <p>Naval vessels designed for defense and combat operations. Equipped with advanced weaponry and radar systems.</p>
                    <ul class="features-list">
                        <li>Weapon systems</li>
                        <li>Radar equipment</li>
                        <li>Armored hull</li>
                    </ul>
                </div>
                <div class="ship-card">
                    <h3>⛴️ Carriers</h3>
                    <p>Carriers are large ships designed to transport cargo, vehicles, or personnel across long distances, often supporting logistical and operational needs at sea.

                </p>
                    <ul class="features-list">
                        <li>Cargo holds</li>
                        <li>Vehicle decks</li>
                        <li>Catapult systems</li>
                    </ul>
                </div>
                <div class="ship-card">
                    <h3>🛳️ Cruise Ships</h3>
                    <p>Passenger ships designed for leisure travel and entertainment, featuring luxury amenities and accommodations.</p>
                    <ul class="features-list">
                        <li>Passenger cabins</li>
                        <li>Entertainment facilities</li>
                        <li>Dining areas</li>
                    </ul>
                </div>
                <div class="ship-card">
                    <h3>🛢️ Tanker Ships</h3>
                    <p>Specialized vessels designed to transport liquid cargo such as oil, chemicals, or liquefied natural gas.</p>
                    <ul class="features-list">
                        <li>Liquid storage tanks</li>
                        <li>Pumping systems</li>
                        <li>Safety equipment</li>
                    </ul>
                </div>
            </div>
        </div>
    </div>

    <script>
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
            // Validate file type
            if (!file.type.startsWith('image/')) {
                showError('Please select a valid image file');
                return;
            }

            // Validate file size (10MB)
            if (file.size > 10 * 1024 * 1024) {
                showError('File size must be less than 10MB');
                return;
            }

            selectedFile = file;
            
            // Show preview
            const reader = new FileReader();
            reader.onload = function(e) {
                const preview = document.getElementById('imagePreview');
                preview.src = e.target.result;
                preview.style.display = 'block';
            };
            reader.readAsDataURL(file);

            // Enable classify button
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

            // Hide placeholder and show result
            document.getElementById('resultPlaceholder').style.display = 'none';
            document.getElementById('resultContent').style.display = 'block';

            // Display main prediction
            document.getElementById('predictedClass').textContent = result.predicted_class + ' Ship';
            document.getElementById('confidenceScore').textContent = 
                `Confidence: ${(result.confidence * 100).toFixed(1)}%`;

            // Display ship details
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

            // Display all predictions
            const allPredictionsDiv = document.getElementById('allPredictions');
            const existingBars = allPredictionsDiv.querySelectorAll('.prediction-bar');
            existingBars.forEach(bar => bar.remove());

            if (result.all_predictions) {
                const sortedPredictions = Object.entries(result.all_predictions)
                    .sort(([,a], [,b]) => b - a);

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
    </script>
</body>
</html>
'''

@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE)

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'success': False, 'error': 'No file uploaded'}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({'success': False, 'error': 'No image selected'}), 400

    try:
        # Process image
        img_bytes = file.read()
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        img = img.resize((224, 224))

        # Prepare for model
        img_array = image.img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        # Make prediction
        predictions = model.predict(img_array)
        class_index = np.argmax(predictions, axis=1)[0]
        confidence = float(np.max(predictions))
        class_name = class_names.get(class_index, "Unknown")

        # Get all predictions
        all_predictions = {}
        for i, prob in enumerate(predictions[0]):
            ship_type = class_names.get(i, f"Class_{i}")
            all_predictions[ship_type] = float(prob)

        return jsonify({
            'success': True,
            'predicted_class': class_name,
            'confidence': confidence,
            'all_predictions': all_predictions,
            'description': ship_info.get(class_name, {})
        })

    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5001)