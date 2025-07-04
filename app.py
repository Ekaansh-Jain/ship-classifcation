from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image
import numpy as np
import io
import os

app = Flask(__name__)
CORS(app)

model = load_model('best_vgg16_model.h5')

class_names = {
    0: 'Cargo',
    1: 'Military', 
    2: 'Carrier',
    3: 'Cruise',
    4: 'Tanker'
}
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


@app.route('/')
def index():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'success': False, 'error': 'No file uploaded'}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({'success': False, 'error': 'No image selected'}), 400

    try:

        img_bytes = file.read()
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        img = img.resize((224, 224))

        img_array = image.img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        predictions = model.predict(img_array)
        class_index = np.argmax(predictions, axis=1)[0]
        confidence = float(np.max(predictions))
        class_name = class_names.get(class_index, "Unknown")

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