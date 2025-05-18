import os, traceback
import numpy as np
from flask import Flask, request, jsonify, render_template
from werkzeug.utils import secure_filename
from PIL import Image
import tensorflow as tf

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['MODEL_PATH']   = 'static/models/tbx11k_classifier.h5'

# Ensure upload folder exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Load your Keras model once
tf.get_logger().setLevel('ERROR')
model = tf.keras.models.load_model(app.config['MODEL_PATH'])

def predict_tb(img_path: str):
    """
    Open with Pillow, resize to 256×256, normalize, run model, and return
    (label, info, probabilities) for TB vs No TB.
    """
    # 1) Load & preprocess
    img = Image.open(img_path).convert('RGB')
    img = img.resize((256, 256), Image.BILINEAR)
    x   = np.array(img, dtype=np.float32) / 255.0
    x   = np.expand_dims(x, axis=0)  # shape (1,256,256,3)

    # 2) Predict
    raw_preds = model.predict(x)[0]
    preds = np.ravel(raw_preds)  # flatten to 1D

    # 3) Interpret outputs
    if preds.size == 1:
        # single-output sigmoid
        prob_tb = float(preds[0])
        prob_no = 1.0 - prob_tb
    elif preds.size >= 2:
        # two-class softmax
        prob_no, prob_tb = float(preds[0]), float(preds[1])
    else:
        raise ValueError(f"Unexpected prediction shape: {preds.shape}")

    # 4) Build outputs
    label = "TB Present" if prob_tb > prob_no else "TB absent"
    info  = f"TB: {prob_tb:.1%}, No TB: {prob_no:.1%}"
    return label, info, {'TB': prob_tb, 'No TB': prob_no}

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/results')
def results():
    return render_template('results.html')

@app.route('/api/predict', methods=['POST'])
def api_predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'Empty filename'}), 400

    # Save upload
    filename = secure_filename(file.filename)
    save_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(save_path)

    try:
        label, info, probs = predict_tb(save_path)
        return jsonify({'label': label, 'info': info, 'probabilities': probs})
    except Exception:
        traceback.print_exc()
        return jsonify({'error': 'Prediction failed. Check server logs for details.'}), 500

if __name__ == '__main__':
    # Requirements: pip install flask tensorflow pillow
    app.run(debug=True)
