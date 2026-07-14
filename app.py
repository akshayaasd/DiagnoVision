from flask import Flask, render_template, request, flash, redirect
import pickle
import numpy as np
from PIL import Image
import logging
from tensorflow.keras.models import load_model

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Cache models
models = {}
logger.info("Loading models into memory...")
try:
    models['diabetes'] = pickle.load(open('models/diabetes.pkl', 'rb'))
    models['breast_cancer'] = pickle.load(open('models/breast_cancer.pkl', 'rb'))
    models['heart'] = pickle.load(open('models/heart.pkl', 'rb'))
    models['kidney'] = pickle.load(open('models/kidney.pkl', 'rb'))
    models['liver'] = pickle.load(open('models/liver.pkl', 'rb'))
    
    # Load DL models
    models['malaria'] = load_model('models/malaria.h5')
    models['pneumonia'] = load_model('models/pneumonia.h5')
    logger.info("Models loaded successfully.")
except Exception as e:
    logger.error(f"Error loading models: {e}")

def predict(values, dic):
    if len(values) == 8:
        model = models['diabetes']
    elif len(values) == 26:
        model = models['breast_cancer']
    elif len(values) == 13:
        model = models['heart']
    elif len(values) == 18:
        model = models['kidney']
    elif len(values) == 10:
        model = models['liver']
    else:
        raise ValueError(f"Unknown feature count: {len(values)}")
        
    values = np.asarray(values)
    return model.predict(values.reshape(1, -1))[0]

@app.route("/")
def home():
    return render_template('home.html')

@app.route("/diabetes", methods=['GET', 'POST'])
def diabetesPage():
    return render_template('diabetes.html')

@app.route("/cancer", methods=['GET', 'POST'])
def cancerPage():
    return render_template('breast_cancer.html')

@app.route("/heart", methods=['GET', 'POST'])
def heartPage():
    return render_template('heart.html')

@app.route("/kidney", methods=['GET', 'POST'])
def kidneyPage():
    return render_template('kidney.html')

@app.route("/liver", methods=['GET', 'POST'])
def liverPage():
    return render_template('liver.html')

@app.route("/malaria", methods=['GET', 'POST'])
def malariaPage():
    return render_template('malaria.html')

@app.route("/pneumonia", methods=['GET', 'POST'])
def pneumoniaPage():
    return render_template('pneumonia.html')

@app.route("/predict", methods = ['POST', 'GET'])
def predictPage():
    if request.method == 'POST':
        try:
            to_predict_dict = request.form.to_dict()
            if not to_predict_dict:
                raise ValueError("No data provided.")
                
            to_predict_list = list(map(float, list(to_predict_dict.values())))
            pred = predict(to_predict_list, to_predict_dict)
            return render_template('predict.html', pred=pred)
        except Exception as e:
            logger.error(f"Prediction error: {str(e)}")
            message = f"Please enter valid Data. Error: {str(e)}"
            return render_template("home.html", message=message)
    return render_template("home.html")

@app.route("/malariapredict", methods = ['POST', 'GET'])
def malariapredictPage():
    if request.method == 'POST':
        try:
            if 'image' not in request.files:
                raise ValueError("No image part in the request.")
                
            file = request.files['image']
            if file.filename == '':
                raise ValueError("No file selected.")
                
            img = Image.open(file)
            img = img.resize((36,36))
            img = np.asarray(img)
            img = img.reshape((1,36,36,3))
            img = img.astype(np.float64)
            
            model = models['malaria']
            pred = np.argmax(model.predict(img)[0])
            return render_template('malaria_predict.html', pred=pred)
        except Exception as e:
            logger.error(f"Malaria prediction error: {str(e)}")
            message = f"Please upload a valid Image. Error: {str(e)}"
            return render_template('malaria.html', message=message)
    return render_template('malaria.html')

@app.route("/pneumoniapredict", methods = ['POST', 'GET'])
def pneumoniapredictPage():
    if request.method == 'POST':
        try:
            if 'image' not in request.files:
                raise ValueError("No image part in the request.")
                
            file = request.files['image']
            if file.filename == '':
                raise ValueError("No file selected.")
                
            img = Image.open(file).convert('L')
            img = img.resize((36,36))
            img = np.asarray(img)
            img = img.reshape((1,36,36,1))
            img = img / 255.0
            
            model = models['pneumonia']
            pred = np.argmax(model.predict(img)[0])
            return render_template('pneumonia_predict.html', pred=pred)
        except Exception as e:
            logger.error(f"Pneumonia prediction error: {str(e)}")
            message = f"Please upload a valid Image. Error: {str(e)}"
            return render_template('pneumonia.html', message=message)
    return render_template('pneumonia.html')

if __name__ == '__main__':
    app.run(debug=True)