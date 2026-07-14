# DiagnoVision

[![Python](https://img.shields.io/badge/Python-3.7+-blue.svg)](https://www.python.org/)
[![Flask](https://img.shields.io/badge/Flask-Framework-green.svg)](https://flask.palletsprojects.com/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-DL-orange.svg)](https://www.tensorflow.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-blue.svg)](https://scikit-learn.org/)

## Overview
DiagnoVision is a web-based healthcare application that uses Machine Learning (ML) and Deep Learning (DL) models to predict multiple diseases. It allows users to enter their medical parameters (clinical data) into forms or upload images, and the system predicts whether the person is healthy or may have a disease.

## Features
- **Diabetes Prediction**
- **Breast Cancer Prediction**
- **Heart Disease Prediction**
- **Kidney Disease Prediction**
- **Liver Disease Prediction**
- **Malaria Prediction** (Image-based)
- **Pneumonia Prediction** (Image-based)

## Architecture Overview
The app consists of a Python Flask backend serving HTML/CSS frontend pages. The ML models are built using `scikit-learn` for structured clinical data, while DL models use `TensorFlow/Keras` for image-based classifications (Malaria and Pneumonia).

## Installation

1. Clone the repository:
   ```bash
   git clone <repository_url>
   ```

2. Navigate to the project directory:
   ```bash
   cd DiagnoVision/Diagno_Vision
   ```

3. Install the dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Run the application:
   ```bash
   python app.py
   ```

5. Access the app locally:
   Open your browser and navigate to `http://localhost:5000`
