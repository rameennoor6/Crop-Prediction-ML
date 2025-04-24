from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load the trained model
model = pickle.load(open('model/crop_model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get values from form and convert to float
    input_features = [float(x) for x in request.form.values()]
    features_array = np.array([input_features])
    
    prediction = model.predict(features_array)
    
    return render_template('index.html', prediction_text=f"Recommended Crop: {prediction[0].title()}")

if __name__ == "__main__":
    app.run(debug=True)

