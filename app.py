import pickle
import numpy as np
import tensorflow
import keras
from keras.models import load_model
from keras.utils import CustomObjectScope
from keras.initializers import glorot_uniform
from flask import Flask, request, render_template, jsonify

with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

with CustomObjectScope({'GlorotUniform': glorot_uniform()}):
    model = load_model("conv_net.h5", compile=False)
    model._make_predict_function()

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('form.html')

@app.route('/predict', methods=['POST'])
def predict_headline():
    review = "".join([char for char in request.form.values()])
    review = tokenizer.texts_to_sequences([review])
    review = tensorflow.keras.preprocessing.sequence.pad_sequences(review, padding='post', truncating='post', maxlen=20)
    prediction = model.predict(review)
    probability = prediction[0][0]
    if prediction[0] >= 0.5:
        return render_template('form.html', prediction_text=f'This news article is FAKE! \nThe model is {np.round(probability * 100)}% confident that this is the headline of a fake news article.')
    elif prediction[0] < 0.5:
        return render_template('form.html', prediction_text=f'This news article is REAL! \nThe model is {np.round((1-probability) * 100)}% confident that this is the headline of a real news article.')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=1234)
