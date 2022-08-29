from flask import Flask, request, jsonify, render_template

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
from keras.preprocessing import image
import os
from werkzeug.utils import secure_filename
from keras.preprocessing import image
import scipy
from keras.models import load_model
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler
import pickle

app = Flask(__name__)
model = load_model('model.h5')


@app.route('/')
def home():
  
    return render_template("index.html")
  
@app.route('/predict',methods=['GET'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    _pollution = float(request.args.get('pollution'))
    _dew = float(request.args.get('dew'))
    temp = float(request.args.get('temp'))
    pressure = float(request.args.get('pressure'))
    w_dir = float(request.args.get('w_dir'))
    w_speed = float(request.args.get('w_speed'))
    snow = float(request.args.get('snow'))
    rain = float(request.args.get('rain'))
    
    def parse(x):
        return datetime.strptime(x, '%Y %m %d %H')
    
    org_col_names=["No", "year","month", "day", "hour", "pm2.5", "DEWP","TEMP", "PRES", "cbwd", "Iws", "Is", "Ir"]
    col_names = ['pollution', 'dew', 'temp', 'pressure', 'w_dir', 'w_speed', 'snow', 'rain']

    dataset=pd.read_csv('pollution.csv')
    dataset_columns = dataset.columns.tolist()
    df = dataset.drop(['dew', 'temp', 'pressure', 'w_dir', 'w_speed', 'snow', 'rain'],axis = 1)
    pollution = np.array([[_pollution]])
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(pollution)
    prediction = model.predict(scaled_data)
    output = scaler.inverse_transform(prediction)

    prediction_text="pollution is predicted :{}".format(output)
    
    return render_template('index.html', prediction_text=prediction_text)


if __name__ == "__main__":
    app.run(debug=True)