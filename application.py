import pickle 
from flask import Flask, render_template,request,jsonify
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

application = Flask(__name__)
app=application

# import ridge regression nad standard scaler pickle
ridge_model=pickle.load(open("Models/ridge.pkl","rb"))
Standard_Scaler=pickle.load(open('Models/scaler.pkl','rb'))

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predictdata',methods=['GET','POST'])
def predict_datapoints():
    if request.method=='POST':
        temperature = float(request.form['Temperature'])
        rh = float(request.form['RH'])
        ws = float(request.form['ws'])
        rain = float(request.form['Rain'])
        ffmc = float(request.form['FFMC'])
        dmc = float(request.form['DMC'])
        isi = float(request.form['ISI'])
        classes = int(request.form['Classes'])
        region = int(request.form['Region'])
        new_data_scaled = Standard_Scaler.transform([[temperature, rh, ws, rain, ffmc, dmc, isi, classes, region]])
        result=ridge_model.predict(new_data_scaled)
        
        return render_template('home.html',results=result[0])
    else:
        return render_template('home.html')

if __name__ == '__main__':
    app.run(host="0.0.0.0" )
