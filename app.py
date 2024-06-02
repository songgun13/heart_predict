from flask import Flask, request, render_template  
import joblib  
import sklearn  
import pickle, gzip  
import pandas as pd  
import numpy as np  
app = Flask(__name__)  
model = joblib.load('heart_predict_test.pkl')  
@app.route('/')  
def home():  
 return render_template("home.html")  
 @app.route("/predict", methods=["POST"])  
 def predict():  
   Age = request.form["Age"]  
   Sex = request.form["Sex"]  
   ChestPainType = request.form["ChestPainType"] 
   RestingBP = request.form["RestingBP"]   
   Cholesterol = request.form["Cholesterol"]  
   FastingBS = request.form["FastingBS"]  
   RestingECG = request.form["RestingECG"]  
   MaxHR = request.form["MaxHR"]
   ExerciseAngina = request.form["ExerciseAngina"]
   Oldpeak = request.form["Oldpeak"]
   ST_Slope = request.form["ST_Slope"]
   arr = np.array([[Age, Sex, ChestPainType, RestingBP, Cholesterol, FastingBS, RestingECG, MaxHR, ExerciseAngina, Oldpeak, ST_Slope]])  
   pred = model.predict(arr)  
   if pred == 0:  
     res_val = "NO HEART PROBLEM"
   else:  
     res_val = "HEART PROBLEM"  
   return render_template('home.html', prediction_text='PATIENT HAS {}'.format(res_val))  
 if __name__ == "__main__":  
   app.run(debug=True)  