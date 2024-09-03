import pickle
from flask import render_template, Flask, request
import numpy as np
from sklearn.naive_bayes import GaussianNB
dbfile = open('HeartAttackPredictionFinalizedModel.sav', 'rb')
HeartAttackModel = pickle.load(dbfile)

feature_col_names = ['Age', 'Gender', 'Hypertension', 'Heart Disease',
                     'Body Mass Index (BMI)', 'Smoking Status', 'Alcohol Intake', 'Physical Activity', 'Previous Heart Disease', 'Family Stroke History',
                     'Dietary Habits', 'HDL Cholesterol Levels', 'LDL Cholesterol Levels', 'Stress Levels', 'Blood Pressure Thu', 'Blood Pressure Truong']

inp = [[]]

fcn_len = len(feature_col_names)

for i in range(fcn_len):
    print(feature_col_names[i] + ':')
    tmp = float(input())
    inp[0].append(tmp)
    

res = HeartAttackModel.predict(inp)

print(res)