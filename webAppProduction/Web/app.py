import pickle
from flask import render_template, Flask, request, jsonify, make_response, session, redirect, url_for
import numpy as np
from sklearn.naive_bayes import GaussianNB
import mysql.connector

app = Flask(__name__)
app.secret_key = '07122006'  # Change this to a random secret key

HAMod=pickle.load(open('HAP.sav','rb'))
DiabMod=pickle.load(open('DP.sav','rb'))

# Database connection configuration
db_config = {
    'host': 'localhost',
    'user': 'bqmnhat2313',
    'password': 'Dongden@2313',
    'database': 'diagnosis'
}

conn = mysql.connector.connect(**db_config)
cursor = conn.cursor()
cursor2 = conn.cursor()

@app.route('/')
def main():
    # Retrieve user's data from the database
    query = "SELECT * FROM healthcareinfo"
    cursor.execute(query)
    user_data = cursor.fetchall()
    return render_template('index.html', user_data=user_data)

@app.route('/prediction', methods=['POST', 'GET'])
def pred():
    age=float(request.form['age'])
    gender=float(request.form['gender'])
    hypertension=float(request.form['ht'])
    heart_disease=float(request.form['hd'])
    height=float(request.form['height'])
    if height>100:
        height=height/100
    weight=float(request.form['weight'])
    bmi=round(weight/(height*height),2)
    smoking=float(request.form['smoke'])
    alcohol=float(request.form['alcohol'])
    phyActivity=float(request.form['phy'])
    prevHD=float(request.form['phd'])
    famStroke=float(request.form['sh'])
    diet=float(request.form['diet'])
    hdl=float(request.form['hdl'])
    ldl=float(request.form['ldl'])
    stress=float(request.form['stress'])
    thu=float(request.form['bpthu']) * 10
    truong=float(request.form['bptruong']) * 10
    sleep=float(request.form['sleep'])
    heart_rate=float(request.form['hr'])

    # Inserting data into the database
    query = "INSERT INTO healthcareinfo (`age`, `gender`, `hypertension`, `heart_disease`, `bmi`, `smoking`, `alcohol`, `phyActivity`, `prevHD`, `famStroke`, `diet`, `hdl`, `ldl`, `stress`, `thu`, `truong`) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)"
    values = (age, gender, hypertension, heart_disease, bmi, smoking, alcohol, phyActivity, prevHD, famStroke, diet, hdl, ldl, stress, thu, truong)

    cursor.execute(query, values)
    conn.commit()

    inp=[[age, gender, hypertension, heart_disease, bmi, smoking, alcohol, phyActivity, prevHD, famStroke, diet, hdl, ldl, stress, thu, truong]]
    ha=HAMod.predict(inp)
    inp2=[[age, gender, thu, truong, heart_rate, famStroke, smoking, alcohol, prevHD, bmi, sleep, ha[0]]]
    dia=DiabMod.predict(inp2)
    if ha[0]==1:
        p1='You have a high chance of having a heart attack'
        c1='And'
        c2='But'
    else: 
        p1='You won\'t be having a heart attack soon'
        c1='But'
        c2='And'
    if dia[0]==1:
        p2= c1 + ' you have diabetes'
    else:
        p2= c2 + ' you don\'t have diabetes'
    return p1 + "<br/>" + p2

if __name__ == "__main__":
    app.run(debug=True)