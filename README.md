# Heart Attack and Diabetes Prediction Model using Naive Bayes

- Author:
    - Nhat Bui Quoc Minh (Leader, A.I developer, web application developer, markdown writer)
    - An Nguyen Pham Quoc (A.I developer, markdown writer)
    - Choi Won Seok (web application developer)
    - Thinh Duong Ngoc (web application developer)
    - Phi Nguyen Nhat (markdown writer)
- Tran Dai Nghia Highschool for the Gifted - class of 12CTin (2021 - 2024)

Content
=================
* 1 [Introduction](#1-Introduction)
    * 1.1 [Diabetes Status](#11-Diabetes-Status)
    * 1.2 [Heart Attack Status](#12-Heart-Attack-Status)
    * 1.3 [Product Goals](#13-Product-Goals)
* 2 [Data](#2-Data)
    * 2.1 [Required Data Fields](#21-Required-Data-Fields)
    * 2.2 [Raw Data](#22-Raw-Data)
* 3 [Product Model Development](#3-Product-Model-Development)
    * 3.1 [Development Process Flowchart](#31-Development-Process-Flowchart)
    * 3.2 [Model Development](#32-Model-Development)
    * 3.3 [Model Training](#33-Model-Training)
    * 3.4 [Model Exportation](#34-Model-Exportation)
    * 3.5 [Web Application Development](#35-Web-Application-Development)
    * 3.6 [Overfitting and Underfitting Testing](#36-Overfitting-and-Underfitting-Testing)
* 4 [Conclusion](#4-Conclusion)
    * 4.1 [Achieved Results](#41-Achieved-Results)
    * 4.2 [Evaluation](#42-Evaluation)

## 1. Introduction

According to a 2012 survey, the diabetes rate in our country was 5.7%. However, by 2022, this rate had risen to 7.3%. This represents a dramatic increase of 1.6% in ten years, posing a significant health threat to many people.
<div style="text-align: center;" markdown="1">

***Chart showing the diabetes rate from 2012-2022 in our country***
![image](https://hackmd.io/_uploads/HyTlcPYSa.png)

</div>

Accompanying diabetes is a high risk of stroke, with approximately 200,000 cases annually, and a mortality rate of 20%.

With the aim of improving the quality of life and health of Vietnamese people as well as the global community, a heart attack and diabetes prediction model has been developed to enhance early detection and timely intervention for these two serious conditions. This diagnostic model can play a crucial role in identifying risks, diagnosing, and treating these diseases.

### 1.1 Diabetes Status 
#### 1.1.1 Vietnam

Before the 2000s, the diabetes rate for both men and women was below 5%. Women had a slightly higher rate, about 1% more than men, but this rate began to rise steadily, reaching approximately 5% by 2000 and then leveling off. For men, the rate continued to increase throughout the period and exceeded that of women by 2010. 
<div style="text-align: center;" markdown="1">

***Chart showing the increase in diabetes rates from 1960-2014 in our country***

![image](https://hackmd.io/_uploads/r14HyuKSp.png)

</div>

It is evident that the diabetes rate in both genders has been on the rise. Before 2000, the rate was higher in women, but after 2000, men became more affected.

This is a very rapid and alarming increase, necessitating immediate action.

Despite this, the situation of diagnosing and treating the disease in Vietnam remains bleak. Specifically, patients often fail to recognize the signs when the disease is mild or ignore the symptoms. Additionally, traveling to the hospital for regular check-ups and diagnoses often **takes a lot of time and effort**. This leads to **a majority of patients not being diagnosed or even treated.**

<div style="text-align: center;" markdown="1">

![image](https://hackmd.io/_uploads/S1X3QaYB6.png)

</div>

Based on the chart above, it can be seen that **the number of undiagnosed patients is overwhelmingly high at 68.9%**. This shows that many individuals in Vietnam are still not being diagnosed and treated in a timely manner. 

#### 1.1.2 Worldwide
Not only in Vietnam, but the situation of diabetes in many countries is also not optimistic:
<div style="text-align: center;" markdown="1">

![image](https://hackmd.io/_uploads/HJHnqaFS6.png)

</div>

Diabetes, once considered a European disease, is now spreading to Asia. Regions in Asia, particularly the Western Pacific, Southeast Asia, and the Middle East, have a large share, accounting for 68.7%. The Western Pacific region has the largest share with 206 million people. The predominance of diabetes cases in Asia may be due to high population, food contamination, etc.
<div style="text-align: center;" markdown="1">

![image](https://hackmd.io/_uploads/r1laZ0tB6.png)

</div>

It is evident that the number of people with diabetes is continuously increasing worldwide. It is projected to rise to 578 million by 2030 and 700 million by 2045.
### 1.2 Heart Attack Status
#### 1.2.1 Vietnam 

Heart attack and other cardiovascular diseases are among the leading causes of death, accounting for 33%.
<div style="text-align: center;" markdown="1">

![image](https://hackmd.io/_uploads/Sk29XCKST.png)

</div>

Heart attack is a serious condition; if not treated promptly to restore blood flow quickly, it can cause permanent heart damage and death.

In the past, this disease was commonly seen in the elderly, but recent statistics show that heart attacks are becoming more common among younger people. According to statistics from major hospitals, the rate of heart attack in young people has risen to 10.5%, with 1.8% being very young.

#### 1.2.2 Worldwide 

Globally, 2.5 million people die each year from heart attacks, with 25% dying during the acute phase of the disease. Another 5% – 10% die within the following year. Each year, about 635,000 new heart attack cases occur, 280,000 are recurrent attacks, and 150,000 are silent heart attacks. It is estimated that every 34 seconds a heart attack occurs, and every minute a person dies from it.

### 1.3 Product Goals:

From the current situation reflected by the statistics, we conclude that:

A more effective model for diagnosis and treatment is needed. Developing a heart attack and diabetes diagnosis model in Vietnam is a solution that opens the way to developing remote health examination systems for these two diseases in particular and for other diseases in general. Doctors will be provided with a platform for remote consultation, supported by artificial intelligence (AI) to assist in diagnosis. This saves time and effort for both patients and doctors, reducing the time spent traveling to hospitals and enabling health checks anywhere, anytime.

**The goal of developing** this product includes the following features:
- Account management and user information: requires a database
- **Support for heart attack and diabetes diagnosis: AI model**
- Allowing doctors to connect with patients for consultation
- Using smartwatches to measure necessary health parameters

The article below will focus on the artificial intelligence (AI) aspect supporting doctors in diagnosing diseases.

## 2. Data

### 2.1 Required Data Fields

Heart attack is a condition that occurs when one or both coronary arteries suddenly become blocked, leading to a situation where the heart muscle does not receive enough blood and may suffer necrosis. There may be one or several factors affecting blood circulation. For example, cholesterol in the blood can block blood vessels.

In addition, health deterioration symptoms often appear before a sudden heart attack. However, since these symptoms are not clearly related to the heart but affect it through other systems in the body, they are often ignored by patients. Therefore, to diagnose the likelihood of a heart attack in a patient, we need to collect numerous health parameters. For our AI model, these data fields will correspond to health metrics, including:

- **Biological** data fields: age, gender, height, weight, blood pressure, history of high blood pressure, history of heart disease, and cholesterol levels in the blood
- **Genetic** data fields: hereditary heart disease
- **Physical** data fields: level of physical activity
- **Other** data fields: smoking status, alcohol consumption status, dietary habits, and stress levels

Through the data fields used for heart attack diagnosis, we can also expand into predicting the likelihood of a patient having diabetes. This is due to two factors:
- "**Risk of heart attack"** is a significant factor in diagnosing diabetes but is difficult to collect from patients.
- Many symptoms of heart attacks overlap with those of diabetes, as evidenced by similar health metrics.

Thus, we can use the collected data and the predicted risk of a heart attack to predict the likelihood of diabetes in the patient.

### 2.2 Raw Data

Below is a table showing the 15 data fields of the raw dataset, collected and compiled from the heart attack dataset on Kaggle.com and the health dataset from the U.S. government on data.cdc.gov:

![DiabetesBeforePreprocess_NoMarks2](https://hackmd.io/_uploads/HkUacHoHa.png)

The 15 fields are:
- Age: Age
- Gender: Gender (0 for female, 1 for male)
- Hypertension: High blood pressure
- Heart Disease: Heart disease condition
- Average Glucose Level: Blood glucose level
- BMI: Body Mass Index
- Smoking Status: Smoking status
- Alcohol Intake: Alcohol consumption status
- Physical Activity: Frequency of physical activity 
- Previous Heart Disease: History of heart disease
- Family Stroke History: Family history of stroke
- Dietary Habits: Dietary habits ("green" eating habits)
- Stress Levels: Stress levels
- Blood Pressure Levels: Blood pressure
- Cholesterol Levels: Cholesterol levels

## 3 Product Model Development

### 3.1 Development Process Flowchart:
![HeartAttackAndDiabetes.drawio](https://hackmd.io/_uploads/HJpjsI5S6.png)

### 3.2 Model Development

For our problem, we will process the data based on the relationships between data fields to eliminate unstable fields or split fields that are parents of many smaller sub-fields to facilitate model training.

Additionally, we will train multiple models and assess the accuracy of all models to select the strongest model or determine the appropriate model development direction for the product.

#### 3.2.1 Data Preprocessing
In the initial data fields, we notice that the two fields "Blood pressure level" and "Cholesterol Level" contain two parameters (sub-fields) within them. This will complicate the extraction of necessary features for model training. Therefore, these two data fields need to be separated into the green-bordered fields as shown below:

![DiabetesBeforePreprocess_HaveAverageGlucose](https://hackmd.io/_uploads/By0r7GcBT.png)

Furthermore, the impact of the health metric recorded in the "Average Glucose Level" data field will correspond to the combination of the "Gender" data field and the "HDL Cholesterol Level" and "LDL Cholesterol Level" fields. This is also demonstrated in the publication:
<div style="text-align: center;" markdown="1">

![AGLCorrelationWithGenderAndCholLevel_Proof](https://hackmd.io/_uploads/BJrJfzqra.png)
*(Source: https://www.researchgate.net/publication/341653979_Correlating_the_Cholesterol_Levels_to_Glucose_for_Men_and_Women)*

</div>

Based on this, we can completely replace the "Average Glucose Level" field by using the "Gender" field and the two "Cholesterol" fields for diagnosis:

![DiabetesBeforePreprocess_AverageGlucoseCorrelationWithGenderAndCholLevels](https://hackmd.io/_uploads/SJuhXfqHT.png)

After fine-tuning the data fields, noticing that the data consists entirely of numerical data, we need to identify the distribution characteristics of the data to build an appropriate AI model.

However, analyzing the data to highlight distribution characteristics using detailed numeric tables is very challenging. Therefore, we need to model and graph the data fields, examine each one individually, and then group the distribution characteristics to choose a suitable Naive Bayes model (Gaussian or Multinomial):

![CodePlotAll](https://hackmd.io/_uploads/HysKLf5Ba.png)
![AllGraphs1](https://hackmd.io/_uploads/rkvX_G5Hp.png)
![AllGraphs2](https://hackmd.io/_uploads/BkCXdzqST.png)
![AllGraphs3](https://hackmd.io/_uploads/BJmV_fqBa.png)
![AllGraphs4](https://hackmd.io/_uploads/B1UNufqrT.png)

Through the above graphs, we can see that the data fields' distributions include two types:
- Discrete Distribution (Multinomial):
    - Age
    - Average Glucose Level
    - BMI
    - Smoking Status
    - Alcohol Intake
    - Physical Activity
    - Dietary Habits
    - Stress Levels
    - Blood Pressure fields
    - Cholesterol fields
- Binary Distribution (Binomial):
    - Gender
    - Hypertension
    - Heart Disease
    - Previous Heart Disease
    - Family Stroke History

#### 3.2.2 Choosing the Appropriate AI Model: Naive Bayes

From the results above, we conclude that developing a diagnosis model using **Naive Bayes MultinomialNB** is an appropriate approach for the dataset we have collected.

Indeed, when we trained and tested the preprocessed dataset with various models (Naive Bayes Multinomial, Random Forest, Logistic Regression, and SVM), we found that Naive Bayes Multinomial had an **accuracy score above the average (Mean)**:
![Evaluation](https://hackmd.io/_uploads/Byp6-P9Sp.png)

#### 3.2.3 Basis for Applying Naive Bayes
Bayesian classification is a part of machine learning and statistics that allows the prediction of the probability that a data point belongs to a specific class. It is based on Bayes' theorem to estimate the probability of an event based on prior information:

![naive](https://hackmd.io/_uploads/rJbg10tSp.png)

Although there are many types and variants, Bayesian classification algorithms are widely applied in computer science, particularly in artificial intelligence.

#### Necessary Parameters

In the Naive Bayes algorithm, some important parameters are required to build the model and perform classification:

 - **Prior probability** of the classes: estimated based on the frequency of occurrence of the classes in the training dataset.
 - **Features** of the dataset
- **Conditional probability of the features for each class**: Conditional probability of each feature corresponding to each class in the classification problem.
 - Smoothing parameter (Laplace Smoothing): **Especially in Multinomial Naive Bayes**, Laplace smoothing is often used to avoid the conditional probability of zero in cases where a feature does not appear in the training set.

#### Applying Naive Bayes to the Problem

Applying the Naive Bayes algorithm to the heart attack and diabetes diagnosis problem, we realize that:

- **Prior probability** in the problem is the frequency of occurrence of patients with the disease and those without the disease in the dataset. For example, in the heart attack dataset:

<div style="text-align: center;" markdown="1">

![PercentageOfClassesInDataset](https://hackmd.io/_uploads/SkWwbIsBp.png)

</div>

- **Features** are the health metrics
- **Conditional probability of the features for each class** is the conditional probability of each type of health metric corresponding to the condition of having or not having the disease. For example:

<div style="text-align: center;" markdown="1">

![AlcoholIntakeWhenPatientHaveHeartAttack](https://hackmd.io/_uploads/B1uFPIjBp.png)

</div>

### 3.3 Model Training

First, we need to split the large dataset into two subsets: train (for training) and test (for testing) so that the train:test ratio is as close to 7:3 as possible (as demonstrated in the paper "Optimal Ratio for Data Splitting" by V. Roshan Joseph):
<div style="text-align: center;" markdown="1">

![Split1](https://hackmd.io/_uploads/HyNqQPcB6.png)
![Split2](https://hackmd.io/_uploads/r1vqQP5rp.png)

</div>

### 3.4 Model Exportation

After successfully training the model, the next step is to export the model for use in the application.

Before being exported, models only exist within Python programs. When the Python program is closed, the model's parameters are lost, and **we cannot use these models in other programs**. **Therefore, a method is needed to export and store the model from Python code to the computer's hard drive.**

This method is called **serialization**. When we serialize a Python object, we convert this object from a complex data type into a **byte stream**. This byte stream is established similarly to encoding data. However, the byte stream must be set up in a special sequence to be **decoded** back into an AI model when needed.

<div style="text-align: center;" markdown="1">

***Serialization of a Python object diagram***
![Serialization](https://hackmd.io/_uploads/HJTXuvqBp.png)

</div>

Python supports many libraries capable of serializing Python objects. Among them, the **pickle** library is the most commonly used due to its simple, convenient, and fast functions.

There are two functions to note:

1. **pickle.dump(model, file)**: Serializes, creates a byte stream of the "model," and saves it into "file"

<div style="text-align: center;" markdown="1">

![dumping](https://hackmd.io/_uploads/H11oIvcSp.png)

</div>

2. **load(file)**: Returns a Python object deserialized from the byte stream in "file"

<div style="text-align: center;" markdown="1">

![Load](https://hackmd.io/_uploads/H12pIvcSa.png)

</div>

After loading, the model can be used with the command **model.predict(data)**, just as in the Python program.

### 3.5 Web Application Development

The web app product integrates the AI model, which can diagnose heart attack and diabetes, and is developed in a way that allows the integration of additional features, aiming to expand into a remote health examination platform in the future.

Below is the web app model:
<div style="text-align: center;" markdown="1">

![WebAppDiagram.drawio](https://hackmd.io/_uploads/HyDDAuiB6.png)

</div>

Web app consists of three main components:
- **User interface including**: patient interface and doctor interface (future development)
- **Server with database**
- **Health monitoring devices** (future development)

### 3.6 Overfitting and Underfitting Testing

#### 3.6.1 Recap
A model is considered underfitting if it does not fit well with both the training dataset and new data when predicting. This could be due to the model not being complex enough to capture the dataset's characteristics.

A model is considered overfitting if it fits the training set very well but does not generalize to new data. This could be due to the model being too complex or a lack of sufficient data for evaluation.

#### 3.6.2 Experiment
To test these properties, we can manipulate model parameters or data values to train (Data Manipulation). In this article, we conduct two experiments:
- **Data Specialization:**
    - We converted all values in the **[Diagnosis]** field of the training set to **"1"**.

    <div style="text-align: center;" markdown="1">

    ![image](https://hackmd.io/_uploads/SkCeat2ST.png)

    </div>
    
    - This leads to the model predicting that all input cases have a **"Diagnosis"** of **"1"** because the dataset is specialized. However, when the new dataset (Testing Data) is introduced into the model, the accuracy drops significantly (due to the new dataset containing both "0" and "1").
    <div style="text-align: center;" markdown="1">

    ![image](https://hackmd.io/_uploads/rkjh6YhST.png)

    </div>

    **=> The model is overfitting.**
    
- **Reducing Data Fields:**
    - We reduced the number of parameters (15 fields mentioned above) used to predict **[Diagnosis]**. From 15 fields, we reduced it to one (the **[Age]** field).

    <div style="text-align: center;" markdown="1">

    ![image](https://hackmd.io/_uploads/BygdRYnrp.png)

    </div>
    
    - This results in the model being too simplistic due to the lack of parameters. Consequently, the accuracy when training the model with both the training and new datasets is very low (<60%).
    <div style="text-align: center;" markdown="1">

    ![image](https://hackmd.io/_uploads/SkyGJ52rT.png)

    </div>

    **=> The model is underfitting.**

## 4 Conclusion

### 4.1 Achieved Results

Currently, the product has completed two functions:
- Account management and user information: requires a database
- **Support for heart attack and diabetes diagnosis: AI model**

<div style="text-align: center;" markdown="1">

**Home page interface**
![Homepage](https://hackmd.io/_uploads/HybYVYjHa.png)

**Login interface**
![Login](https://hackmd.io/_uploads/H10pNKsSp.png)

**Diagnosis interface**
![Predict](https://hackmd.io/_uploads/BJ49O5nS6.png)
![Predict2](https://hackmd.io/_uploads/H1uq_92Ba.png)
**Graph displaying health parameters (future development to show health parameters over time)**
![Predict3](https://hackmd.io/_uploads/SJncd53B6.png)

</div>

### 4.2 Evaluation:

Although the doctor interface and the connection between doctors and patients are still under development, they are highly feasible due to the comprehensive data structure already in place. The product can eventually include video call support and integrate additional smart health monitoring devices to measure health indicators directly at home.

Once complete, the product will serve as a foundation for an intelligent remote healthcare system integrated with artificial intelligence technology and smartwatches to measure health indicators. This will create a positive change in people's healthcare habits, encouraging more frequent participation in healthcare thanks to the product's efficiency.
