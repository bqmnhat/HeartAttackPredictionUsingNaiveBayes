<style>
 text-align: justify;
</style>

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

Below is a table showing the 15 data fields of the raw dataset, collected and compiled from the heart attack dataset on
