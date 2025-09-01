# 🛳️ Titanic Survival Prediction with PySpark

## 📌 Project Overview
This project analyzes the famous **Titanic dataset** using **PySpark** to uncover survival patterns and build a machine learning model that predicts passenger survival.  
It demonstrates **EDA, feature engineering, and MLlib classification** in a scalable big-data pipeline.

---

## 📊 Dataset
The dataset contains passenger details with the following columns:  

- **Survived** → Survival status (0 = No, 1 = Yes)  
- **Pclass** → Passenger class (1, 2, 3)  
- **Name** → Passenger name  
- **Sex** → Gender  
- **Age** → Age in years  
- **Siblings/Spouses Aboard**  
- **Parents/Children Aboard**  
- **Fare** → Ticket price  

---

## 🔎 Exploratory Data Analysis (EDA)
Key insights discovered:  
- Women had a **74% survival rate** vs men at 19%.  
- First-class passengers had the **highest survival (63%)**.  
- Younger passengers had higher survival likelihood.  
- Very large families had **lower survival chances**.  

---

## ⚙️ Preprocessing & Feature Engineering
- Handled missing values in **Age** and **Fare**.  
- Encoded categorical variables (**Sex**).  
- Renamed family columns (`SibSp`, `Parch`) and created **FamilySize** feature.  
- Assembled numeric + categorical features into Spark ML vectors.  

---

## 🤖 Machine Learning Model
- **Algorithm**: Random Forest Classifier (PySpark MLlib)  
- **Training/Test Split**: 70/30  
- **Evaluation Metric**: AUC (Area Under ROC)  
- **Performance**: AUC ≈ **0.83**  

---

## 🛠️ Technologies
- **PySpark** (DataFrames, Spark SQL, MLlib)  
- **Python**  
- **PyCharm**  
- **Big Data ETL & Machine Learning**  

---


